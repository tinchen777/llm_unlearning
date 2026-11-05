"""Stage 2 of W_down training: end-to-end refinement with an output-level KL term.

Stage 1 (`METHOD._common.editing`) fits W_down of every layer by a LOCAL regression on recorded (x, target) tokens: it never sees what the
edited model says. Here the W_down of the edited layers are trained TOGETHER through the whole model (they start from the stage-1 solution):

    loss = forget_weight * rel_MSE(forget MLP outputs, original + shift)        (keeps the redirection of the activations)
         + retain_weight * rel_MSE(retain / neighbour MLP outputs, original)      (keeps the rest)
         + kl_weight     * KL( p_orig(. | neighbour prompt, neighbour answer prefix) || p_edited(. | forget prompt, same prefix) )

The KL is on the first `n_tokens` answer tokens: the edited model, given a forget prompt and the answer of a (train) neighbour of that
question, should predict that answer like the ORIGINAL model does for the neighbour prompt itself. Only the original model is needed
(`ref_model`, a frozen copy taken before any edit); no retain model.
`rel_MSE` = MSE / mean(target^2) of the layer, so the three terms have comparable scale.
"""
from __future__ import annotations
import logging
import random
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from model.arch import get_out_proj

logger = logging.getLogger("main(w_down)")


def build_kl_pairs(
    forget_ds: Any,
    neighbor_full_ds: Any,
    qa_per_author: int,
    author_offset: int,
    n_tokens: int,
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    """Pair every TRAIN neighbour question with the forget question it paraphrases: `{forget row: [pair, ...]}`.

    Pairing = same author (`author_id`) and same question number (`qa_id`) -> forget row `(author - author_offset) * qa_per_author + qa_id`.
    If the neighbour csv has `source_question` it must equal the forget question (a wrong row order is an error, not a silent mismatch).
    A pair: `forget_prompt` (ids of the forget prompt, generation mode), `neighbor_prompt` (ids), `answer` (the first `n_tokens` answer ids
    of the neighbour). Neighbours with a shorter answer are skipped.
    """
    question_key = getattr(forget_ds, "question_key", "question")
    groups: Dict[int, List[Dict[str, torch.Tensor]]] = {}
    for j in range(len(neighbor_full_ds)):
        nb = neighbor_full_ds[j]
        author, qa = int(nb["author_id"]), int(nb["qa_id"])
        fi = (author - author_offset) * qa_per_author + qa
        if not 0 <= fi < len(forget_ds):
            raise ValueError(f"neighbour row {j} (author {author}, qa_id {qa}) -> forget row {fi}, outside 0..{len(forget_ds) - 1}.")
        source = nb.get("source_question", None)
        if source is not None and str(source).strip() != str(forget_ds.raw_data[fi][question_key]).strip():
            raise ValueError(
                f"neighbour row {j} (author {author}, qa_id {qa}) paraphrases `{source}`, but forget row {fi} is "
                f"`{forget_ds.raw_data[fi][question_key]}`: the order of the forget split does not match the neighbour csv."
            )
        ids, labels = torch.as_tensor(nb["input_ids"]), torch.as_tensor(nb["labels"])
        prompt_len = int((labels == -100).sum())
        answer = ids[prompt_len:prompt_len + n_tokens]
        if len(answer) < n_tokens:
            continue
        groups.setdefault(fi, []).append({
            "forget_prompt": torch.as_tensor(forget_ds[fi]["input_ids"]),
            "neighbor_prompt": ids[:prompt_len],
            "answer": answer,
        })
    if not groups:
        raise ValueError("No usable (forget question, neighbour answer) pair was found for the KL term.")
    return groups


def _left_pad(seqs: Sequence[torch.Tensor], pad_id: int, device: Any) -> Dict[str, torch.Tensor]:
    """Left-padded `input_ids`, `attention_mask` and mask-aware `position_ids` (so that the padding does not move the positions)."""
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = s
        mask[i, n - len(s):] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "position_ids": pos.to(device)}


def _answer_logits(model: Any, inputs: Mapping[str, torch.Tensor], n_tokens: int) -> torch.Tensor:
    """Logits that predict the last `n_tokens` tokens (the answer prefix), [bsz, n_tokens, vocab]; the sequences are left padded, so
    they end at the same index. Only these positions go through the LM head."""
    out = model(**inputs, logits_to_keep=n_tokens + 1)
    return out.logits[:, :-1].float()


@contextmanager
def trainable_w_down(model: Any, layers: Sequence[int], init: Mapping[int, torch.Tensor]):
    """Make W_down of `layers` trainable through hooks: fp32 master weights `w[layer]` (start: `init`) replace the output of the
    projection (`x @ w.T`, cast back to the model dtype); the MLP outputs are kept in `captured[layer]` (with grad). Removes the hooks on exit."""
    weights: Dict[int, torch.nn.Parameter] = {}
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for _l in layers:
        weights[_l] = torch.nn.Parameter(init[_l].detach().to(device=model.device, dtype=torch.float32).clone())

        def hook(module, inputs, output, _l=_l):
            out = F.linear(inputs[0].float(), weights[_l]).to(output.dtype)
            captured[_l] = out
            return out

        handles.append(get_out_proj(model, _l).register_forward_hook(hook))
    try:
        yield weights, captured
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_mlp_out(model: Any, layers: Sequence[int]):
    """Keep the MLP outputs of `layers` in a dict during the forward passes inside the block."""
    captured: Dict[int, torch.Tensor] = {}
    handles = [
        get_out_proj(model, _l).register_forward_hook(lambda m, i, o, _l=_l: captured.__setitem__(_l, o))
        for _l in layers
    ]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _rel_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over the real tokens, divided by the mean square of the target (a relative error)."""
    m = mask.bool().unsqueeze(-1)
    pred, target = pred.float()[m.expand_as(pred)], target.float()[m.expand_as(target)]
    return F.mse_loss(pred, target) / target.pow(2).mean().clamp_min(1e-12)


def refine_w_down(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    layers: Sequence[int],
    init_weights: Mapping[int, torch.Tensor],
    forget_batches: Callable[[], Any],
    retain_batches: Callable[[], Any],
    shift_fn: Callable[[Mapping[str, Any], int, torch.Tensor], torch.Tensor],
    kl_groups: Mapping[int, List[Dict[str, torch.Tensor]]],
    cfg: Mapping[str, Any],
    forget_weight: float,
    retain_weight: float,
    seed: int,
) -> Dict[str, Any]:
    """Train the W_down of `layers` end to end (see the module docstring). `forget_batches()` / `retain_batches()` return a FRESH iterable
    of batches (`input_ids`, `attention_mask`, `labels`, `index`) over prompt + answer tokens. The best-effort result: the trained weights
    (fp32, cpu) and a history. The weights are NOT written into `model`."""
    steps, lr = int(cfg["steps"]), float(cfg["lr"])
    kl_w, n_tokens, kl_bs = float(cfg["weight"]), int(cfg["n_tokens"]), int(cfg["batch_size"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rng = random.Random(seed)
    device = model.device
    for p in model.parameters():
        p.requires_grad_(False)
    forget_list, retain_list = list(forget_batches()), list(retain_batches())
    fidx = sorted(kl_groups)
    history: List[Dict[str, float]] = []

    def to_dev(batch):
        return {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}

    with trainable_w_down(model, layers, init_weights) as (weights, edited_out), capture_mlp_out(ref_model, layers) as ref_out:
        optimizer = torch.optim.AdamW(list(weights.values()), lr=lr, weight_decay=0.0)
        for step in range(steps):
            fb, rb = rng.choice(forget_list), rng.choice(retain_list)
            terms = {"forget": 0.0, "retain": 0.0}
            optimizer.zero_grad()

            # (a) activations: forget tokens must move by the shift, retain / neighbour tokens must stay
            loss_f = loss_r = 0.0
            for kind, batch in (("forget", fb), ("retain", rb)):
                inputs = to_dev(batch)
                with torch.no_grad():
                    ref_model(**inputs)
                model(**inputs)
                mask = inputs["attention_mask"]
                for _l in layers:
                    target = ref_out[_l].float()
                    if kind == "forget":
                        target = target + shift_fn(batch, _l, mask).to(target)
                    term = _rel_mse(edited_out[_l], target, mask) / len(layers)
                    if kind == "forget":
                        loss_f = loss_f + term
                    else:
                        loss_r = loss_r + term
                part = (forget_weight * loss_f) if kind == "forget" else (retain_weight * loss_r)
                part.backward()
                terms[kind] = float((loss_f if kind == "forget" else loss_r).detach())
                edited_out.clear()
                ref_out.clear()

            # (b) the output distribution of the first answer tokens
            loss_kl = torch.zeros((), device=device)
            if kl_w > 0:
                picked = [rng.choice(kl_groups[rng.choice(fidx)]) for _ in range(kl_bs)]
                edited_in = _left_pad([torch.cat([p["forget_prompt"], p["answer"]]) for p in picked], pad_id, device)
                ref_in = _left_pad([torch.cat([p["neighbor_prompt"], p["answer"]]) for p in picked], pad_id, device)
                with torch.no_grad():
                    ref_logp = F.log_softmax(_answer_logits(ref_model, ref_in, n_tokens), dim=-1)
                edited_logp = F.log_softmax(_answer_logits(model, edited_in, n_tokens), dim=-1)
                loss_kl = (ref_logp.exp() * (ref_logp - edited_logp)).sum(-1).mean()
                (kl_w * loss_kl).backward()
                edited_out.clear()
                ref_out.clear()

            optimizer.step()
            history.append({"step": step + 1, "forget_rel_mse": terms["forget"], "retain_rel_mse": terms["retain"], "kl": float(loss_kl.detach())})
            if (step + 1) % 10 == 0 or step == 0:
                logger.info(f"KL refine {step + 1}/{steps}  forget_rel_mse={terms['forget']:.4g}  retain_rel_mse={terms['retain']:.4g}  kl={float(loss_kl.detach()):.4g}")
        new_weights = {_l: w.detach().cpu() for _l, w in weights.items()}
    return {"weights": new_weights, "history": history}
