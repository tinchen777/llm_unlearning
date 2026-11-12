"""Stage 2 of W_down training: end-to-end refinement with an output-level KL term.

Stage 1 (`METHOD._common.editing`) fits W_down of every layer by a LOCAL regression on recorded (x, target) tokens: it never sees what the
edited model says. Here the W_down of the edited layers are trained TOGETHER through the whole model (they start from the stage-1 solution):

    loss = forget_weight * rel_MSE(forget MLP outputs, original + shift)        (keeps the redirection of the activations)
         + retain_weight * rel_MSE(retain / neighbour MLP outputs, original)      (keeps the rest)
         + kl_weight     * KL( p_orig(. | neighbour prompt, neighbour answer prefix) || p_edited(. | forget prompt, same prefix) )

The KL is on the first `n_tokens` answer tokens: the edited model, given a forget prompt and the answer of a (train) neighbour of that
question, should predict that answer like the ORIGINAL model does for the neighbour prompt itself. Only the original model is needed
(`ref_model`, a frozen copy taken before any edit); no retain model.
Optional (`forget_prob_weight` mu > 0; modes hinge | match | mean | neighbor, see `forget_prob_penalty`): + mu * relu(log p_true - log `forget_prob_target`), p_true = the (teacher forced) probability of the TRUE forget answer,
a hinge that pushes it down to the retain model's level (~0.17) and not beyond: the KL above shapes the beginning of the answer like a neighbour's but does not
lower the probability of the true answer tokens (the evaluation measures that).
`rel_MSE` = MSE / mean(target^2) of the layer, so the three terms have comparable scale.
"""
from __future__ import annotations
import logging
import math
import random
import re
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from model.arch import get_out_proj

logger = logging.getLogger("main(w_down)")


def name_token_mask(answer_ids: Sequence[int], name: str, tokenizer: Any) -> torch.Tensor:
    """1 for the answer tokens to keep in the KL, 0 for those that spell the neighbour's NAME (the whole name, the first name and the last name, with and
    without a leading space): the teacher reads the name in the neighbour question and copies it, which the student (who reads the forget question) can only
    learn by heart, so the student ends up using the names of the training neighbours (`paper_notes/8` §2.3)."""
    ids = [int(x) for x in answer_ids]
    keep = torch.ones(len(ids))
    parts = [name] + [p for p in re.split(r"[\s\-]+", name) if len(p) > 2]
    for part in parts:
        for variant in (part, " " + part):
            tok = list(tokenizer.encode(variant, add_special_tokens=False))
            if not tok:
                continue
            for i in range(len(ids) - len(tok) + 1):
                if ids[i:i + len(tok)] == tok:
                    keep[i:i + len(tok)] = 0.0
    return keep


def build_kl_pairs(
    forget_ds: Any,
    neighbor_full_ds: Any,
    qa_per_author: int,
    author_offset: int,
    n_tokens: int,
    neighbors_per_question: Optional[int] = None,
    tokenizer: Any = None,
    mask_neighbor_name: bool = False,
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    """Pair every TRAIN neighbour question with the forget question it paraphrases: `{forget row: [pair, ...]}`.

    Pairing = same author (`author_id`) and same question number (`qa_id`) -> forget row `(author - author_offset) * qa_per_author + qa_id`.
    If the neighbour csv has `source_question` it must equal the forget question (a wrong row order is an error, not a silent mismatch).
    A pair: `forget_prompt` (ids of the forget prompt, generation mode), `neighbor_prompt` (ids), `answer` (the first `n_tokens` answer ids
    of the neighbour), `mask` ([n_tokens], 0 = left out of the KL). Neighbours with a shorter answer are skipped.

    `neighbors_per_question` m: every forget question keeps only m of the neighbours of its author, chosen by a fixed rotation over the `train_rank` order
    (forget row f starts at rank position f mod R), so the student fits ONE (or m) neighbour(s) per question instead of the mixture of all of them, and
    different questions get different neighbours. None = all of them. `mask_neighbor_name`: leave the neighbour's name tokens (`neighbor_name` of the csv) out of the KL.
    """
    question_key = getattr(forget_ds, "question_key", "question")
    rows: Dict[int, List[Tuple[int, int, Dict[str, torch.Tensor]]]] = {}
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
        mask = torch.ones(n_tokens)
        if mask_neighbor_name:
            if tokenizer is None or "neighbor_name" not in nb:
                raise ValueError("mask_neighbor_name needs the tokenizer and the `neighbor_name` column of the neighbour csv (extra_keys / metadata_keys of data.neighbor).")
            mask = name_token_mask(answer.tolist(), str(nb["neighbor_name"]), tokenizer)
        rank = int(nb["train_rank"]) if "train_rank" in nb else j
        rows.setdefault(fi, []).append((rank, j, {
            "forget_prompt": torch.as_tensor(forget_ds[fi]["input_ids"]),
            "neighbor_prompt": ids[:prompt_len],
            "answer": answer,
            "mask": mask,
        }))
    groups: Dict[int, List[Dict[str, torch.Tensor]]] = {}
    for fi, items in rows.items():
        items.sort(key=lambda t: (t[0], t[1]))
        if neighbors_per_question is not None:
            m, r = int(neighbors_per_question), len(items)
            items = [items[(fi + t) % r] for t in range(min(m, r))]
        groups[fi] = [x[2] for x in items]
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


def retain_answer_kl(ref_logits: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean over the ANSWER tokens (labels != -100) of KL(ref || edited) of the next-token distribution (float32; only the answer positions go through the softmax)."""
    mask = labels[:, 1:] != -100
    if not bool(mask.any()):
        return logits.new_zeros((), dtype=torch.float32)
    ref_lp = torch.log_softmax(ref_logits[:, :-1][mask].float(), dim=-1)
    lp = torch.log_softmax(logits[:, :-1][mask].float(), dim=-1)
    return (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean()


@torch.no_grad()
def estimate_unseen_prob(model: Any, batches: Any) -> float:
    """The mean `forget_Q_A_Prob`-style probability (per sample exp(mean token log-prob of the answer), averaged over samples) that `model` gives to the
    answers of data it has NOT seen (e.g. the TOFU holdout authors): the probability of an answer to an unseen question. Needs only the ORIGINAL model and
    unseen questions, no retain model: on TOFU it is ~ the retain model's forget probability (1B 0.161 vs 0.169, 3B 0.159 vs 0.178, 8B 0.130 vs 0.139)."""
    probs = []
    device = model.device
    for batch in batches:
        inputs = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        logits = model(**inputs).logits
        _, p = forget_prob_penalty(logits, inputs["input_ids"], batch["labels"].to(device), 1.0, per_sample=True)
        probs.extend(p.tolist())
    return float(sum(probs) / len(probs))


def forget_prob_penalty(logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor, target: Any, mode: str = "hinge",
                        per_sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalty on the probability of the TRUE answer: per sample p_i = exp(mean token log-prob of the answer tokens) (the `forget_Q_A_Prob` of the evaluation,
    teacher forced). `target`: a number (the level of a model that never saw the answers, ~0.17) or, for `hinge` / `match` / `neighbor`, a tensor [batch] with one level per sample.
    Modes (gap_i = log p_i - log target_i):
      hinge     mean_i relu(gap_i)         only while MORE likely than the target (overshoots: target 0.17 -> 0.08)
      match     mean_i gap_i^2             every sample towards ITS target from either side; with ONE number as target all samples are pulled to the same value
                                           (the spread of log p_i collapses: sd 0.04-0.4 where an unseen question gives 0.8-1.0, and the loss-based MIA sees it)
      mean      (log mean_i p_i - log target)^2   only the BATCH MEAN of the probabilities is matched: the natural spread between the samples is kept
      neighbor_scaled  `neighbor` with the levels multiplied by one factor so that their mean is the level of unseen data (w_down.py; the neighbour answers are
                the less probable text, mean 0.09 instead of 0.16 on 1B, which over-forgets: Prob 0.09, mia 0.25 vs 0.48)
      neighbor  mean_i gap_i^2 with a per-sample `target` tensor = the original model's probability of the NEIGHBOUR answers of that question (`neighbor_answer_probs`):
                the target itself has the spread of an unseen question
    Returns (loss, mean p of the batch)."""
    mask = labels[:, 1:] != -100
    # only the answer positions go through the (float32) log-softmax over the vocabulary: that is what costs the memory
    sel = torch.log_softmax(logits[:, :-1][mask].float(), dim=-1).gather(-1, input_ids[:, 1:][mask].unsqueeze(-1)).squeeze(-1)
    sums = torch.zeros(logits.shape[0], device=logits.device, dtype=sel.dtype).index_add(0, mask.nonzero()[:, 0], sel)
    mean_logp = sums / mask.sum(1).clamp_min(1)
    if per_sample:
        return mean_logp.new_zeros(()), mean_logp.exp().detach()
    if mode not in ("hinge", "match", "mean", "neighbor"):
        raise ValueError(f"forget_prob_mode must be `hinge`, `match`, `mean` or `neighbor`, got `{mode}`.")
    if torch.is_tensor(target):
        log_target = torch.log(target.to(mean_logp))
    else:
        if mode == "neighbor":
            raise ValueError("forget_prob_mode `neighbor` needs one target per sample (a tensor), not a number.")
        log_target = math.log(float(target))
    if mode == "mean":
        if torch.is_tensor(target) and target.numel() > 1:
            log_target = log_target.mean()
        loss = (torch.logsumexp(mean_logp, dim=0) - math.log(mean_logp.shape[0]) - log_target).pow(2)
    else:
        gap = mean_logp - log_target
        loss = F.relu(gap).mean() if mode == "hinge" else gap.pow(2).mean()
    return loss, mean_logp.exp().mean().detach()


@torch.no_grad()
def neighbor_answer_probs(
    ref_model: Any,
    neighbor_full_ds: Any,
    qa_per_author: int,
    author_offset: int,
    pad_id: int,
    batch_size: int = 16,
) -> Dict[int, float]:
    """Per forget question, the mean over its `train` neighbours of the ORIGINAL model's probability of the neighbour's full answer,
    p = exp(mean token log-prob of the answer | neighbour question) (the same quantity as `forget_Q_A_Prob`, teacher forced): what the model gives to an answer to a
    question about an entity it does not know. This is the per-sample target of `forget_prob_mode: neighbor`. `{forget row: probability}`."""
    device = ref_model.device
    rows, ids_list, labels_list = [], [], []
    for j in range(len(neighbor_full_ds)):
        nb = neighbor_full_ds[j]
        rows.append((int(nb["author_id"]) - author_offset) * qa_per_author + int(nb["qa_id"]))
        ids_list.append(torch.as_tensor(nb["input_ids"]))
        labels_list.append(torch.as_tensor(nb["labels"]))
    probs: Dict[int, List[float]] = {}
    for i in range(0, len(rows), batch_size):
        ids, labels = ids_list[i:i + batch_size], labels_list[i:i + batch_size]
        n = max(len(x) for x in ids)
        ii = torch.full((len(ids), n), pad_id, dtype=torch.long)
        ll = torch.full((len(ids), n), -100, dtype=torch.long)
        mm = torch.zeros((len(ids), n), dtype=torch.long)
        for b, (x, y) in enumerate(zip(ids, labels)):  # right padding: the positions of the real tokens are unchanged
            ii[b, :len(x)], ll[b, :len(y)], mm[b, :len(x)] = x, y, 1
        logits = ref_model(input_ids=ii.to(device), attention_mask=mm.to(device)).logits
        _, p = forget_prob_penalty(logits, ii.to(device), ll.to(device), 1.0, per_sample=True)
        for r, v in zip(rows[i:i + batch_size], p.tolist()):
            probs.setdefault(r, []).append(v)
    return {r: sum(v) / len(v) for r, v in probs.items()}


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
    forget_targets: Optional[Mapping[int, float]] = None,
    drift_fn: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Train the W_down of `layers` end to end (see the module docstring). `forget_batches()` / `retain_batches()` return a FRESH iterable
    of batches (`input_ids`, `attention_mask`, `labels`, `index`) over prompt + answer tokens. The best-effort result: the trained weights
    (fp32, cpu) and a history. The weights are NOT written into `model`.

    Two optional terms against the global likelihood drift seen at forget05 / forget10 (TODO/paper_notes/10 section 10: every answer of the edited model became less probable,
    the forget answers stayed at the STATIC target of the original model and ended 2-3x above unseen data):
      cfg.retain_kl_weight > 0             + w * KL(original || edited) of the next-token distribution on the ANSWER tokens of the retain batch (retain + train neighbours)
      cfg.forget_prob_dynamic_every = N>0  every N steps `drift_fn()` (the CURRENT model's probability of an unseen reference set) / its value at step 0 rescales the forget
                                           targets: the forget answers are kept at the level of unseen data AS THE EDITED MODEL SEES IT, not as the original model did."""
    steps, lr = int(cfg["steps"]), float(cfg["lr"])
    kl_w, n_tokens, kl_bs = float(cfg["weight"]), int(cfg["n_tokens"]), int(cfg["batch_size"])
    prob_w = float(cfg.get("forget_prob_weight", 0.0) or 0.0)
    # the target is only needed (and must be a number: w_down.py resolves `unseen`) when the hinge is on
    prob_target = float(cfg.get("forget_prob_target", 0.17)) if prob_w > 0 else 0.17
    prob_mode = str(cfg.get("forget_prob_mode", "hinge"))
    rkl_w = float(cfg.get("retain_kl_weight", 0.0) or 0.0)
    dyn_every = int(cfg.get("forget_prob_dynamic_every", 0) or 0)
    if dyn_every > 0 and drift_fn is None:
        raise ValueError("forget_prob_dynamic_every > 0 needs drift_fn (the probability of the unseen reference set under the current model).")
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
        scale, drift0, drift_now = 1.0, None, None
        if dyn_every > 0 and prob_w > 0:
            drift0 = drift_now = float(drift_fn())        # the starting weights (init: original -> the original model)
            edited_out.clear(); ref_out.clear()
            logger.info(f"dynamic forget target: unseen reference {drift0:.4f} at step 0")
        for step in range(steps):
            if drift0 is not None and step > 0 and step % dyn_every == 0:
                drift_now = float(drift_fn())
                edited_out.clear(); ref_out.clear()
                scale = drift_now / drift0
            fb, rb = rng.choice(forget_list), rng.choice(retain_list)
            terms = {"forget": 0.0, "retain": 0.0}
            optimizer.zero_grad()

            # (a) activations: forget tokens must move by the shift, retain / neighbour tokens must stay
            loss_f = loss_r = 0.0
            loss_p, prob_now = torch.zeros((), device=device), 0.0
            loss_rkl = torch.zeros((), device=device)
            for kind, batch in (("forget", fb), ("retain", rb)):
                inputs = to_dev(batch)
                with torch.no_grad():
                    ref_res = ref_model(**inputs)                       # fills ref_out (the MLP outputs); its logits are only kept for the retain KL
                    ref_logits = ref_res.logits if (kind == "retain" and rkl_w > 0) else None
                    del ref_res
                out = model(**inputs)
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
                if kind == "forget" and prob_w > 0:  # the true answers of the forget samples must not stay more likely than the retain model would give them
                    target = prob_target * scale
                    if prob_mode in ("neighbor", "neighbor_scaled"):  # one level per forget question: the original model's probability of its neighbours' answers
                        if forget_targets is None:
                            raise ValueError("forget_prob_mode `neighbor` needs `forget_targets` (neighbor_answer_probs).")
                        target = torch.tensor([forget_targets[int(i)] * scale for i in batch["index"]], device=device)
                    loss_p, p_mean = forget_prob_penalty(out.logits, inputs["input_ids"], batch["labels"].to(device), target, "neighbor" if prob_mode == "neighbor_scaled" else prob_mode)
                    part = part + prob_w * loss_p
                    prob_now = float(p_mean)
                if kind == "retain" and rkl_w > 0:  # the retained answers must keep the ORIGINAL next-token distribution (not only the MLP output of the edited layers)
                    loss_rkl = retain_answer_kl(ref_logits, out.logits, batch["labels"].to(device))
                    part = part + rkl_w * loss_rkl
                    del ref_logits
                part.backward()
                del out, part  # the logits of this batch are not needed any more (the peak memory)
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
                kl_tok = (ref_logp.exp() * (ref_logp - edited_logp)).sum(-1)          # [pairs, n_tokens]
                keep = torch.stack([p["mask"] for p in picked]).to(kl_tok)             # 0 = a neighbour-name token (mask_neighbor_name)
                loss_kl = (kl_tok * keep).sum() / keep.sum().clamp_min(1.0)
                (kl_w * loss_kl).backward()
                edited_out.clear()
                ref_out.clear()

            optimizer.step()
            history.append({"step": step + 1, "forget_rel_mse": terms["forget"], "retain_rel_mse": terms["retain"], "kl": float(loss_kl.detach()), "forget_prob_penalty": float(loss_p.detach()), "forget_prob": prob_now,
                            "retain_kl": float(loss_rkl.detach()), "target_scale": scale, "unseen_ref": drift_now})
            if (step + 1) % 10 == 0 or step == 0:
                logger.info(f"KL refine {step + 1}/{steps}  forget_rel_mse={terms['forget']:.4g}  retain_rel_mse={terms['retain']:.4g}  kl={float(loss_kl.detach()):.4g}")
        new_weights = {_l: w.detach().cpu() for _l, w in weights.items()}
    return {"weights": new_weights, "history": history}
