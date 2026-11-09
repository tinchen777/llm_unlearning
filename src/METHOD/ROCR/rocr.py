# ==== ROCR (Rank-One Concept Redirection) baseline, re-implemented from the released code of
#   Ye, Zhang, Wu: "LLM Unlearning Should Be Form-Independent" (arXiv 2506.07795), https://github.com/Acruxos/ORT (MIT licence),
#   LLaMA-Factory/src/llmtuner/train/rocr/. The paper describes a single-layer rank-one closed form; the RELEASED code (which produced Table 1) is a
#   MEMIT / AlphaEdit style multi-layer ridge update, which is what is implemented here:
#
#   for the layers l_0 < ... < l_{n-1} (consecutive; the paper: 4,5,6 of Llama-3-8B), one after the other, with the CURRENT model:
#       K      = mean over 6 chat templates of the input of down_proj (d_ff) at the LAST token of the forget name          [d_ff, m]
#       z      = hidden state at the OUTPUT of the last edited layer, last token of the TARGET name (chat template 0)    [d, m]
#       cur_z  = the same position of the forget name in the bare prompt "<name>."                                         [d, m]
#       R      = (z - cur_z) / (n - i)                                                    (the residual is shared by the remaining layers)
#       Delta^T = solve( P (K K^T + C) + L2 I ,  P K R^T )     P = null-space projector of the down_proj inputs of generic text (Wikipedia / WikiText)
#       W <- W + Delta                                                                                                    (C = 0 for one batch of edits)
#   P = U_null U_null^T, U_null = eigenvectors of the mean second moment of the inputs with eigenvalue < `nullspace_threshold` (2e-2).
#
#   `--target famous`   the original ROCR: every forget author is redirected to the real, well-known person `--famous` (default Donald Trump)
#   `--target neighbor` the ONLY change: the target name is the author's train_rank == 1 neighbour (a fictitious, attribute-matched author)
#
#   Deviations from the release (documented in TODO/exp/17): the second-moment statistics come from `--wiki` (WikiText-103 by default; the release:
#   100k articles of the 20200501 Wikipedia dump, script dataset that current `datasets` cannot load), and the edited model is saved for the standard
#   evaluation of this repository instead of the ORT benchmark.
#
# Usage (offline once `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` is in the cache):
#   python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget-split forget01 --author-offset 199 \
#       --neighbor-csv data/tofu/tofu_neighbor01.csv --layers 2 3 4 --target neighbor --out saves/ROCR/<name>
# Output: <out>/model (edited model + tokenizer), <out>/rocr_summary.json (names, layers, residual norms), responses_*.json are written by the evaluation.
# ==============================================================================

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.arch import get_decoder_layers, get_out_proj  # noqa: E402

STARTERS = ["", "Please answer the question about", "Choose the best answer about", "Write the answer about", "Tell me about", "Write something about"]
_STOP = {"Author", "What", "Which", "Who", "Where", "When", "How", "In", "The", "Is", "Does", "Did", "Can", "Could", "Please", "Tell", "Write", "Describe",
         "Name", "List", "Explain", "Give", "As", "His", "Her", "He", "She", "They", "A", "An", "On", "At", "For", "Of", "And", "To", "With", "By"}


# ---------------------------------------------------------------- names
def extract_author_name(texts: Sequence[str]) -> str:
    """The name of the author of ONE author's QA rows: the capitalised word sequence that occurs in the most rows (ties: the longest).
    Sentence-initial words (`Author`, `What`, ...) are not part of it."""
    cnt: Counter = Counter()
    for t in texts:
        seen = set()
        for m in re.finditer(r"[A-Z][\w'’\-\.]*(?:\s+[A-Z][\w'’\-\.]*)*", t):
            words = m.group(0).split()
            for i in range(len(words)):
                for j in range(i + 2, len(words) + 1):
                    cand = words[i:j]
                    if cand[0] in _STOP:
                        continue
                    seen.add(" ".join(cand).rstrip(".'’"))
        cnt.update(seen)
    if not cnt:
        raise ValueError("no capitalised name found in the rows of this author")
    return max(cnt, key=lambda c: (cnt[c], len(c)))


def forget_author_names(questions: Sequence[str], answers: Sequence[str], qa_per_author: int = 20) -> List[str]:
    """Names of the authors of the forget rows (TOFU: the QAs of one author are contiguous, 20 per author)."""
    out = []
    for a in range(0, len(questions), qa_per_author):
        out.append(extract_author_name([f"{q} {r}" for q, r in zip(questions[a:a + qa_per_author], answers[a:a + qa_per_author])]))
    return out


def neighbor_names(csv_path: str, authors: Sequence[int], rank: int = 1) -> List[str]:
    """`neighbor_name` of the `train_rank == rank` neighbour of every author (the neighbour csv of this repository)."""
    by_author: Dict[int, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train" and int(row["train_rank"]) == rank:
                by_author.setdefault(int(row["author_id"]), row["neighbor_name"])
    missing = [a for a in authors if a not in by_author]
    if missing:
        raise ValueError(f"authors {missing} have no train_rank={rank} neighbour in {csv_path}")
    return [by_author[a] for a in authors]


# ---------------------------------------------------------------- prompts and positions
def chat_prompts(tok, name: str, starters: Sequence[str] = STARTERS) -> List[str]:
    """The 6 chat-templated prompts of the release: user message `<starter> <name>.` + generation prompt + `Answer:`."""
    out = []
    for s in starters:
        msg = [{"role": "user", "content": f"{s} {name}."}]
        out.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "Answer:")
    return out


def last_name_token(tok, text: str, name: str) -> int:
    """Index (in the tokenisation of `text`, special tokens included) of the last token of the LAST occurrence of `name`."""
    start = text.rindex(name)
    end = start + len(name)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    idx = None
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s < end and e > start:
            idx = i
    if idx is None:
        raise ValueError(f"`{name}` not found in the tokens of `{text}`")
    return idx


@torch.no_grad()
def hidden_at(model, tok, texts: Sequence[str], names: Sequence[str], layer: int, kind: str) -> torch.Tensor:
    """[len(texts), dim]: `kind='key'` the input of down_proj of `layer`, `kind='hidden'` the output of decoder layer `layer`, at the last token of the name."""
    dev = next(model.parameters()).device
    store: Dict[str, torch.Tensor] = {}
    if kind == "key":
        mod = get_out_proj(model, layer)
        handle = mod.register_forward_hook(lambda m, i, o: store.__setitem__("x", i[0].detach()))
    elif kind == "hidden":
        mod = get_decoder_layers(model)[layer]
        handle = mod.register_forward_hook(lambda m, i, o: store.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
    else:
        raise ValueError(kind)
    rows = []
    try:
        for text, name in zip(texts, names):
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
            model(**ids)
            rows.append(store["x"][0, last_name_token(tok, text, name)].float())
    finally:
        handle.remove()
    return torch.stack(rows)


# ---------------------------------------------------------------- null space of generic text
@torch.no_grad()
def collect_second_moment(model, tok, layers: Sequence[int], texts: Sequence[str], max_len: int = 512, batch_size: int = 16):
    """{layer: mean over tokens of x x^T} for the inputs x of down_proj, all layers in one pass."""
    dev = next(model.parameters()).device
    acc: Dict[int, torch.Tensor] = {}
    count = 0
    cur: Dict[int, torch.Tensor] = {}
    handles = [get_out_proj(model, l).register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, i[0].detach())) for l in layers]
    try:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), batch_size):
            batch = [texts[i] for i in order[s:s + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
            model(**enc)
            keep = enc["attention_mask"].bool().reshape(-1)
            for l in layers:
                x = cur[l].reshape(-1, cur[l].shape[-1])[keep].float()
                acc[l] = acc.get(l, 0) + x.T @ x
            count += int(keep.sum())
    finally:
        for h in handles:
            h.remove()
    return {l: a / count for l, a in acc.items()}, count


def null_space_projector(second_moment: torch.Tensor, threshold: float) -> torch.Tensor:
    """P = U_null U_null^T, U_null = the eigenvectors of the (symmetric, PSD) second moment with eigenvalue < threshold."""
    evals, evecs = torch.linalg.eigh(second_moment.double())
    u = evecs[:, evals < threshold].float()
    return u @ u.T


# ---------------------------------------------------------------- the update
def ridge_update(K: torch.Tensor, resid: torch.Tensor, P: torch.Tensor, l2: float = 1.0, C: torch.Tensor | None = None) -> torch.Tensor:
    """Delta [d, d_ff] = solve(P (K K^T + C) + l2 I, P K R^T)^T for K [d_ff, m], resid R [d, m], projector P [d_ff, d_ff]."""
    kk = K @ K.T + (0 if C is None else C)
    a = P @ kk + l2 * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    return torch.linalg.solve(a, P @ K @ resid.T).T


@torch.no_grad()
def apply_rocr(model, tok, forget_names: Sequence[str], target_names: Sequence[str], layers: Sequence[int], projectors: Dict[int, torch.Tensor],
               l2: float = 1.0, scale: float = 1.0, target_templates: str = "first") -> Dict[str, list]:
    if list(layers) != list(range(layers[0], layers[0] + len(layers))):
        raise ValueError(f"layers must be consecutive, got {list(layers)}")
    last = layers[-1]
    # target z: the hidden state at the output of the last edited layer of the TARGET name (template 0 of the release; `mean`: the mean over the 6 templates)
    z = []
    for t in target_names:
        ps = chat_prompts(tok, t)
        ps = ps[:1] if target_templates == "first" else ps
        z.append(hidden_at(model, tok, ps, [t] * len(ps), last, "hidden").mean(0))
    z = torch.stack(z, 1)                                                   # [d, m]
    log: Dict[str, list] = {"z_error": [], "update_norm": [], "weight_norm": []}
    for i, layer in enumerate(layers):
        ks = torch.stack([hidden_at(model, tok, chat_prompts(tok, n), [n] * len(STARTERS), layer, "key").mean(0) for n in forget_names], 1)  # [d_ff, m]
        cur = torch.stack([hidden_at(model, tok, [f"{n}."], [n], last, "hidden")[0] for n in forget_names], 1)                                # [d, m]
        resid = scale * (z - cur) / (len(layers) - i)
        delta = ridge_update(ks.float(), resid.float(), projectors[layer].to(ks.device), l2)
        proj = get_out_proj(model, layer)
        log["z_error"].append(float((z - cur).norm(dim=0).mean()))
        log["update_norm"].append(float(delta.norm()))
        log["weight_norm"].append(float(proj.weight.float().norm()))
        proj.weight.copy_((proj.weight.float() + delta).to(proj.weight))
    return log


# ---------------------------------------------------------------- generic text
def load_wiki(n_docs: int, seed: int = 0, dataset: str = "wikitext", min_chars: int = 200) -> List[str]:
    """n_docs paragraphs of WikiText-103 (the official way: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1"), offline from the cache)."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    if dataset != "wikitext":
        raise ValueError("only `wikitext` is wired (the release's Wikipedia dump is a script dataset that current `datasets` cannot load)")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g).tolist()
    out: List[str] = []
    for i in idx:
        t = ds[i]["text"].strip()
        if len(t) >= min_chars and not t.startswith("="):
            out.append(t)
            if len(out) >= n_docs:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--model", required=True, help="the TOFU full model (hf id or path)")
    ap.add_argument("--forget-split", default="forget01")
    ap.add_argument("--author-offset", type=int, required=True, help="author_id of the first forget author in the neighbour csv (199 / 191 / 181 for forget01 / 05 / 10)")
    ap.add_argument("--neighbor-csv", default=None, help="needed for --target neighbor")
    ap.add_argument("--target", choices=["famous", "neighbor"], default="famous")
    ap.add_argument("--famous", default="Donald Trump")
    ap.add_argument("--layers", type=int, nargs="+", required=True, help="consecutive layers (release, Llama-3-8B: 4 5 6)")
    ap.add_argument("--threshold", type=float, default=2e-2)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0, help="factor on the residual (1 = ROCR; >1 is an extra, non-ROCR row)")
    ap.add_argument("--target-templates", choices=["first", "mean"], default="first")
    ap.add_argument("--n-docs", type=int, default=100000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--stats-dir", default="data/rocr", help="cache of the null-space projectors (per model and layer)")
    ap.add_argument("--forget-names", default=None, help="json list overriding the extracted names of the forget authors")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    forget = load_dataset("locuslab/TOFU", a.forget_split)["train"]
    names = json.load(open(a.forget_names)) if a.forget_names else forget_author_names(forget["question"], forget["answer"])
    authors = [a.author_offset + i for i in range(len(names))]
    if a.target == "neighbor":
        if not a.neighbor_csv:
            raise SystemExit("--target neighbor needs --neighbor-csv")
        targets = neighbor_names(a.neighbor_csv, authors, rank=1)
    else:
        targets = [a.famous] * len(names)
    print("forget authors -> targets:", list(zip(names, targets)))

    tag = Path(a.model.rstrip("/")).name
    projectors = {}
    for l in a.layers:
        f = Path(a.stats_dir) / tag / f"P_layer{l}_t{a.threshold:g}_n{a.n_docs}.pt"
        if f.exists():
            projectors[l] = torch.load(f, map_location=dev)
    todo = [l for l in a.layers if l not in projectors]
    if todo:
        docs = load_wiki(a.n_docs)
        print(f"second moment of {len(docs)} paragraphs for layers {todo}")
        sm, n_tok = collect_second_moment(model, tok, todo, docs, a.max_len)
        for l in todo:
            projectors[l] = null_space_projector(sm[l], a.threshold)
            f = Path(a.stats_dir) / tag / f"P_layer{l}_t{a.threshold:g}_n{a.n_docs}.pt"
            f.parent.mkdir(parents=True, exist_ok=True)
            torch.save(projectors[l].cpu(), f)
            print(f"layer {l}: tokens={n_tok}, null-space dim={int(round(float(torch.trace(projectors[l]))))}/{sm[l].shape[0]}")

    log = apply_rocr(model, tok, names, targets, a.layers, projectors, l2=a.l2, scale=a.scale, target_templates=a.target_templates)
    out = Path(a.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    summary = {"model": a.model, "forget_split": a.forget_split, "target": a.target, "famous": a.famous, "layers": a.layers, "threshold": a.threshold, "l2": a.l2,
               "scale": a.scale, "target_templates": a.target_templates, "n_docs": a.n_docs, "forget_names": names, "target_names": targets, **log}
    json.dump(summary, open(out / "rocr_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("z_error", "update_norm", "weight_norm")}))


if __name__ == "__main__":
    main()
