# ==== MEOW baseline (Gu et al., "From Evasion to Concealment: Stealthy Knowledge Unlearning for LLMs", ACL Findings 2025; code https://github.com/Carol-gutianle/MEOW, data in it) =====
# Re-implemented for this repository (no code is copied; the data are the authors' released inverted facts):
#   1. inverted facts: for every forget question 4 answers that "do not factually match" the true answer (GPT-4, released in data/tofu/generated_data/forget{01,05,10}_perturbed_generated.json;
#      `answer_0..3`, the true answer is `undesired_answer`), generated for every question ALONE (question + true answer only: no information about the other questions of the author).
#   2. MEMO (suffix mode, ROUGE-1 precision, window 5 characters): the memorisation of each inverted answer by the ORIGINAL model; per question the k answers with the largest (`largest`, the choice of the
#      paper for Llama), smallest or random MEMO are the labels.
#   3. plain cross-entropy fine-tuning (all weights) of the original model on (forget question, selected inverted answer) pairs, every question repeated `num_batch` = 8 times with its k answers in turn;
#      no retain data, no auxiliary model. The paper trains 93-250 steps (batch 32 rows); here checkpoints are written at `--save-steps` and the step is chosen afterwards (matching rule of this repository).
#   `--target neighbor`  the ONE change used as a control (MEOW -> neighbour): the labels are the answers of the author's synthetic TRAIN neighbours (train_rank < k) instead of the inverted facts; the
#                        mechanism (loss, optimiser, repetition) is the same.
#
#   python src/METHOD/MEOW/meow.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget-split forget05 --meow-json <.../forget05_perturbed_generated.json> \
#       --memo-json saves/MEOW/memo/Llama-3.2-1B-Instruct_forget05_memo.json --k 3 --select largest --save-steps 10 20 30 50 --out saves/MEOW/<name>
#   (--memo-json is computed first if the file does not exist; --template-config configs/model/tofu/tofu_<MODEL>_full.yaml gives the chat template of the evaluation)
# Output: <out>/checkpoint-<step>/ (model + tokenizer), <out>/train_log.json, <out>/selection.json. Evaluate with scripts/exp/meow.sh (the standard evaluation of this repository).
# ==============================================================================

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.datasets.datasets_utils import prepare_chat_header, tok_chat_sample  # noqa: E402

FORGET_OFFSET = {"forget01": 360, "forget05": 200, "forget10": 0}      # row of the first forget question inside the 400 rows of forget10 (the numbering of `forget_idx` in the neighbour csv)
ANSWER_RE = re.compile(r"^answer_\d+$")


# ----------------------------------------------------------------------------- data
def answer_keys(sample: Dict) -> List[str]:
    return sorted((k for k in sample if ANSWER_RE.match(k) and str(sample[k]).strip()), key=lambda k: int(k.split("_")[1]))


def load_inverted(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- MEMO
def split_chunks(question: str, answer: str, window: int = 5):
    """suffix mode of the paper: (prompt = question + ' ' + answer[:n], label = answer[n:]) for n = 1, 1 + window, ... < len(answer) (a character window)."""
    return [(question + " " + answer[:n], answer[n:]) for n in range(1, len(answer), window)]


def rouge1_precision(prediction: str, reference: str) -> float:
    """ROUGE-1 precision (stemmed unigrams) of `prediction` against `reference`, as `rouge_score` computes it."""
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True).score(reference, prediction)["rouge1"].precision


@torch.no_grad()
def memo_of_answers(model, tokenizer, question: str, answers: Sequence[str], window: int = 5, max_new_tokens: int = 200, batch_size: int = 16) -> List[float]:
    """MEMO of every answer: the mean ROUGE-1 precision, over the chunks, of the model's greedy continuation of `question + ' ' + answer[:n]` against the rest of the answer (raw text, no chat template,
    as in the released code)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    jobs = [(i, p, l) for i, a in enumerate(answers) for p, l in split_chunks(question, a, window)]
    sums, counts = [0.0] * len(answers), [0] * len(answers)
    for s in range(0, len(jobs), batch_size):
        batch = jobs[s:s + batch_size]
        enc = tokenizer([p for _, p, _ in batch], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (i, _, label), text in zip(batch, texts):
            sums[i] += scorer.score(label, text)["rouge1"].precision
            counts[i] += 1
    return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(len(answers))]


def add_memo(model, tokenizer, data: List[Dict], window: int = 5, max_new_tokens: int = 200, batch_size: int = 16, log=print) -> List[Dict]:
    """Adds `<key>_rouge1` for `undesired_answer` and every `answer_i` (the format of the released memo_data)."""
    out = []
    for n, sample in enumerate(data):
        keys = ["undesired_answer"] + answer_keys(sample)
        scores = memo_of_answers(model, tokenizer, sample["question"], [sample[k] for k in keys], window, max_new_tokens, batch_size)
        out.append({**sample, **{f"{k}_rouge1": v for k, v in zip(keys, scores)}})
        if log and (n + 1) % 20 == 0:
            log(f"MEMO {n + 1}/{len(data)}")
    return out


def select_answers(sample: Dict, k: int, metric: str = "largest", seed: int = 42) -> List[str]:
    """The k answer keys (`answer_i`) of a question: the largest / smallest MEMO, or random. Fewer than k answers: the list is cycled by the caller."""
    keys = [x for x in answer_keys(sample) if f"{x}_rouge1" in sample]
    if not keys:
        keys = answer_keys(sample)
    k = min(k, len(keys))
    if metric == "random" or not all(f"{x}_rouge1" in sample for x in keys):
        return random.Random(seed).sample(keys, k)
    ranked = sorted(keys, key=lambda x: sample[f"{x}_rouge1"], reverse=(metric == "largest"))
    return ranked[:k]


# ----------------------------------------------------------------------------- targets
def neighbor_answers(csv_path: str, split: str, n: int, k: int) -> List[List[str]]:
    """[per forget question: the answers of the author's synthetic TRAIN neighbours with train_rank < k] (the neighbour csv of this repository, `forget_idx` numbering of forget10)."""
    off = FORGET_OFFSET[split]
    by_idx: Dict[int, Dict[int, str]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train" and int(r["train_rank"]) < k:
                by_idx.setdefault(int(r["forget_idx"]), {})[int(r["train_rank"])] = r["answer"]
    out = []
    for i in range(n):
        d = by_idx.get(off + i, {})
        if not d:
            raise ValueError(f"no train neighbour answer for forget question {i} (forget_idx {off + i}) in {csv_path}")
        out.append([d[r] for r in sorted(d)])
    return out


def build_pairs(data: List[Dict], target: str, k: int, select: str, neighbor_csv: Optional[str] = None, split: str = "forget05"):
    """-> (pairs: [(question, [k answers])], selection: [{question, keys / answers}])"""
    if target == "inverted":
        sel = [select_answers(s, k, select) for s in data]
        pairs = [(s["question"], [s[x] for x in keys]) for s, keys in zip(data, sel)]
        return pairs, [{"question": s["question"], "keys": keys} for s, keys in zip(data, sel)]
    if target == "neighbor":
        ans = neighbor_answers(neighbor_csv, split, len(data), k)
        return [(s["question"], a) for s, a in zip(data, ans)], [{"question": s["question"], "answers": a} for s, a in zip(data, ans)]
    raise ValueError(f"target must be `inverted` or `neighbor`, got {target}")


def expand_rows(pairs, num_batch: int = 8):
    """The released training data: every question appears `num_batch` times, its k answers in turn."""
    return [[(q, answers[i % len(answers)]) for i in range(num_batch)] for q, answers in pairs]


# ----------------------------------------------------------------------------- training
def tokenize_rows(rows, tokenizer, template_args, max_length=512):
    header = prepare_chat_header(template_args)
    out = []
    for group in rows:
        g = []
        for q, a in group:
            d = tok_chat_sample(q, a, 0, tokenizer=tokenizer, chat_header=header, template_args=template_args, max_length=max_length)
            g.append((d["input_ids"][0], d["labels"][0]))
        out.append(g)
    return out


def collate(items, pad_id):
    ml = max(len(i) for i, _ in items)
    ids = torch.full((len(items), ml), pad_id, dtype=torch.long)
    lab = torch.full((len(items), ml), -100, dtype=torch.long)
    att = torch.zeros((len(items), ml), dtype=torch.long)
    for r, (i, l) in enumerate(items):
        ids[r, :len(i)] = torch.tensor(i)
        lab[r, :len(l)] = torch.tensor(l)
        att[r, :len(i)] = 1
    return ids, lab, att


def row_loss(model, ids, lab, att):
    """Mean over the rows of the per-row mean token cross-entropy of the answer tokens."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="none").view(tgt.shape)
    mask = (tgt != -100).float()
    return ((ce * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()


def train(model, tokenizer, groups, steps: int, lr=1e-5, weight_decay=0.01, rows_per_step=32, micro_rows=8, save_steps: Sequence[int] = (), out_dir=None, seed=42, log=print):
    """`groups`: tokenised rows per question (num_batch rows each). One optimiser step = `rows_per_step` rows = rows_per_step / num_batch questions; epochs over shuffled questions."""
    rng = random.Random(seed)
    num_batch = len(groups[0])
    q_per_step = max(1, rows_per_step // num_batch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.train()
    order, history = [], []
    for step in range(1, steps + 1):
        if len(order) < q_per_step:
            fresh = list(range(len(groups)))
            rng.shuffle(fresh)
            order += fresh
        batch_q, order = order[:q_per_step], order[q_per_step:]
        items = [row for q in batch_q for row in groups[q]]
        total = 0.0
        for s in range(0, len(items), micro_rows):
            chunk = items[s:s + micro_rows]
            loss = row_loss(model, *[t.to(model.device) for t in collate(chunk, pad)]) * (len(chunk) / len(items))
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        history.append({"step": step, "loss": total})
        if log and (step % 5 == 0 or step == 1):
            log(f"step {step}/{steps} loss {total:.4f}")
        if out_dir and step in set(save_steps):
            ck = Path(out_dir) / f"checkpoint-{step}"
            model.save_pretrained(ck)
            tokenizer.save_pretrained(ck)
    return history


def main():
    ap = argparse.ArgumentParser(description="MEOW baseline: inverted facts selected by MEMO (or the neighbours answers), cross-entropy fine-tuning, checkpoints at --save-steps")
    ap.add_argument("--model", required=True)
    ap.add_argument("--forget-split", default="forget05", choices=list(FORGET_OFFSET))
    ap.add_argument("--meow-json", required=True, help="the released inverted facts of the split")
    ap.add_argument("--memo-json", default=None, help="MEMO scores of the ORIGINAL model (computed and written when missing)")
    ap.add_argument("--target", default="inverted", choices=["inverted", "neighbor"])
    ap.add_argument("--neighbor-csv", default=None)
    ap.add_argument("--k", type=int, default=3, help="answers per question (the paper, Llama-2: 2 / 3 / 6 for 1 % / 5 % / 10 %)")
    ap.add_argument("--select", default="largest", choices=["largest", "smallest", "random"])
    ap.add_argument("--num-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=None, help="default: max(save-steps)")
    ap.add_argument("--save-steps", type=int, nargs="+", default=[10, 20, 30, 50])
    ap.add_argument("--rows-per-step", type=int, default=32)
    ap.add_argument("--micro-rows", type=int, default=8)
    ap.add_argument("--template-config", required=True, help="configs/model/tofu/tofu_<MODEL>_full.yaml (chat template of the evaluation)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--memo-max-new-tokens", type=int, default=200, help="the released code generates 200; the ranking of the answers of ONE question does not need it (64 is ~3x faster)")
    ap.add_argument("--memo-batch-size", type=int, default=32)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    template_args = OmegaConf.to_container(OmegaConf.load(a.template_config)["template_args"], resolve=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(dev)
    data = load_inverted(a.meow_json)
    if a.target == "inverted" and a.select != "random":
        if a.memo_json and os.path.exists(a.memo_json):
            data = load_inverted(a.memo_json)
        else:
            model.eval()
            data = add_memo(model, tok, data, max_new_tokens=a.memo_max_new_tokens, batch_size=a.memo_batch_size)
            if a.memo_json:
                os.makedirs(os.path.dirname(a.memo_json) or ".", exist_ok=True)
                json.dump(data, open(a.memo_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pairs, selection = build_pairs(data, a.target, a.k, a.select, a.neighbor_csv, a.forget_split)
    groups = tokenize_rows(expand_rows(pairs, a.num_batch), tok, template_args, a.max_length)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "selection": selection}, open(os.path.join(a.out, "selection.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    steps = a.steps or max(a.save_steps)
    hist = train(model, tok, groups, steps, a.lr, 0.01, a.rows_per_step, a.micro_rows, a.save_steps, a.out, a.seed)
    json.dump(hist, open(os.path.join(a.out, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
