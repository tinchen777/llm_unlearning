"""Sampling-level detectability: does a model answer the forgotten questions the way it answers questions about authors it never saw?

The greedy single answer (forget_responses.json) hides how the probability mass was moved. An observer who samples N answers per question can see
(a) how much the answers disagree (a model that does not know the author invents a different answer every time; a model that was made to
    forget may be more or less stable),
(b) whether the same substitute name keeps coming back (across samples and across questions),
(c) whether the true answer still leaks (leak@k: at least one of the first k samples contains the true content).
The reference is the SAME model on the holdout questions (authors it has never seen: no retain model needed); a retain model, if given, is one more row.

    python scripts/analysis/sample_consistency.py --gpu 0 --out saves/analysis/sample_1B \
        --model retain=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99 \
        --model full=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --model GA=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-1B-Instruct_forget01_lr1e-4_ep5/checkpoint-2 \
        --model new=near:saves/NEAR/single_v3/<run>/k5_c1_all/l11-12-13 \
        --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full  [--n 10 --temperature 1.0]

Model specs  label=hf:<hub id or folder>   a HF model (retain / full / a baseline checkpoint-N folder)
             label=near:<run folder>       a NEAR run: <folder>/model if it was saved, else w_down.pt applied to --base
Data         locuslab/TOFU (offline cache): forget01 / holdout01, `train` split, columns question / answer; same chat prompt as the evaluation.
Output       <out>/<label>.json (per-set metrics + the samples), <out>/summary.md (the table), printed.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List, Sequence

NAME_RE = re.compile(r"\b[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+)+")   # "Basil Mahfouz Al-Kuwaiti", "Khalid Qais Al-Marzouq"
_KEY_RE = re.compile(r"\b(?:[A-Z][\w'\-]+|\d[\d/,.\-]*)\b")       # capitalised words and numbers = the factual content of an answer
_STOP = {"The", "A", "An", "He", "She", "His", "Her", "It", "In", "On", "Of", "And", "This", "That", "Yes", "No", "As", "At", "By", "For", "To", "With"}


def tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    l = lcs_len(ta, tb)
    return 0.0 if l == 0 else 2 * l / (len(ta) + len(tb))


def rouge_l_recall(pred: str, ref: str) -> float:
    tp, tr = tokens(pred), tokens(ref)
    return 0.0 if not tr else lcs_len(tp, tr) / len(tr)


def names(text: str) -> List[str]:
    return [n.strip() for n in NAME_RE.findall(text)]


def key_tokens(text: str) -> List[str]:
    return sorted({t.lower() for t in _KEY_RE.findall(text) if t not in _STOP})


def key_recall(sample: str, truth: str) -> float:
    """Share of the factual tokens (capitalised words, numbers) of the true answer that the sample contains."""
    keys = {w for k in key_tokens(truth) for w in tokens(k)}
    if not keys:
        return 0.0
    return len(keys & set(tokens(sample))) / len(keys)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def compute_metrics(questions: Sequence[str], truths: Sequence[str], samples: Sequence[Sequence[str]],
                    rouge_thr: float = 0.7, key_thr: float = 0.75, ks: Sequence[int] = (1, 5)) -> Dict[str, float]:
    """`samples[i]` = the N sampled answers to question i. All numbers are means over the questions."""
    n = len(samples[0])
    ks = sorted({k for k in list(ks) + [n] if k <= n})
    agree, n_names, modal_in_q = [], [], []
    leak_key = {k: [] for k in ks}
    leak_rouge = {k: [] for k in ks}
    all_names = Counter()
    for q, t, ss in zip(questions, truths, samples):
        qn = {x.lower() for x in names(q)}
        qtok = set(tokens(q))
        agree.append(_mean(rouge_l_f1(a, b) for a, b in combinations(ss, 2)))
        # substitute names: name-like spans of the answer that are NOT in the question
        per_sample = []
        for s in ss:
            sub = [x for x in names(s) if x.lower() not in qn and not set(tokens(x)) <= qtok]
            per_sample.append(sub[0] if sub else None)
        got = [x for x in per_sample if x]
        n_names.append(len(set(got)))
        modal_in_q.append(Counter(got).most_common(1)[0][1] / n if got else 0.0)
        all_names.update(set(got))
        for k in ks:
            leak_key[k].append(float(any(key_recall(s, t) >= key_thr for s in ss[:k])))
            leak_rouge[k].append(float(any(rouge_l_recall(s, t) >= rouge_thr for s in ss[:k])))
    total_q = len(samples)
    out = {
        "n_questions": total_q, "n_samples": n,
        "self_agreement": _mean(agree),                 # mean pairwise ROUGE-L F1 of the N answers (1 = always the same answer)
        "distinct_names": _mean(n_names),               # distinct substitute names per question (over N samples)
        "modal_name_share": _mean(modal_in_q),          # share of the samples that use the question's most frequent substitute name
        # across questions: does ONE substitute name come back for many different questions? (questions per name, max over names / n questions)
        "cross_question_name_share": (all_names.most_common(1)[0][1] / total_q) if all_names else 0.0,
        "cross_distinct_names": len(all_names) / total_q,
    }
    for k in ks:
        out[f"leak_key@{k}"] = _mean(leak_key[k])
        out[f"leak_rouge@{k}"] = _mean(leak_rouge[k])
    return out


# ----------------------------------------------------------------------------- model / generation (needs torch + transformers)
def load_questions(split: str, limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset("locuslab/TOFU", split, split="train")
    qs, ans = list(ds["question"]), list(ds["answer"])
    return (qs[:limit], ans[:limit]) if limit else (qs, ans)


def build_prompts(tokenizer, questions, system="You are a helpful assistant.", date_string="10 Apr 2025"):
    out = []
    for q in questions:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        try:
            out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=date_string))
        except TypeError:
            out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def sample_answers(model, tokenizer, prompts, n, temperature=1.0, top_p=1.0, max_new_tokens=64, batch_size=8, seed=0):
    import torch
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    outs: List[List[str]] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n,
                                 max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for b in range(len(batch)):
            outs.append([t.strip() for t in texts[b * n:(b + 1) * n]])
    return outs


def load_model(spec: str, base: str, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM
    kind, _, path = spec.partition(":")
    dt = getattr(torch, dtype)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "hf":
        if os.path.isdir(path) and not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"{path} has no config.json: a baseline checkpoint-N folder only holds the evaluation (evals/) unless the run was trained with "
                "EXTRA_TRAIN='trainer.args.save_strategy=steps +trainer.args.save_steps=<steps per epoch>' (configs/trainer/args/default.yaml: save_strategy no); "
                "the run folder itself holds only the FINAL model.")
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt).to(dev)
    if kind == "near":
        if os.path.isdir(os.path.join(path, "model")):
            return AutoModelForCausalLM.from_pretrained(os.path.join(path, "model"), torch_dtype=dt).to(dev)
        saved = torch.load(os.path.join(path, "w_down.pt"), map_location="cpu")
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dt).to(dev)
        for l, w in saved["weights"].items():
            model.model.layers[int(l)].mlp.down_proj.weight.data.copy_(w.to(dev, dtype=model.dtype))
        return model
    raise ValueError(f"unknown model spec '{spec}' (hf:<id/folder> | near:<run folder>)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="label=hf:<path> | label=near:<run folder> (repeatable)")
    ap.add_argument("--base", required=True, help="the TOFU full model (tokenizer, and the base of near: runs without a saved model)")
    ap.add_argument("--forget", default="forget01")
    ap.add_argument("--holdout", default="holdout01")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first LIMIT questions of each set (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None, help="GPU id (sets CUDA_VISIBLE_DEVICES before torch is imported); default: the visible GPU 0 (cpu if none)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)   # torch is imported lazily below, so this takes effect
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    sets = {"forget": load_questions(a.forget, a.limit), "holdout": load_questions(a.holdout, a.limit)}
    rows = []
    for spec in a.model:
        label, _, s = spec.partition("=")
        print(f"==== {label}: {s}", flush=True)
        model = load_model(s, a.base)
        res = {"spec": s, "sets": {}}
        for name, (qs, ans) in sets.items():
            samples = sample_answers(model, tok, build_prompts(tok, qs), a.n, a.temperature, a.top_p, a.max_new_tokens, a.batch_size, a.seed)
            res["sets"][name] = {"metrics": compute_metrics(qs, ans, samples), "samples": [{"question": q, "truth": t, "answers": x} for q, t, x in zip(qs, ans, samples)]}
        with open(os.path.join(a.out, f"{label}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        rows.append((label, res["sets"]["forget"]["metrics"], res["sets"]["holdout"]["metrics"]))
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    keys = ["self_agreement", "distinct_names", "modal_name_share", "cross_question_name_share", "cross_distinct_names"]
    leaks = [k for k in rows[0][1] if k.startswith("leak_")]
    lines = ["| model | " + " | ".join(f"{k} (forget / holdout)" for k in keys) + " | " + " | ".join(leaks) + " |",
             "|---|" + "---|" * (len(keys) + len(leaks))]
    for label, f, h in rows:
        lines.append(f"| {label} | " + " | ".join(f"{f[k]:.3f} / {h[k]:.3f}" for k in keys) + " | " + " | ".join(f"{f[k]:.3f}" for k in leaks) + " |")
    md = "\n".join(lines)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
