"""The two "naturalness" attacks of the MEOW paper (Gu et al. 2025, appendix A) on the responses an evaluation saved, plus their false-positive rate on the holdout questions.

  PPL attack          perplexity (under a reference language model) of  question + answer ; a response is flagged as unlearned when PPL > 200   (garbled text)
  Consistency attack  similarity of the response to the canned refusal "Sorry, I don't know" ; flagged when > 0.2                                 (template refusals)
                      with `--embedder <sentence-transformers model>` the cosine similarity; otherwise the TAS refusal score of the evaluation (`tas_refusal` in the responses, regular expressions)
                      >= 0.5 is used (a different detector of the same kind, said in the output)
The attack success rate (ASR) = the share of the FORGET responses that is flagged; the same on the HOLDOUT responses is the false-positive rate (the paper does not report it).
"Natural-looking" in this repository = both ASRs low. Whether natural-looking responses are also indistinguishable is the question of rank_attack_v2.py / the sampling and likelihood channels.

    python scripts/analysis/naturalness.py --ref-model open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 \\
        MEOW=saves/eval/MEOW/<run> NEAR-Dist=saves/eval/NEAR/<prefix>/<run> retain=saves/eval/retain/Llama-3.2-3B-Instruct/forget05 [--embedder sentence-transformers/all-MiniLM-L6-v2] [--out file.md]
Each label=folder holds forget_responses.json and holdout_responses.json of an evaluation (src/eval.py experiment=eval/tofu/probe). No generation is run; the reference model is only scored.
"""
import argparse
import json
import math
import os
import re
import statistics as st
import sys
from typing import Dict, List, Sequence

PPL_THRESHOLD = 200.0
SIM_THRESHOLD = 0.2
CANNED = "Sorry, I don't know"


def question_of(record: Dict) -> str:
    """The question inside the saved (decoded, templated) prompt: after the last `user` marker, before the final `assistant`."""
    text = record["input"].split("user\n\n")[-1]
    return text.rsplit("assistant", 1)[0].strip()


def perplexities(model, tok, texts: Sequence[str], batch_size: int = 8) -> List[float]:
    """exp(mean token NLL) of every text under the model (a BOS token is prepended)."""
    import torch
    out = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        for t in batch:
            ids = tok(t, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
            if ids.shape[1] < 2:
                out.append(float("nan"))
                continue
            with torch.no_grad():
                nll = model(input_ids=ids, labels=ids).loss.item()
            out.append(math.exp(min(nll, 50.0)))
    return out


def embed_similarity(embedder, texts: Sequence[str]) -> List[float]:
    from sentence_transformers import SentenceTransformer, util
    m = SentenceTransformer(embedder)
    e = m.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
    c = m.encode([CANNED], convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(e, c)[:, 0].tolist()


def attack(records: Sequence[Dict], ppls: Sequence[float], sims: Sequence[float] = None) -> Dict[str, float]:
    n = len(records)
    flag_ppl = [p > PPL_THRESHOLD for p in ppls if p == p]
    if sims is not None:
        flag_cons = [s > SIM_THRESHOLD for s in sims]
    else:
        flag_cons = [float(r.get("tas_refusal", 0.0)) >= 0.5 for r in records]
    return {"n": n, "ppl_median": st.median([p for p in ppls if p == p]) if ppls else float("nan"), "asr_ppl": sum(flag_ppl) / max(len(flag_ppl), 1),
            "asr_consistency": sum(flag_cons) / max(len(flag_cons), 1), "degenerate": sum(bool(r.get("degenerate")) for r in records) / max(n, 1)}


def load(folder: str, which: str):
    with open(os.path.join(folder, f"{which}_responses.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-model", required=True, help="reference language model for the PPL (e.g. the retain model of the family)")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("models", nargs="+", help="label=<evaluation folder>")
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ref_model)
    model = AutoModelForCausalLM.from_pretrained(a.ref_model, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    rows = []
    for spec in a.models:
        label, _, folder = spec.partition("=")
        res = {}
        for which in ("forget", "holdout"):
            try:
                recs = load(folder, which)
            except OSError:
                continue
            texts = [f"{question_of(r)} {r['generation']}" for r in recs]
            sims = embed_similarity(a.embedder, [r["generation"] for r in recs]) if a.embedder else None
            res[which] = attack(recs, perplexities(model, tok, texts), sims)
        rows.append((label, res))
    det = f"embedding similarity to \"{CANNED}\" > {SIM_THRESHOLD} ({a.embedder})" if a.embedder else "TAS refusal score >= 0.5 (regular expressions; no embedder given)"
    lines = [f"PPL attack: PPL(question + answer) > {PPL_THRESHOLD:g} under {a.ref_model}.  Consistency attack: {det}.", "",
             "ASR = share of FORGET responses flagged; FPR = the same on HOLDOUT responses (authors the model never saw).", "",
             "| model | PPL median (forget) | PPL ASR | PPL FPR | consistency ASR | consistency FPR | degenerate (forget) |", "|---|---|---|---|---|---|---|"]
    for label, r in rows:
        f, h = r.get("forget"), r.get("holdout")
        g = lambda d, k: "-" if d is None else f"{d[k]:.2f}"
        lines.append(f"| {label} | {g(f, 'ppl_median')} | {g(f, 'asr_ppl')} | {g(h, 'asr_ppl')} | {g(f, 'asr_consistency')} | {g(h, 'asr_consistency')} | {g(f, 'degenerate')} |")
    md = "\n".join(lines)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
