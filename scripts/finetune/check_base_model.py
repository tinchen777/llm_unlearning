"""Is a base checkpoint loaded correctly? (TEMPORARY, for the 1B model whose evaluation before any training is garbage:
checkpoint-0 of saves/finetune/test/tofu_Llama-3.2-1B-Instruct_ep3_*/ has avg_loss ~21.6 per token and generates "oooo...".)

    python scripts/finetune/check_base_model.py                      # the three default repos (must be in the local HF cache)
    python scripts/finetune/check_base_model.py --repo unsloth/Llama-3.2-1B-Instruct --repo <local dir> [--device cuda]

For every repo: the keys the loader reports missing / unexpected (a missing key = randomly initialised weights), whether the input
embedding and lm_head are tied the way the config says, weight statistics (NaN / inf / huge values), the per-token loss on a plain
English sentence (a healthy 1B Instruct model: ~2-4; garbage: > 10) and a greedy generation, with sdpa and eager attention.
The healthy reference is the official TOFU full model (open-unlearning/tofu_Llama-3.2-1B-Instruct_full), same architecture.
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = "The capital of France is Paris, and the Eiffel Tower is one of the most visited monuments in the world."
DEFAULT_REPOS = [
    "unsloth/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
]


def stats(name, t):
    t = t.detach().float()
    return f"{name}: shape={tuple(t.shape)} mean={t.mean():.4g} std={t.std():.4g} absmax={t.abs().max():.4g} nan={bool(t.isnan().any())} inf={bool(t.isinf().any())}"


def check(repo, device, dtype, attn):
    print(f"\n=== {repo} | dtype={dtype} attn={attn}")
    tok = AutoTokenizer.from_pretrained(repo)
    model, info = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, attn_implementation=attn, output_loading_info=True
    )
    model.to(device).eval()
    print("missing keys   :", info.get("missing_keys"))
    print("unexpected keys:", info.get("unexpected_keys"))
    print("mismatched     :", info.get("mismatched_keys"))
    cfg = model.config
    emb, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    print(f"config.tie_word_embeddings={cfg.tie_word_embeddings}; embed and lm_head share memory: {emb.data_ptr() == head.data_ptr()}")
    print(stats("embed_tokens", emb)); print(stats("lm_head     ", head))
    print(stats("layer0.q_proj", model.model.layers[0].self_attn.q_proj.weight))
    print(stats("layer0.down_proj", model.model.layers[0].mlp.down_proj.weight))
    print(f"tokenizer: len={len(tok)} vocab_size(config)={cfg.vocab_size} pad={tok.pad_token!r} eos={tok.eos_token!r}")
    ids = tok(TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(**ids, labels=ids["input_ids"]).loss.item()
    print(f"per-token loss on a plain sentence: {loss:.3f}   (healthy ~2-4, garbage > 10)")
    try:
        chat = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            out = model.generate(**chat, max_new_tokens=30, do_sample=False)
        print("chat generation:", repr(tok.decode(out[0, chat["input_ids"].shape[1]:], skip_special_tokens=True)))
    except Exception as e:  # e.g. a tokenizer without a chat template
        print(f"chat generation skipped: {type(e).__name__}: {str(e)[:120]}")
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--attn", action="append", default=None, help="sdpa and eager by default")
    args = parser.parse_args()
    print("HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
    for repo in args.repo or DEFAULT_REPOS:
        for attn in args.attn or ["sdpa", "eager"]:
            try:
                check(repo, args.device, getattr(torch, args.dtype), attn)
            except Exception as e:  # not in the cache, no access, ...
                print(f"\n=== {repo} | attn={attn}: FAILED: {type(e).__name__}: {str(e)[:300]}")
