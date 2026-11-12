"""Write a model spec (hf:<id/folder> | near:<run folder>, as in sample_consistency.py) to a plain HF folder, so that a shadow edit can start from it (A2: the shadow models are edits of M').
    python scripts/analysis/spa_materialize.py --model near:saves/NEAR/.../l12-13-14 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --out saves/spa/models/3B_act_mprime
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default=None)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if os.path.isfile(os.path.join(a.out, "config.json")):
        print(f"{a.out} exists")
        return
    import sample_consistency as sc
    from transformers import AutoTokenizer
    model = sc.load_model(a.model, a.base)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    print(f"saved {a.model} -> {a.out}")


if __name__ == "__main__":
    main()
