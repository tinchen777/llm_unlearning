"""End-to-end smoke test of the LUNAR and NEAR scripts on a tiny LOCAL model (no network, no GPU).

Runs the real hydra entry points in fresh processes - r_uv -> select_layer -> w_down, the same order as the shell scripts -
with a random 4-layer Llama, a word-level tokenizer, a tiny encoder (the sentence embedder of LUNAR's layer selection) and
json / csv data, and checks the OUTPUT LAYOUT of the docs in the file headers:

    <output_dir>/r_uv.pt                                          shared by every setting
    <output_dir>/<setting>/layer_selection.json, responses_steer_l<l>.json
    <output_dir>/<setting>/l<layer>/model, w_down.pt, w_down_summary.json, responses_{before,after}_*

    python -m pytest tests/test_pipeline_smoke.py -q          (about 4 minutes)    deselect with -m "not smoke"
"""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.smoke


def make_assets(root: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    root.mkdir(parents=True, exist_ok=True)
    words = ["Q:", "A:", "who", "what", "where", "is", "the", "of", "a", "b", "c", "d", "e", "f", "g", "h", "?", ".", "cannot",
             "know", "I", "apologize", "author", "book", "born", "genre", "name", "bad", "things", "do", "how", "to",
             "Where", "was", "this", "born?"] + [f"w{i}" for i in range(40)]
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(0)
    llama = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256, eos_token_id=1, pad_token_id=0))
    llama.save_pretrained(root / "llama")
    tokenizer.save_pretrained(root / "llama")
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64)).save_pretrained(root / "encoder")
    tokenizer.save_pretrained(root / "encoder")

    rnd = random.Random(0)
    sent = lambda n: " ".join(rnd.choice(words[3:]) for _ in range(n))
    qa = lambda n, p: [{"question": f"{p} {sent(rnd.randint(3, 6))} ?", "answer": sent(rnd.randint(3, 8)) + " ."} for _ in range(n)]
    data = root / "data"
    data.mkdir(exist_ok=True)
    for name, n in [("forget", 40), ("retain", 40), ("holdout", 20)]:
        (data / f"{name}.json").write_text(json.dumps(qa(n, name)))
    (data / "ref.json").write_text(json.dumps([{"instruction": f"how to do bad things {sent(3)} ?"} for _ in range(12)]))
    with open(data / "neighbor.csv", "w", newline="") as f:  # neighbours of the 2 forget authors (numbered 5, 6)
        w = csv.DictWriter(f, fieldnames=["question", "answer", "author_id", "neighbor_id"])
        w.writeheader()
        for author in (5, 6):
            for nb in range(3):
                for _ in range(4):
                    w.writerow({"question": f"nb {sent(4)} ?", "answer": sent(4) + " .", "author_id": author, "neighbor_id": nb})
    return root


def common_args(assets: Path, out_root: Path, experiment: str, task_name: str):
    a = assets
    args = [
        f"experiment=generate/{experiment}", "model=tofu/tofu_Llama-3.2-1B-Instruct_full",
        f"model.pretrained.name_or_path={a / 'llama'}", f"model.tokenizer.name_or_path={a / 'llama'}",
        "model.pretrained.args.device_map=cpu", "model.pretrained.dtype=float32",
        "model.template_args.apply_chat_template=false",
        'model.template_args.user_start_tag="Q: "', 'model.template_args.user_end_tag=" "',
        'model.template_args.asst_start_tag="A: "', 'model.template_args.asst_end_tag=""',
        'model.template_args.system_prompt_with_special_tokens=""',
        "generation_args.max_new_tokens=6", f"paths.root_dir={out_root}", f"task_name={task_name}",
    ]
    for split in ("forget", "retain", "holdout"):
        args += [f"data.{split}.datasets.args.hf_args.path=json", f"~data.{split}.datasets.args.hf_args.name",
                 f"+data.{split}.datasets.args.hf_args.data_files={a / 'data' / (split + '.json')}"]
    args += ["data.retain.datasets.args.hf_args.split=train", "data.holdout.datasets.args.hf_args.split=train"]
    return args


def run(script: str, args):
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "METHOD" / script), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"{script}\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}"


def check_run_folder(run_dir: Path, splits):
    assert (run_dir / "model" / "config.json").exists() and (run_dir / "w_down.pt").exists()
    summary = json.loads((run_dir / "w_down_summary.json").read_text())
    assert summary["per_layer"] and summary["achieved_vs_intended"]
    for split in splits:
        assert (run_dir / f"responses_before_{split}.json").exists() and (run_dir / f"responses_after_{split}.json").exists()
    assert (run_dir / "responses_before_summary.json").exists() and (run_dir / "responses_after_summary.json").exists()


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return make_assets(tmp_path_factory.mktemp("assets"))


def test_lunar_chain_writes_the_documented_layout(assets, tmp_path):
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "LUNAR", "smoke")
    args += ["data/datasets@data.ref.datasets=LUNAR/LUNAR_harmful", f"data.ref.datasets.args.hf_args.data_files={assets / 'data' / 'ref.json'}",
             "forget_hf_split='train[:20]'"]
    run("LUNAR/r_uv.py", args + ["ruv.batch_size=8"])
    out = out_root / "saves" / "LUNAR" / "smoke"
    saved = torch.load(out / "r_uv.pt", weights_only=False)
    assert saved["positions"] == [-1] and tuple(saved["r_uv"].shape) == (1, 4, 32)  # `A:` is the only token after the question
    assert saved["num_forget"] == 20 and saved["num_ref"] == 12  # LUNAR: the first 20 QAs of the forget split

    run("LUNAR/select_layer.py", args + [f"select.embedder={assets / 'encoder'}", "select.max_samples=8", "select.batch_size=4",
                                         "select.layer_fraction=null"])
    selection = json.loads((out / "c1_all" / "layer_selection.json").read_text())
    assert selection["selected_layer"] in range(4) and len(selection["ranking"]) == 4
    assert all((out / "c1_all" / f"responses_steer_l{l}.json").exists() for l in range(4))

    run("LUNAR/w_down.py", args + ["wdown.epochs=3", "wdown.batch_size=32", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4"])
    check_run_folder(out / "c1_all" / f"l{selection['selected_layer']}", ["forget", "retain", "holdout"])


def test_near_chain_writes_the_documented_layout(assets, tmp_path):
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "smoke")
    args += [f"data.neighbor.datasets.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
             "basic.author_offset=5", "basic.k=2", "ruv.ks=[1,2]", "ruv.batch_size=8",
             "select.max_samples=8", "select.batch_size=4", "select.layer_fraction=null",
             "wdown.epochs=3", "wdown.batch_size=32", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4",
             "wdown.retain_max_samples=16"]
    run("NEAR/r_uv.py", args)
    out = out_root / "saves" / "NEAR" / "smoke"
    assert (out / "r_uv.pt").exists() and (out / "r_uv_summary.json").exists()

    run("NEAR/select_layer.py", args)
    selection = json.loads((out / "k2_c1_eoi" / "layer_selection.json").read_text())
    assert selection["selected_layer"] in range(4)
    assert all((out / "k2_c1_eoi" / f"responses_steer_l{l}.json").exists() for l in range(4))

    run("NEAR/w_down.py", args)
    check_run_folder(out / "k2_c1_eoi" / f"l{selection['selected_layer']}", ["forget", "neighbor", "retain", "holdout"])
