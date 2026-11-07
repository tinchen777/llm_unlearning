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
    with open(data / "neighbor.csv", "w", newline="") as f:  # neighbours of the 2 forget authors (numbered 5, 6): 2 train (rank 1, 2) + 1 heldout
        w = csv.DictWriter(f, fieldnames=["question", "answer", "author_id", "neighbor_id", "split", "train_rank", "qa_id", "source_question"])
        w.writeheader()
        forget_rows = json.loads((data / "forget.json").read_text())
        for author in (5, 6):
            for nb in range(3):
                for qa_id in range(4):
                    w.writerow({"question": f"nb {sent(4)} ?", "answer": sent(4) + " .", "author_id": author, "neighbor_id": nb,
                                "qa_id": qa_id, "source_question": forget_rows[(author - 5) * 20 + qa_id]["question"],
                                "split": "train" if nb < 2 else "heldout", "train_rank": nb + 1 if nb < 2 else -1})
    return root


def common_args(assets: Path, out_root: Path, experiment: str, task_name: str):
    a = assets
    args = [
        f"experiment=unlearn/tofu/{experiment}", "model=tofu/tofu_Llama-3.2-1B-Instruct_full",
        f"model.pretrained.name_or_path={a / 'llama'}", f"model.tokenizer.name_or_path={a / 'llama'}",
        "model.pretrained.args.device_map=cpu", "model.pretrained.dtype=float32",
        "model.template_args.apply_chat_template=false",
        'model.template_args.user_start_tag="Q: "', 'model.template_args.user_end_tag=" "',
        'model.template_args.asst_start_tag="A: "', 'model.template_args.asst_end_tag=""',
        'model.template_args.system_prompt_with_special_tokens=""',
        "generation_args.max_new_tokens=6", f"paths.root_dir={out_root}", f"task_name={task_name}",
    ]
    for split in ("forget", "retain", "holdout"):
        args += [f"data.{split}.dataset.args.hf_args.path=json", f"~data.{split}.dataset.args.hf_args.name",
                 f"+data.{split}.dataset.args.hf_args.data_files={a / 'data' / (split + '.json')}"]
    args += ["data.retain.dataset.args.hf_args.split=train", "data.holdout.dataset.args.hf_args.split=train"]
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
    args += ["data/datasets@data.ref.dataset=LUNAR/LUNAR_harmful", f"data.ref.dataset.args.hf_args.data_files={assets / 'data' / 'ref.json'}",
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

    run("LUNAR/w_down.py", args + [f"wdown.layer_selection_path={out / 'c1_all' / 'layer_selection.json'}", "wdown.epochs=3", "wdown.batch_size=32", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4"])
    check_run_folder(out / "c1_all" / f"l{selection['selected_layer']}", ["forget", "retain", "holdout"])


def test_near_chain_writes_the_documented_layout(assets, tmp_path):
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "smoke")
    args += [f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
             "basic.author_offset=5", "basic.k=2", "ruv.ks=[1,2]", "ruv.batch_size=8",
             "select.max_samples=8", "select.batch_size=4", "select.layer_fraction=null",
             "wdown.epochs=3", "wdown.batch_size=32", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4",
             "wdown.retain_max_samples=16"]
    run("NEAR/r_uv.py", args)
    out = out_root / "saves" / "NEAR" / "smoke"
    assert (out / "r_uv.pt").exists() and (out / "r_uv_summary.json").exists()
    saved = torch.load(out / "r_uv.pt", weights_only=False)
    assert {"forget_split", "num_forget", "authors", "model"} <= set(saved)  # identity, checked by the next stages

    run("NEAR/select_layer.py", args)
    selection = json.loads((out / "k2_c1_eoi" / "layer_selection.json").read_text())
    assert selection["selected_layer"] in range(4)
    assert all((out / "k2_c1_eoi" / f"responses_steer_l{l}.json").exists() for l in range(4))
    # the detectability of the steered answers and the footprint of the shift are recorded for every layer (not used by default)
    assert all({"ud", "ud_tas", "footprint", "selection_score"} <= set(v) and 0.0 <= v["ud"] <= 1.0 and v["footprint"] > 0
               for v in selection["per_layer"].values())
    assert selection["ud_weight"] == 0.0 and selection["max_ud"] is None
    assert all(abs(v["selection_score"] - v["unseen_gap"]) < 1e-12 for v in selection["per_layer"].values())  # off: the old criterion

    run("NEAR/w_down.py", args + [f"wdown.layer_selection_path={out / 'k2_c1_eoi' / 'layer_selection.json'}"])
    check_run_folder(out / "k2_c1_eoi" / f"l{selection['selected_layer']}", ["forget", "neighbor_heldout", "retain", "holdout"])

    # lineage: w_down took everything from the selection file (no k / coeff / shift_mode on its command line)
    summary = json.loads((out / "k2_c1_eoi" / f"l{selection['selected_layer']}" / "w_down_summary.json").read_text())
    assert summary["k"] == 2 and summary["layer_selection_path"].endswith("layer_selection.json")
    assert selection["r_uv_path"].endswith("r_uv.pt") and selection["forget_split"] == saved["forget_split"]

    # a config value that disagrees with the selection file is rejected
    bad = subprocess.run([sys.executable, str(ROOT / "src" / "METHOD" / "NEAR/w_down.py"), *args,
                          f"wdown.layer_selection_path={out / 'k2_c1_eoi' / 'layer_selection.json'}", "wdown.coeff=7"],
                         cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert bad.returncode != 0 and "disagrees" in bad.stdout + bad.stderr

    # the probe of an edited model takes the hook point / positions / layers / author numbering from r_uv.pt
    run("NEAR/probe.py", args + [f"probe.activations.from_r_uv={out / 'r_uv.pt'}", "probe.batch_size=4", "probe.responses=null",
                                 "probe.activations.max_act_samples=8", "task_name=smoke_probe"])
    probe_saved = torch.load(out_root / "saves" / "NEAR" / "smoke_probe" / "activations.pt", weights_only=False)
    assert probe_saved["positions"] == saved["positions"] and probe_saved["layers"] == saved["layers"]


def test_near_w_down_can_train_on_the_answer_tokens_too(assets, tmp_path):
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "ans")
    args += [f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
             "basic.author_offset=5", "basic.k=2", "basic.shift_mode=all", "ruv.ks=[1,2]", "ruv.batch_size=8",
             "select.max_samples=8", "select.batch_size=4", "select.layer_fraction=null",
             "wdown.epochs=2", "wdown.batch_size=32", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4",
             "wdown.retain_max_samples=16", "wdown.responses_splits=[forget]"]
    run("NEAR/r_uv.py", args)
    run("NEAR/select_layer.py", args)
    out = out_root / "saves" / "NEAR" / "ans" / "k2_c1_all"
    sel_path = out / "layer_selection.json"
    layer = json.loads(sel_path.read_text())["selected_layer"]

    run("NEAR/w_down.py", args + [f"wdown.layer_selection_path={sel_path}", "wdown.layers=[1]"])
    prompt_only = json.loads((out / "l1" / "w_down_summary.json").read_text())
    assert prompt_only["include_answer"] is False

    run("NEAR/w_down.py", args + [f"wdown.layer_selection_path={sel_path}", "wdown.layers=[1]", "wdown.include_answer=true",
                                  "task_name=ans_answer"])
    with_answer = json.loads((out_root / "saves" / "NEAR" / "ans_answer" / "k2_c1_all" / "l1" / "w_down_summary.json").read_text())
    assert with_answer["include_answer"] is True
    # prompt + answer: more tokens than the prompt alone, for the forget samples and for the retain ones
    assert with_answer["num_tokens"]["forget"] > prompt_only["num_tokens"]["forget"]
    assert with_answer["num_tokens"]["retain"] > prompt_only["num_tokens"]["retain"]

    # answer_coeff scales only the answer tokens of the forget samples (the same tokens are recorded)
    run("NEAR/w_down.py", args + [f"wdown.layer_selection_path={sel_path}", "wdown.layers=[1]", "wdown.include_answer=true",
                                  "wdown.answer_coeff=0.5", "task_name=ans_half"])
    half = json.loads((out_root / "saves" / "NEAR" / "ans_half" / "k2_c1_all" / "l1" / "w_down_summary.json").read_text())
    assert half["answer_coeff"] == 0.5 and with_answer["answer_coeff"] == 1.0
    assert half["num_tokens"] == with_answer["num_tokens"]

    # `eoi` counts its positions from the end of the prompt: not combinable with the answer tokens
    bad = subprocess.run([sys.executable, str(ROOT / "src" / "METHOD" / "NEAR/w_down.py"), *args, "basic.shift_mode=eoi",
                          "wdown.include_answer=true", "wdown.layer_selection_path=" + str(sel_path), "wdown.layers=[1]",
                          "wdown.shift_mode=eoi"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert bad.returncode != 0


def test_layer_increments_add_up_to_the_vector_of_the_last_layer():
    sys.path.insert(0, str(ROOT / "src"))
    from METHOD._common.editing import layer_increments

    r = {5: {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])},
         7: {"a": torch.tensor([3.0, 1.0]), "b": torch.tensor([1.0, 4.0])},
         6: {"a": torch.tensor([2.0, 0.5]), "b": torch.tensor([0.5, 2.0])}}
    inc = layer_increments(r)
    assert list(inc) == [5, 6, 7]
    assert torch.equal(inc[5]["a"], r[5]["a"])  # the lowest layer takes its whole vector
    for author in "ab":
        assert torch.allclose(sum(inc[_l][author] for _l in inc), r[7][author])  # telescoping


def test_near_w_down_edits_several_layers_with_shares_of_the_shift(assets, tmp_path):
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "multi")
    args += [f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
             "basic.author_offset=5", "basic.k=2", "basic.shift_mode=all", "ruv.ks=[1,2]", "ruv.batch_size=8",
             "select.max_samples=8", "select.batch_size=4", "select.layer_fraction=null",
             "wdown.epochs=2", "wdown.batch_size=32", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4",
             "wdown.retain_max_samples=16", "wdown.responses_splits=[forget]", "wdown.save_model=false"]
    run("NEAR/r_uv.py", args)
    run("NEAR/select_layer.py", args)
    sel_path = out_root / "saves" / "NEAR" / "multi" / "k2_c1_all" / "layer_selection.json"

    for name, extra in (("seq", ["wdown.sequential=true"]), ("indep", [])):
        run("NEAR/w_down.py", args + [f"wdown.layer_selection_path={sel_path}", "wdown.layers=[2,1]", "wdown.multi_layer=increment",
                                      f"task_name=multi_{name}"] + extra)
        summary = json.loads((out_root / "saves" / "NEAR" / f"multi_{name}" / "k2_c1_all" / "l1-2" / "w_down_summary.json").read_text())
        assert summary["layers"] == [1, 2] and summary["multi_layer"] == "increment" and summary["sequential"] == bool(extra)
        assert set(summary["achieved_vs_intended"]) == {"1", "2"} and set(summary["per_layer"]) == {"1", "2"}

    bad = subprocess.run([sys.executable, str(ROOT / "src" / "METHOD" / "NEAR/w_down.py"), *args,
                          f"wdown.layer_selection_path={sel_path}", "wdown.layers=[1,2]", "wdown.multi_layer=nonsense"],
                         cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert bad.returncode != 0 and "multi_layer" in bad.stdout + bad.stderr


def test_near_layer_selection_can_weigh_the_detectability_and_the_footprint(assets, tmp_path):
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "udsel") + [
        f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
        "basic.author_offset=5", "basic.k=2", "basic.shift_mode=all", "ruv.ks=[1,2]", "ruv.batch_size=8",
        "select.max_samples=8", "select.batch_size=4", "select.layer_fraction=null"]
    run("NEAR/r_uv.py", args)
    run("NEAR/select_layer.py", args + ["select.ud_weight=1.0", "select.footprint_weight=0.5", "select.max_ud=1.0"])
    sel = json.loads((out_root / "saves" / "NEAR" / "udsel" / "k2_c1_all" / "layer_selection.json").read_text())
    assert sel["ud_weight"] == 1.0 and sel["footprint_weight"] == 0.5 and sel["max_ud"] == 1.0
    per_layer = sel["per_layer"]
    for v in per_layer.values():
        assert abs(v["selection_score"] - (v["unseen_gap"] + v["ud"] + 0.5 * v["footprint"])) < 1e-9
    assert sel["selected_layer"] == min(sel["ranking"], key=lambda l: per_layer[str(l)]["selection_score"])
    bad = subprocess.run([sys.executable, str(ROOT / "src" / "METHOD" / "NEAR/select_layer.py"), *args,
                          "select.criterion=answer_logprob", "select.ud_weight=1.0"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert bad.returncode != 0 and "ud_weight" in bad.stdout + bad.stderr


def test_near_w_down_kl_refinement_runs_end_to_end(assets, tmp_path):
    """Stage 2: the W_down of the edited layers are refined through the whole model with the KL term; the weights change and the
    history is logged. A neighbour csv whose `source_question` does not match the forget split is an error."""
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "klr")
    args += [f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
             "basic.author_offset=5", "basic.k=2", "basic.shift_mode=all", "ruv.ks=[1,2]", "ruv.batch_size=8",
             "select.max_samples=8", "select.batch_size=4", "select.layer_fraction=null",
             "wdown.epochs=2", "wdown.batch_size=8", "wdown.max_gen_samples=8", "wdown.gen_batch_size=4",
             "wdown.retain_max_samples=16", "wdown.responses_splits=[forget]", "wdown.include_answer=true"]
    run("NEAR/r_uv.py", args)
    run("NEAR/select_layer.py", args)
    sel_path = out_root / "saves" / "NEAR" / "klr" / "k2_c1_all" / "layer_selection.json"
    kl = ["wdown.kl.enabled=true", "wdown.kl.steps=3", "wdown.kl.batch_size=4", "wdown.kl.n_tokens=3",
          "wdown.kl.retain_max_samples=8", "wdown.kl.forget_prob_weight=1", "wdown.kl.forget_prob_target=unseen", "wdown.kl.forget_prob_mode=match"]
    run("NEAR/w_down.py", args + [f"wdown.layer_selection_path={sel_path}", "wdown.layers=[1,2]", *kl, "task_name=klr_on"])
    summary = json.loads((out_root / "saves" / "NEAR" / "klr_on" / "k2_c1_all" / "l1-2" / "w_down_summary.json").read_text())
    assert summary["kl"]["steps"] == 3 and len(summary["kl"]["history"]) == 3
    assert all(h["kl"] >= 0 for h in summary["kl"]["history"])
    assert summary["kl"]["forget_prob_weight"] == 1 and all(0 <= h["forget_prob"] <= 1 and h["forget_prob_penalty"] >= 0 for h in summary["kl"]["history"])
    assert any(h["forget_prob"] > 0 for h in summary["kl"]["history"])
    assert 0 < summary["kl"]["forget_prob_target"] < 1 and summary["kl"]["forget_prob_mode"] == "match"  # `unseen` was resolved to a number (the original model on the holdout answers)
