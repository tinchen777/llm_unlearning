"""The `responses` metric (evals/metrics/memorization.py, configs/eval/tofu_metrics/splits_responses*.yaml) and the data loaders
it uses, on a tiny LOCAL model (no network, no GPU):

    python -m pytest tests/test_eval_responses.py -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_smoke import make_assets  # noqa: E402

from data import Data  # noqa: E402


class Numbers(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, i):
        return {"x": i}


def test_data_loader_uses_the_collator_the_subset_size_and_the_seed():
    collate = lambda samples: {"x": torch.tensor([s["x"] for s in samples]), "marker": True}
    data = Data("a", Numbers(), collate, split="s")
    loader = data.set_loader(batch_size=4, max_samples=6, sampling_seed=7).loader()
    batches = list(loader)
    assert all(b["marker"] for b in batches), "the collator of the split must reach the DataLoader"
    seen = torch.cat([b["x"] for b in batches]).tolist()
    assert len(seen) == 6 and seen == sorted(seen) and len(set(seen)) == 6
    again = torch.cat([b["x"] for b in data.set_loader(batch_size=3, max_samples=6, sampling_seed=7).loader()]).tolist()
    other = torch.cat([b["x"] for b in data.set_loader(batch_size=3, max_samples=6, sampling_seed=8).loader()]).tolist()
    assert again == seen, "the same seed -> the same fixed subset"
    assert other != seen, "another seed -> another subset"
    assert len(data.set_loader(batch_size=4).loader().dataset) == 10  # max_samples=None -> everything


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return make_assets(tmp_path_factory.mktemp("assets"))


def run_eval(assets: Path, out_root: Path, task: str, max_samples: str, extra=(), expect_failure: bool = False):
    def metric(split):
        return f"eval.tofu.metrics.{split}_responses"
    args = [
        sys.executable, str(ROOT / "src" / "eval.py"), "--config-name=eval", "eval=probe",
        "model=tofu/tofu_Llama-3.2-1B-Instruct_full",
        f"model.pretrained.name_or_path={assets / 'llama'}", f"model.tokenizer.name_or_path={assets / 'llama'}",
        "model.pretrained.args.device_map=cpu", "model.pretrained.dtype=float32",
        "model.template_args.apply_chat_template=false",
        'model.template_args.user_start_tag="Q: "', 'model.template_args.user_end_tag=" "',
        'model.template_args.asst_start_tag="A: "', 'model.template_args.asst_end_tag=""',
        'model.template_args.system_prompt_with_special_tokens=""',
        f"paths.root_dir={out_root}", f"task_name={task}",
        f"{metric('neighbor')}.data.eval.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
    ]
    for split in ("forget", "retain", "holdout"):
        hf = f"{metric(split)}.data.eval.dataset.args.hf_args"
        args += [f"{hf}.path=json", f"~{hf}.name", f"+{hf}.data_files={assets / 'data' / (split + '.json')}"]
    for split in ("forget", "retain", "holdout", "neighbor"):
        args += [f"{metric(split)}.generation_args.max_new_tokens=6", f"{metric(split)}.loader_args.max_samples={max_samples}",
                 f"{metric(split)}.text_metrics_args.tas_embedder={assets / 'encoder'}"]  # the semantic channel of the TAS refusal score, on the tiny encoder
    # the activations: the same local data
    act = "eval.tofu.metrics.activation_probe"
    args += [f"{act}.loader_args.max_samples={max_samples}", f"{act}.data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}"]
    for split in ("forget", "retain", "holdout"):
        hf = f"{act}.data.{split}.dataset.args.hf_args"
        args += [f"{hf}.path=json", f"~{hf}.name", f"+{hf}.data_files={assets / 'data' / (split + '.json')}"]
    proc = subprocess.run(args + list(extra), cwd=ROOT, capture_output=True, text=True, timeout=900,
                          env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    if expect_failure:
        assert proc.returncode != 0
        return proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}"
    return out_root / "saves" / "eval" / task


@pytest.mark.smoke
def test_responses_metric_exposes_its_summary_and_keeps_the_records(assets, tmp_path):
    out = run_eval(assets, tmp_path / "out", "resp", "8")
    sizes = {"forget": 40, "retain": 40, "holdout": 20, "neighbor": 8}  # neighbor: the held-out neighbours only
    logs = json.loads((out / "TOFU_EVAL.json").read_text())
    summary = json.loads((out / "TOFU_SUMMARY.json").read_text())
    for name in sizes:
        main = logs[f"{name}_responses"]
        assert main["agg_value"] is None and main["summary"]["n"] == 8
        assert len(main["value_by_index"]) == 8, f"{name}: max_samples=8 must give exactly 8 records"
        indices = sorted(int(i) for i in main["value_by_index"])
        assert len(set(indices)) == 8 and max(indices) < sizes[name]
        assert {"input", "ground_truth", "generation", "rougeL_recall", "answer_logprob", "refusal", "degenerate",
                "hallucination"} <= set(next(iter(main["value_by_index"].values())))
        # the exposed keys are metrics of their own: only `agg_value`, same level as the other metrics, in the summary file
        for key in ("rougeL_recall", "answer_prob", "refusal_rate", "degenerate_rate", "hallucination_rate"):
            assert set(logs[f"{name}_{key}"]) == {"agg_value"}
            assert summary[f"{name}_{key}"] == main["summary"][key]
        assert f"{name}_responses" not in summary, "the main metric has no agg_value"
        assert len(json.loads((out / f"{name}_responses.json").read_text())) == 8
    neighbor = json.loads((out / "neighbor_responses.json").read_text())
    assert {r["author_id"] for r in neighbor} <= {5, 6}, "the metadata of the neighbour csv reaches the records"

    # the TAS refusal score and the repetition of every answer, and the detectability computed from the answers and the probe
    for name in sizes:
        main = logs[f"{name}_responses"]
        assert all(0.0 <= r["tas_refusal"] <= 1.0 and "repetition" in r for r in main["value_by_index"].values())
        assert {"tas_refusal_mean", "tas_refusal_rate", "repetition_mean", "dup_rate"} <= set(main["summary"])
        assert summary[f"{name}_tas_refusal_rate"] == main["summary"]["tas_refusal_rate"]
    ud = logs["unlearning_detectability"]
    assert ud["agg_value"] == ud["summary"]["ud"] == ud["summary"]["bb"] and 0.0 <= ud["agg_value"] <= 1.0
    assert {"bb", "bb_tas", "wb_sup", "wb_unsup", "ud_max", "forget_tas_rate", "benign_tas_rate"} <= set(ud["summary"])
    assert summary["ud_bb"] == ud["summary"]["bb"] and summary["ud_wb_unsup"] == ud["summary"]["wb_unsup"]
    assert "anom_auc[forget | benign]" in next(iter(logs["activation_probe"]["value_by_layer"].values()))

    # no max_samples -> every sample
    full = run_eval(assets, tmp_path / "out", "resp_full", "null")
    for name, n in sizes.items():
        assert len(json.loads((full / f"{name}_responses.json").read_text())) == n


@pytest.mark.smoke
def test_activation_probe_takes_its_settings_from_an_r_uv_and_exposes_the_aucs(assets, tmp_path):
    from test_pipeline_smoke import common_args, run

    out_root = tmp_path / "out"
    ruv_args = common_args(assets, out_root, "NEAR", "ruv") + [
        f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
        "basic.author_offset=5", "ruv.ks=[1,2]", "ruv.batch_size=8",
    ]
    run("NEAR/r_uv.py", ruv_args)
    r_uv_path = out_root / "saves" / "NEAR" / "ruv" / "r_uv.pt"
    saved = torch.load(r_uv_path, weights_only=False)

    act = "eval.tofu.metrics.activation_probe"
    out = run_eval(assets, out_root, "act", "8",
                   [f"{act}.r_uv_path={r_uv_path}", f"{act}.focus_layer=2", "eval.tofu.forget_split=" + saved["forget_split"]])
    logs = json.loads((out / "TOFU_EVAL.json").read_text())
    summary = json.loads((out / "TOFU_SUMMARY.json").read_text())
    main = logs["activation_probe"]
    assert main["agg_value"] is None and set(main["value_by_layer"]) == {"0", "1", "2", "3"}
    assert main["summary"]["auc_forget_neighbor_focus"] == main["value_by_layer"]["2"]["auc[forget | neighbor]"]
    assert main["summary"]["auc_forget_neighbor_max"] == max(v["auc[forget | neighbor]"] for v in main["value_by_layer"].values())
    assert 0.0 <= summary["activation_auc_forget_neighbor_max"] <= 1.0
    assert "activation_auc_group_forget_neighbor_max" not in summary, "2 authors: the group-wise AUC does not exist (None is not summarised)"
    # the seen | unseen pair is published as `seen_unseen` whatever the splits are: the metric names do not depend on them
    pair = "auc[forget+retain | neighbor+holdout]"
    assert main["summary"]["auc_seen_unseen_max"] == max(v[pair] for v in main["value_by_layer"].values())
    assert 0.0 <= summary["activation_auc_seen_unseen_max"] <= 1.0
    out_unlearned = run_eval(assets, out_root, "act_unlearned", "8", [
        f"{act}.r_uv_path={r_uv_path}", "eval.tofu.forget_split=" + saved["forget_split"],
        f"{act}.seen=[retain]", f"{act}.unseen=[forget,neighbor,holdout]",
    ])
    main_unlearned = json.loads((out_unlearned / "TOFU_EVAL.json").read_text())["activation_probe"]
    pair = "auc[retain | forget+neighbor+holdout]"
    assert pair in next(iter(main_unlearned["value_by_layer"].values()))
    assert main_unlearned["summary"]["auc_seen_unseen_max"] == max(v[pair] for v in main_unlearned["value_by_layer"].values())
    assert "activation_auc_seen_unseen_max" in json.loads((out_unlearned / "TOFU_SUMMARY.json").read_text())
    assert json.loads((out / "activations_summary.json").read_text())["layers"].keys() == main["value_by_layer"].keys()
    probe_saved = torch.load(out / "activations.pt", weights_only=False)
    assert probe_saved["positions"] == saved["positions"] and probe_saved["layers"] == saved["layers"]

    # another forget split than the one r_uv.pt was computed on is refused
    message = run_eval(assets, out_root, "act_bad", "8", [f"{act}.r_uv_path={r_uv_path}", "eval.tofu.forget_split=forget05"],
                       expect_failure=True)
    assert "forget_split" in message


def test_hallucination_is_computed_from_the_other_text_metrics_of_the_same_record():
    from evals.response.text_metric_utils import TextMetric

    metric = TextMetric.use_all(hallucination_rouge=0.3)
    right = metric.compute_metrics({"ground_truth": "the author was born in Taipei", "generation": "the author was born in Taipei"})
    wrong = metric.compute_metrics({"ground_truth": "the author was born in Taipei", "generation": "bananas are yellow"})
    refused = metric.compute_metrics({"ground_truth": "the author was born in Taipei", "generation": "I apologize, but this information cannot be provided."})
    assert right["rougeL_recall"] == 1.0 and right["hallucination"] is False
    assert wrong["hallucination"] is True  # confident answer, far from the ground truth
    assert refused["refusal"] is True and refused["hallucination"] is False
