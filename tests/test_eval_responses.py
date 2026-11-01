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
    data = Data("s", {"a": Numbers()}, collate)
    loader = data.set_loader(batch_size=4, max_samples=6, sampling_seed=7).one_loader()
    batches = list(loader)
    assert all(b["marker"] for b in batches), "the collator of the split must reach the DataLoader"
    seen = torch.cat([b["x"] for b in batches]).tolist()
    assert len(seen) == 6 and seen == sorted(seen) and len(set(seen)) == 6
    again = torch.cat([b["x"] for b in data.set_loader(batch_size=3, max_samples=6, sampling_seed=7).one_loader()]).tolist()
    other = torch.cat([b["x"] for b in data.set_loader(batch_size=3, max_samples=6, sampling_seed=8).one_loader()]).tolist()
    assert again == seen, "the same seed -> the same fixed subset"
    assert other != seen, "another seed -> another subset"
    assert len(data.set_loader(batch_size=4).one_loader().dataset) == 10  # max_samples=None -> everything


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return make_assets(tmp_path_factory.mktemp("assets"))


def run_eval(assets: Path, out_root: Path, task: str, extra=()):
    metric = "eval.tofu.metrics.splits_responses"
    args = [
        sys.executable, str(ROOT / "src" / "eval.py"), "--config-name=eval", "eval=responses",
        "model=tofu/tofu_Llama-3.2-1B-Instruct_full",
        f"model.pretrained.name_or_path={assets / 'llama'}", f"model.tokenizer.name_or_path={assets / 'llama'}",
        "model.pretrained.args.device_map=cpu", "model.pretrained.dtype=float32",
        "model.template_args.apply_chat_template=false",
        'model.template_args.user_start_tag="Q: "', 'model.template_args.user_end_tag=" "',
        'model.template_args.asst_start_tag="A: "', 'model.template_args.asst_end_tag=""',
        'model.template_args.system_prompt_with_special_tokens=""',
        f"{metric}.generation_args.max_new_tokens=6", f"paths.root_dir={out_root}", f"task_name={task}",
        f"{metric}.datasets.neighbor.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
    ]
    for split in ("forget", "retain", "holdout"):
        args += [f"{metric}.datasets.{split}.args.hf_args.path=json", f"~{metric}.datasets.{split}.args.hf_args.name",
                 f"+{metric}.datasets.{split}.args.hf_args.data_files={assets / 'data' / (split + '.json')}"]
    proc = subprocess.run(args + list(extra), cwd=ROOT, capture_output=True, text=True, timeout=900,
                          env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}"
    return out_root / "saves" / "eval" / task


@pytest.mark.smoke
def test_responses_metric_writes_records_and_summary_of_every_dataset(assets, tmp_path):
    out = run_eval(assets, tmp_path / "out", "resp", ["eval.tofu.metrics.splits_responses.loader_args.max_samples=8"])
    sizes = {"forget": 40, "retain": 40, "holdout": 20, "neighbor": 24}
    summary = json.loads((out / "responses_summary.json").read_text())
    assert set(summary) == set(sizes)
    for name in sizes:
        records = json.loads((out / f"responses_{name}.json").read_text())
        assert len(records) == 8, f"{name}: max_samples=8 must give exactly 8 records"
        assert {"input", "ground_truth", "generation", "rougeL_recall", "answer_logprob", "refusal", "degenerate",
                "hallucination", "index"} <= set(records[0])
        assert summary[name]["n"] == 8 and "hallucination_rate" in summary[name] and "answer_prob" in summary[name]
        indices = [r["index"] for r in records]
        assert indices == sorted(indices) and len(set(indices)) == 8 and max(indices) < sizes[name]
    neighbor = json.loads((out / "responses_neighbor.json").read_text())
    assert {r["author_id"] for r in neighbor} <= {5, 6}, "the metadata of the neighbour csv reaches the records"

    # no max_samples -> every sample, the evaluator log keeps the (empty) aggregate
    full = run_eval(assets, tmp_path / "out", "resp_full", ["eval.tofu.metrics.splits_responses.loader_args.max_samples=null"])
    for name, n in sizes.items():
        assert len(json.loads((full / f"responses_{name}.json").read_text())) == n
    assert json.loads((full / "TOFU_EVAL.json").read_text())["splits_responses"] == {"agg_value": None}
