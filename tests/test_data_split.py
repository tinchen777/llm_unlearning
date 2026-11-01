"""The data configs / `SplitData` of the "one dataset per split" layout (configs/data/, configs/experiment/unlearn/tofu):

    python -m pytest tests/test_data_split.py -q
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_smoke import make_assets  # noqa: E402

from hydra import compose, initialize_config_dir  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from data import SplitData  # noqa: E402
from utils.config import TrackingConfig  # noqa: E402


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return make_assets(tmp_path_factory.mktemp("assets"))


def compose_cfg(config_name, overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name=config_name, overrides=overrides + ["task_name=t"])


def test_unlearn_split_combines_forget_and_retain_into_one_dataset(assets):
    data_files = lambda split: str(assets / "data" / f"{split}.json")
    overrides = [
        "experiment=unlearn/tofu/default", "model=tofu/tofu_Llama-3.2-1B-Instruct_full",
        "model.template_args.apply_chat_template=false",
        'model.template_args.user_start_tag="Q: "', 'model.template_args.user_end_tag=" "',
        'model.template_args.asst_start_tag="A: "', 'model.template_args.asst_end_tag=""',
        'model.template_args.system_prompt_with_special_tokens=""',
    ]
    for split in ("forget", "retain"):
        hf = f"data.train.dataset.{split}.args.hf_args"
        overrides += [f"{hf}.path=json", f"~{hf}.name", f"+{hf}.data_files={data_files(split)}"]
    cfg = compose_cfg("unlearn", overrides)
    tokenizer = AutoTokenizer.from_pretrained(assets / "llama")
    template_args = TrackingConfig(cfg)["model"]["template_args"]
    split_data = SplitData(TrackingConfig(cfg)["data"], tokenizer=tokenizer, template_args=template_args).set_split_loader(batch_size=4)

    assert split_data.splits == ["train"]
    train = split_data["train"]
    assert len(train.dataset) == 40  # the anchor (forget) decides the length
    sample = train.dataset[0]
    assert set(sample) == {"forget", "retain"} and "input_ids" in sample["forget"]
    batch = next(iter(train.loader()))
    assert set(batch) == {"forget", "retain"} and batch["forget"]["input_ids"].shape[0] == 4


def test_every_experiment_config_composes_with_one_dataset_per_split():
    import glob
    experiments = sorted(
        p[len(str(ROOT / "configs" / "experiment")) + 1:-5] for p in glob.glob(str(ROOT / "configs" / "experiment" / "**" / "*.yaml"), recursive=True)
    )
    assert experiments
    for exp in experiments:
        kind = exp.split("/")[0]
        name = "generate" if exp.startswith(("unlearn/tofu/NEAR", "unlearn/tofu/LUNAR")) else kind
        cfg = compose_cfg(name, [f"experiment={exp}"])
        for split, split_cfg in cfg.get("data", {}).items():
            assert "dataset" in split_cfg and "datasets" not in split_cfg, f"{exp}: data.{split} must hold ONE dataset"
