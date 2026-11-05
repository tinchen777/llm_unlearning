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
        "experiment=unlearn/tofu/default", "model=tofu/tofu_Llama-3.2-1B-Instruct_full", "trainer=GradAscent",
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
        # the experiments leave the model (and the trainer) to the command line
        required = ["model=tofu/tofu_Llama-3.2-1B-Instruct_full"]
        if kind in ("unlearn", "finetune"):
            required.append("trainer=GradAscent" if kind == "unlearn" else "trainer=finetune")
        cfg = compose_cfg(name, [f"experiment={exp}"] + required)
        for split, split_cfg in cfg.get("data", {}).items():
            assert "dataset" in split_cfg and "datasets" not in split_cfg, f"{exp}: data.{split} must hold ONE dataset"


def test_tofu_experiments_hand_their_splits_to_the_evaluation():
    """`eval.tofu.<x>_split` default to `<x>??` (= must be given): every experiment with a TOFU evaluation has to set them from its own
    `<x>_split`, else the evaluation tries to load the split `retain??` (seen with unlearn/tofu/idk)."""
    import glob
    splits = {"forget_split": "forget05", "retain_split": "retain95", "holdout_split": "holdout05"}
    for path in sorted(glob.glob(str(ROOT / "configs" / "experiment" / "**" / "*.yaml"), recursive=True)):
        exp = path[len(str(ROOT / "configs" / "experiment")) + 1:-5]
        kind = exp.split("/")[0]
        if "tofu" not in exp or kind not in ("unlearn", "finetune", "eval"):
            continue
        required = ["model=tofu/tofu_Llama-3.2-1B-Instruct_full"]
        if kind in ("unlearn", "finetune"):
            required.append("trainer=GradAscent" if kind == "unlearn" else "trainer=finetune")
        name = "generate" if exp.startswith(("unlearn/tofu/NEAR", "unlearn/tofu/LUNAR")) else kind
        cfg = compose_cfg(name, [f"experiment={exp}"] + required)
        if "eval" not in cfg or "tofu" not in cfg.eval:
            continue
        given = [f"{key}={value}" for key, value in splits.items() if key in cfg]
        cfg = compose_cfg(name, [f"experiment={exp}"] + required + given)
        for key, value in splits.items():
            if key in cfg.eval.tofu:
                assert cfg.eval.tofu[key] == value, f"{exp}: eval.tofu.{key} = {cfg.eval.tofu[key]!r}, not set from `{key}`"


def test_forget005_is_the_last_author_of_forget01():
    from data.datasets.datasets_utils import load_hf_dataset, resolve_virtual_split

    assert resolve_virtual_split("forget005") == ("forget01", "train[20:]")  # author 200
    assert resolve_virtual_split("forget005", "train") == ("forget01", "train[20:]")
    assert resolve_virtual_split("forget005_perturbed") == ("forget01_perturbed", "train[20:]")
    assert resolve_virtual_split("forget01", "train") == ("forget01", "train")  # an official split is untouched
    assert resolve_virtual_split(None, None) == (None, None)
    with pytest.raises(ValueError, match="already train\\[20:\\] of"):
        resolve_virtual_split("forget005", "train[:10]")

    import datasets

    seen = []
    original = datasets.load_dataset
    datasets.load_dataset = lambda path, **kwargs: seen.append((path, kwargs)) or []
    try:
        load_hf_dataset("locuslab/TOFU", name="forget005", split="train")
        load_hf_dataset("locuslab/TOFU", name="forget01", split="train")
    finally:
        datasets.load_dataset = original
    assert seen == [("locuslab/TOFU", {"name": "forget01", "split": "train[20:]"}),
                    ("locuslab/TOFU", {"name": "forget01", "split": "train"})]
