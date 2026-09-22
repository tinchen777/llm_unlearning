"""The FWT Hydra configs must compose against the project's own config tree."""

from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fwt.tests._stubs import stub_optional_deps  # noqa: E402

stub_optional_deps()

from hydra import compose, initialize_config_dir  # noqa: E402

from fwt._paths import CONFIG_DIR  # noqa: E402
from fwt.hydra_plugin import register_fwt_configs  # noqa: E402

register_fwt_configs()
import fwt.register  # noqa: E402, F401

from data import DATASET_REGISTRY  # noqa: E402
from trainer import TRAINER_REGISTRY  # noqa: E402


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="unlearn", overrides=list(overrides))


def test_components_are_registered():
    assert "NeighborRedirect" in TRAINER_REGISTRY
    assert "NeighborRedirectDataset" in DATASET_REGISTRY


def test_default_experiment_composes():
    cfg = _compose("experiment=unlearn/tofu/fwt", "task_name=fwt/test")

    assert cfg.trainer.handler == "NeighborRedirect"
    assert cfg.collator.DataCollatorForSupervisedDataset.args.index == "index", (
        "the trainer looks up cached targets by row index"
    )
    method = cfg.trainer.method_args
    assert list(method.layers) == [7]
    assert method.target_source == "precompute"
    assert method.position_strategy == "last_prompt"
    assert method.contrastive_weight == 0.0

    forget = cfg.data.forget.TOFU_QA_forget_neighbor
    assert forget.handler == "NeighborRedirectDataset"
    assert forget.args.hf_args.name == "forget10"
    assert forget.args.neighbors_path.endswith("neighbors_forget10.json")
    assert forget.args.return_neighbors is False
    assert cfg.mode == "unlearn"


def test_overrides_reach_the_method_args():
    cfg = _compose(
        "experiment=unlearn/tofu/fwt",
        "task_name=fwt/test",
        "forget_split=forget05",
        "trainer.method_args.layers=[5,7]",
        "trainer.method_args.contrastive_weight=0.5",
    )
    assert list(cfg.trainer.method_args.layers) == [5, 7]
    assert cfg.trainer.method_args.contrastive_weight == 0.5
    assert cfg.data.forget.TOFU_QA_forget_neighbor.args.hf_args.name == "forget05"
    assert cfg.data.forget.TOFU_QA_forget_neighbor.args.neighbors_path.endswith(
        "neighbors_forget05.json"
    ), "the neighbour bank must follow the forget split"


def test_variant_experiments_compose():
    contrastive = _compose("experiment=unlearn/tofu/fwt_contrastive", "task_name=t")
    assert contrastive.trainer.method_args.contrastive_weight == 0.5

    online = _compose("experiment=unlearn/tofu/fwt_online", "task_name=t")
    assert online.trainer.method_args.target_source == "online"
    assert online.trainer.method_args.target_mode == "sample"
    assert online.data.forget.TOFU_QA_forget_neighbor.args.return_neighbors is True, (
        "online targets need the neighbour samples in the batch"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all config tests passed")
