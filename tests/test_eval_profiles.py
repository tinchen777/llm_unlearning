"""The fast evaluation profile (configs/eval/tofu_fast.yaml): a subset of the metrics of tofu.yaml, selectable on the command line also on top of
the probe experiment (which sets `override /eval: tofu`), with everything but the metric list identical."""
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
BASE = ["task_name=t", "model=tofu/tofu_Llama-3.2-1B-Instruct_full"]


def metrics(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="eval", overrides=BASE + overrides)
    return cfg.eval.tofu, set(cfg.eval.tofu.metrics)


def test_fast_profile_is_a_subset_that_keeps_the_analysis_metrics_and_the_probe():
    full_cfg, full = metrics(["experiment=eval/tofu/probe"])
    fast_cfg, fast = metrics(["experiment=eval/tofu/probe", "eval=tofu_fast"])
    assert fast < full
    assert {"forget_Q_A_Prob", "extraction_strength", "mia_loss", "model_utility", "forget_responses", "retain_responses",
            "holdout_responses", "neighbor_responses", "activation_probe", "unlearning_detectability"} <= fast
    assert not fast & {"mia_gradnorm", "mia_zlib", "mia_min_k_plus_plus", "mia_min_k", "privleak", "exact_memorization",
                       "forget_quality", "forget_truth_ratio"}
    for key in ("handler", "batch_size", "retain_logs_path", "probe_ref_path", "question_key"):
        assert fast_cfg[key] == full_cfg[key]
    for name in fast:  # the shared metrics are configured identically
        assert fast_cfg.metrics[name] == full_cfg.metrics[name]
