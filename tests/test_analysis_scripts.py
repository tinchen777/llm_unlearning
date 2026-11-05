"""scripts/analysis/baseline_table.py (the best epoch of a baseline run) and the tuning scripts."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import baseline_table  # noqa: E402


def write_run(run, steps):
    for step, summary in steps.items():
        d = run / f"checkpoint-{step}" / "evals"
        d.mkdir(parents=True)
        (d / "TOFU_SUMMARY.json").write_text(json.dumps(summary))


def test_best_epoch_is_the_largest_forget_quality_within_the_utility_budget(tmp_path):
    run = tmp_path / "NPO" / "tofu_M_forget01_lr1e-4"
    write_run(run, {
        0: {"model_utility": 0.60, "forget_quality": 0.0},
        2: {"model_utility": 0.59, "forget_quality": 0.1},
        4: {"model_utility": 0.58, "forget_quality": 0.4},   # within 5 % of 0.60 (>= 0.57): the best
        6: {"model_utility": 0.50, "forget_quality": 0.9},   # forgets more but the utility is gone
    })
    step, summary, ok = baseline_table.best_epoch(baseline_table.checkpoints(run), 0.05)
    assert (step, ok) == (4, True) and summary["forget_quality"] == 0.4
    # nothing within the budget: the epoch with the best utility, flagged
    run2 = tmp_path / "GA" / "tofu_M_forget01_lr1e-3"
    write_run(run2, {0: {"model_utility": 0.60}, 2: {"model_utility": 0.30, "forget_quality": 0.5}, 4: {"model_utility": 0.40, "forget_quality": 0.9}})
    step, _, ok = baseline_table.best_epoch(baseline_table.checkpoints(run2), 0.05)
    assert (step, ok) == (4, False)
    assert baseline_table.best_epoch([(0, {"model_utility": 0.6})], 0.05) is None  # only the original model


def test_the_tuning_scripts_are_valid_and_print_their_runs():
    for script in ("scripts/unlearn/tune.sh", "scripts/LUNAR/sweep_coeff.sh", "scripts/NEAR/sweep_footprint.sh"):
        assert subprocess.run(["bash", "-n", script], cwd=ROOT).returncode == 0, script
    out = subprocess.run(["bash", "scripts/unlearn/tune.sh", "0"], cwd=ROOT, capture_output=True, text=True, env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin", "LRS": "5e-5 1e-4", "DTYPES": "bf16 float32", "METHODS": "NPO"}).stdout
    assert out.count("1_unlearn.sh") == 4 and "DTYPE=float32" in out and "LR=1e-4" in out
    out = subprocess.run(["bash", "scripts/LUNAR/sweep_coeff.sh", "0"], cwd=ROOT, capture_output=True, text=True, env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin", "COEFFS": "0.5 1", "FORGET_SPLITS": "forget005"}).stdout
    assert "c0.5_all" in out and "c1_all" in out and "forget_split=forget005" in out
