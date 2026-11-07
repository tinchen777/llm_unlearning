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


def test_pareto_collects_the_families_and_matches_the_forgetting(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("pareto", ROOT / "scripts" / "analysis" / "pareto.py")
    pareto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pareto)

    def summary(prob, util, mia=0.5, **more):
        return {"forget_Q_A_Prob": prob, "model_utility": util, "mia_loss": mia, "extraction_strength": 0.05, "retain_rougeL_recall": 0.5, **more}

    def put(path, data):
        path.mkdir(parents=True, exist_ok=True)
        (path / "TOFU_SUMMARY.json").write_text(json.dumps(data))

    near, lunar, base = tmp_path / "near", tmp_path / "lunar", tmp_path / "base"
    put(near / "rw1_ans_increment_seq_c7_l7-8-9_k5_l7-8-9_c7_all", summary(0.2, 0.57, ud_wb_unsup=0.99))
    put(near / "rw1_ans_kl1_increment_seq_c7_l7-8-9_k5_l7-8-9_c7_all", summary(0.22, 0.55))
    put(near / "rw1_ans_kl0_increment_seq_c2_l11-12_k5_l11-12_c2_all", summary(0.75, 0.58))  # kl0 = no KL term: plain NEAR
    put(near / "rw1_ans_kl1_fp1_ftunseen_fmmatch_initoriginal_kfw0_increment_seq_c1_l11-12-13_k5_l11-12-13_c1_all", summary(0.16, 0.57))  # no steering
    put(lunar / "lunar_tofu_M_forget01_LUNAR_harmful_c1.5_l8_all", summary(0.11, 0.40, ud_bb=0.5))
    put(lunar / "lunar_tofu_M_forget05_LUNAR_harmful_c1.5_l8_all", summary(0.01, 0.10))  # another forget split: not collected
    for step, (prob, util, mia) in {0: (0.9, 0.6, 1.0), 2: (0.15, 0.45, 0.45), 4: (0.01, 0.30, 0.02)}.items():
        put(base / "NPO" / "tofu_M_forget01_ep5" / f"checkpoint-{step}" / "evals", summary(prob, util, mia))
    (base / "NPO" / "tofu_M_forget01_ep5" / "checkpoint-9" / "evals").mkdir(parents=True)  # no summary: skipped
    put(base / "RMU" / "tofu_M_forget01_lr5e-5_ep5" / "checkpoint-2" / "evals", summary(0.3, 0.5))
    put(base / "RMU" / "tofu_M_forget01_lr5e-5_ep5_ud" / "checkpoint-2" / "evals", summary(0.3, 0.5, ud_wb_unsup=0.9))  # the re-run with the probe replaces it

    points = pareto.collect(str(near), str(lunar), str(base), "M", "forget01")
    assert sorted(p["family"] for p in points) == ["KL+hinge", "LUNAR", "NEAR", "NEAR", "NEAR+KL", "baseline", "baseline", "baseline"]
    assert [p["run"] for p in points if p["method"] == "RMU"] == ["lr5e-5_ep5_ud@2"] and [p for p in points if p["method"] == "RMU"][0]["ud_wb_unsup"] == 0.9  # step 0 and the split 05 run are left out
    rows = {r["method"]: r for r in pareto.matched(points, tau=0.25, mia_centre=0.48, mia_tol=0.25)}
    put(base / "NPO" / "tofu_M_forget01" / "checkpoint-2" / "evals", summary(0.01, 0.9))  # the earlier 10-epoch protocol (no _ep suffix): not collected
    assert len(pareto.collect(str(near), str(lunar), str(base), "M", "forget01", "all")) == len(points) + 1
    assert rows["NPO"]["run"] == "ep5@2" and rows["NPO"]["reached"]   # the over-forgetting step 4 (mia 0.02) is not retain-like
    assert rows["NEAR"]["model_utility"] == 0.57 and rows["NEAR+KL"]["reached"]
    assert pareto.front([p for p in points if p["family"] == "NEAR"])[0]["forget_Q_A_Prob"] == 0.2
    out = tmp_path / "out"
    pareto.write_csv(points, str(out) + ".csv")
    pareto.plot(points, {"retain": summary(0.17, 0.6)}, 0.25, str(tmp_path / "p"))  # writes p.png and p.pdf
    assert (tmp_path / "p.png").stat().st_size > 1000 and (tmp_path / "p.pdf").stat().st_size > 1000
