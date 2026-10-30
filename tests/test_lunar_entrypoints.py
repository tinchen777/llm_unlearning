"""The LUNAR scripts (src/METHOD/LUNAR/) are launched as `python src/METHOD/LUNAR/<script>.py` like the NEAR ones; see
tests/test_near_entrypoints.py. `--cfg job` composes the config (incl. the CLI choice of the reference dataset) without loading a model."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["r_uv", "select_layer", "w_down"])
@pytest.mark.parametrize("ref", ["LUNAR/LUNAR_harmful", "LUNAR/LUNAR_unverified"])
def test_entry_point_runs_from_its_own_directory(script, ref):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "METHOD" / "LUNAR" / f"{script}.py"),
         "experiment=unlearn/tofu/LUNAR", "model=tofu/tofu_Llama-3.2-1B-Instruct_full", "task_name=t",
         f"data/datasets@data.ref.datasets={ref}", "--cfg", "job"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "mode: LUNAR" in proc.stdout and ref.split("/")[1] in proc.stdout


def test_scripts_call_existing_files():
    for sh in (ROOT / "scripts" / "LUNAR").glob("*.sh"):
        text = sh.read_text()
        for path in re.findall(r"python (src/\S+\.py)", text):
            assert (ROOT / path).exists(), f"{sh.name}: {path} does not exist"
        for exp in re.findall(r"experiment=(unlearn/tofu/\S+)", text):
            assert (ROOT / "configs" / "experiment" / f"{exp}.yaml").exists(), f"{sh.name}: experiment {exp} does not exist"


@pytest.mark.parametrize("script", ["r_uv", "select_layer", "w_down"])
def test_split_overrides_of_the_shell_scripts_are_valid_hydra_overrides(script):
    """The shell scripts pass "${SPLIT_ARGS[@]}" of _common.sh: `forget_hf_split=train[:20]` UNQUOTED is rejected by Hydra
    ("mismatched input '[' expecting <EOF>"), so the value must reach the script quoted. Run the same call as the scripts."""
    cmd = (
        f'source scripts/LUNAR/_common.sh; python src/METHOD/LUNAR/{script}.py experiment=unlearn/tofu/LUNAR '
        'model=${FULL_MODEL} "${SPLIT_ARGS[@]}" "data/datasets@data.ref.datasets=${REF_DATASET}" '
        'task_name=${TASK_NAME} --cfg job'
    )
    proc = subprocess.run(["bash", "-c", cmd], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-1500:]
    assert "forget_hf_split: train[:20]" in proc.stdout


@pytest.mark.parametrize("coeff, solver", [("1.0", "adam"), ("2.5", "closed_form")])
def test_shell_and_python_agree_on_the_output_folder_names(coeff, solver):
    """4_eval.sh finds the edited model through `config_dir` / `run_dir` of _common.sh; the scripts write it through the
    python functions of the same names. A mismatch means 4_eval.sh evaluates a model that does not exist."""
    sys.path.insert(0, str(ROOT / "src"))
    from METHOD.LUNAR._utils import selection as lunar
    from METHOD.NEAR._utils import neighbors as near

    def shell(method, expr):
        cmd = f'COEFF={coeff} SOLVER={solver}; source scripts/{method}/_common.sh; COEFF={coeff}; SOLVER={solver}; {expr}'
        return subprocess.run(["bash", "-c", cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    out = Path("x")
    assert shell("LUNAR", "config_dir all") == lunar.config_dir(out, float(coeff), "all").name
    assert shell("LUNAR", "run_dir 16") == lunar.run_dir(out, float(coeff), "all", [16], solver).name
    assert shell("NEAR", "config_dir 5 eoi") == near.config_dir(out, 5, float(coeff), "eoi").name
    assert shell("NEAR", "run_dir 16") == near.run_dir(out, 5, float(coeff), "eoi", [16], solver).name
