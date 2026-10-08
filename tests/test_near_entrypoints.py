"""The NEAR scripts live in src/METHOD/NEAR/ and are launched as `python src/METHOD/NEAR/<script>.py`.

Two things broke when they were moved from src/: `sys.path[0]` became their own directory (so `import data` /
`model` / `utils` failed) and the relative Hydra `config_path` pointed to src/METHOD/configs. Run every entry point in a
fresh process the way the shell scripts do; `--cfg job` makes Hydra compose and print the config without loading a model.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = ["probe", "r_uv", "select_layer", "w_down"]


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_runs_from_its_own_directory(script):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "METHOD" / "NEAR" / f"{script}.py"),
         "experiment=generate/NEAR", "model=tofu/tofu_Llama-3.2-1B-Instruct_full", "task_name=t", "--cfg", "job"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "mode: NEAR" in proc.stdout and "open-unlearning/tofu_Llama-3.2-1B-Instruct_full" in proc.stdout


def test_scripts_call_existing_files():
    """Every `python src/...py` / `experiment=...` in scripts/NEAR points to something that exists."""
    import re
    for sh in (ROOT / "scripts" / "NEAR").glob("*.sh"):
        text = sh.read_text()
        for path in re.findall(r"python (src/\S+\.py)", text):
            assert (ROOT / path).exists(), f"{sh.name}: {path} does not exist"
        for exp in re.findall(r"experiment=(generate/\S+)", text):
            assert (ROOT / "configs" / "experiment" / f"{exp}.yaml").exists(), f"{sh.name}: experiment {exp} does not exist"
