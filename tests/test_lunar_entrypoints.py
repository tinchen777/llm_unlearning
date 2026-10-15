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
         "experiment=generate/LUNAR", "model=tofu/tofu_Llama-3.2-1B-Instruct_full", "task_name=t",
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
        for exp in re.findall(r"experiment=(generate/\S+)", text):
            assert (ROOT / "configs" / "experiment" / f"{exp}.yaml").exists(), f"{sh.name}: experiment {exp} does not exist"
