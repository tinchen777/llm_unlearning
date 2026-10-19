"""Folder layout and file names of the LUNAR outputs, shared by select_layer.py (writes), w_down.py (reads) and 4_eval.sh.

Below `output_dir` (= saves/LUNAR/<task_name>/):
    r_uv.pt, r_uv_summary.json                  shared by every setting (r_uv.py)
    c<coeff>_<shift_mode>/                      one folder per setting (`config_dir`)
        layer_selection.json, responses_steer_l<layer>.json          (select_layer.py)
        l<layers>[_cf]/                          one folder per W_down run (`run_dir`): layers, closed form or AdamW
            w_down.pt, w_down_summary.json, responses_before_*, responses_after_*, model/        (w_down.py)
scripts/LUNAR/_common.sh mirrors these names (`config_dir`, `run_dir`).
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

SELECTION_FILE = "layer_selection.json"


def config_dir(output_dir: Path, coeff: float, shift_mode: str) -> Path:
    """Folder of one setting (coeff, shift mode): the selected layer depends on both, so W_down training (w_down.py)
    must read the selection of the SAME setting."""
    return Path(output_dir) / f"c{coeff:g}_{shift_mode}"


def run_dir(output_dir: Path, coeff: float, shift_mode: str, layers: Sequence[int], solver: str) -> Path:
    """Folder of one W_down run inside the setting folder: the trained layers and the solver (`_cf` = closed form)."""
    return config_dir(output_dir, coeff, shift_mode) / (
        "l" + "-".join(str(l) for l in layers) + ("_cf" if solver == "closed_form" else "")
    )


def selection_path(output_dir: Path, coeff: float, shift_mode: str) -> Path:
    """Layer-selection file of a setting, written by select_layer.py and read by w_down.py."""
    return config_dir(output_dir, coeff, shift_mode) / SELECTION_FILE
