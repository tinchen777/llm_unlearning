"""File naming shared by select_layer.py (writes) and w_down.py (reads) of LUNAR."""

from __future__ import annotations


def selection_file_name(coeff: float, shift_mode: str) -> str:
    """Name of the layer-selection file: the selected layer depends on the coefficient and the shift mode, so W_down
    training (w_down.py) must read the file of the SAME ones."""
    return f"layer_selection_c{coeff:g}_{shift_mode}.json"
