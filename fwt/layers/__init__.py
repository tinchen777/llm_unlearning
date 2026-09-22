"""Activation access and layer selection (stage 2 of the FWT pipeline)."""

from __future__ import annotations

from .activations import (
    capture_activations,
    down_proj_regex,
    extract_activations,
    forward_with_activations,
    get_layer_modules,
    layer_module_name,
    num_layers,
    pool_activations,
    position_mask,
)
from .metrics import layer_diagnostics, probe_auc, relative_distance, target_cohesion
from .select import (
    DEFAULT_CRITERIA,
    LayerScore,
    analyze_layers,
    format_table,
    load_selection,
    save_selection,
    score_layers,
)

__all__ = [
    "capture_activations", "forward_with_activations", "extract_activations",
    "get_layer_modules", "layer_module_name", "num_layers", "down_proj_regex",
    "pool_activations", "position_mask",
    "layer_diagnostics", "probe_auc", "relative_distance", "target_cohesion",
    "LayerScore", "analyze_layers", "score_layers", "save_selection",
    "load_selection", "format_table", "DEFAULT_CRITERIA",
]
