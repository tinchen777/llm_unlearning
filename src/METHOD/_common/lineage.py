"""Alignment between the stages  r_UV -> layer selection -> W_down  (and the probe of an edited model).

Every stage writes what the next stage needs into its output file and the next stage READS it from there; the config of
a later stage only holds a pointer to the earlier file (plus its own hyperparameters):

    r_uv.pt               positions, layers, point, ks, qa_per_author, author_offset, seed (how r_UV was computed)
                          + identity: forget_split, num_forget, authors, model (what it was computed on)
    layer_selection.json  selected_layer, k, coeff, shift_mode, ref_position, positions + r_uv_path + the identity
    w_down_summary.json   everything of the run + layer_selection_path / r_uv_path

A config value that is also stored upstream may be left `null` (taken from the file) or given; when it is given it must
agree with the file (`pick`). The identity is checked against the data / model the stage is about to use
(`check_r_uv_identity`), because the author of a forget row is computed from its row index: another forget split would
silently use the wrong r_UV.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def model_name(model_cfg: Mapping[str, Any]) -> str:
    """Identifier of the model of a config: `model.pretrained.name_or_path`."""
    return str(model_cfg["pretrained"]["name_or_path"])


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, Sequence) and not isinstance(a, str) and isinstance(b, Sequence) and not isinstance(b, str):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def pick(name: str, cfg_value: Any, upstream_value: Any, source: str = "the upstream file") -> Any:
    """The value of `name`: the one stored upstream; `cfg_value` (None = not set) may only repeat it.

    Without an upstream value (no such file given) the config value is required.
    """
    if upstream_value is None:
        if cfg_value is None:
            raise ValueError(f"`{name}` is neither stored in {source} nor given in the config.")
        return cfg_value
    if cfg_value is not None and not _same(cfg_value, upstream_value):
        raise ValueError(
            f"`{name}`={cfg_value!r} in the config disagrees with {upstream_value!r} stored in {source}. "
            f"Set it to null to take the stored value, or run the earlier stage again with the value you want."
        )
    return upstream_value


def check_r_uv_identity(
    saved: Mapping[str, Any],
    *,
    forget_split: Optional[str] = None,
    num_forget: Optional[int] = None,
    model: Optional[str] = None,
    source: str = "r_uv.pt",
) -> None:
    """The data / model of this stage must be the ones `r_uv.pt` was computed on (`None` skips that check)."""
    for key, current in (("forget_split", forget_split), ("num_forget", num_forget), ("model", model)):
        if current is None:
            continue
        if key not in saved:
            raise ValueError(f"{source} has no `{key}` (written by an older r_uv.py): run r_uv.py again.")
        if saved[key] != current:
            raise ValueError(f"{source} was computed with {key}={saved[key]!r}, but this run uses {key}={current!r}.")


def resolve_r_uv_path(recorded: Optional[str], selection_path: Optional[Path] = None) -> Path:
    """`r_uv.pt` recorded in a layer-selection file; falls back to the one next to the setting folder
    (`<output_dir>/<setting>/layer_selection.json` -> `<output_dir>/r_uv.pt`) when the recorded path no longer exists."""
    candidates = [Path(recorded)] if recorded else []
    if selection_path is not None:
        candidates.append(Path(selection_path).parent.parent / "r_uv.pt")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"r_uv.pt not found, tried: {[str(p) for p in candidates]}.")
