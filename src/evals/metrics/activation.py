from __future__ import annotations
import logging
import torch
from typing import Any, Optional, Sequence, TYPE_CHECKING

from data.datasets.datasets_utils import get_eoi_positions
from model.hooks.collector import ActivationCollector
from .base import MetricFunc
from ..activation.analysis import (
    activation_scalars, analyze_activations, check_r_uv_identity, save_activation_outputs,
)

if TYPE_CHECKING:
    from pathlib import Path
    from data import SplitData

logger = logging.getLogger("eval.metric")


@MetricFunc
def activation_probe(
    model: Any,
    tokenizer: Any,
    template_args: Any,
    split_data: SplitData,
    output_dir: Optional[Path],
    forget: str,
    reference: str,
    seen: Sequence[str],
    unseen: Sequence[str],
    point: str,
    positions: Optional[Sequence[int]],
    layers: Optional[Sequence[int]],
    qa_per_author: int,
    author_offset: int,
    forget_split: str,
    r_uv_path: Optional[str],
    focus_layer: Optional[int],
    seed: int,
    **kwargs
):
    """Per-layer activations of the model at the end-of-instruction tokens on the splits of `data`: dataset means, the global
    r_UV (`reference` - `forget`), norms / cosines and the linear separability (AUC) of the pairs of splits.

    - `r_uv_path`: positions / layers / point / author numbering are read from that `r_uv.pt` (the probe of an edited model,
      comparable with the run that wrote it) and its forget split / rows are checked; `null`: those of the config.
    - no `agg_value`: `summary` holds the scalars of `activation_scalars` for `expose`; `value_by_layer` the whole table
      (also written, with `activations.pt`, to `output_dir`).
    """
    if r_uv_path is not None:
        ref = torch.load(str(r_uv_path), weights_only=False)
        check_r_uv_identity(
            ref, forget_split=forget_split, num_forget=len(split_data[forget].dataset), source=str(r_uv_path)
        )
        from_file = {
            "positions": list(ref["positions"]), "layers": list(ref["layers"]), "point": ref["point"],
            "qa_per_author": int(ref["qa_per_author"]), "author_offset": int(ref["author_offset"]),
        }
        given = {
            "positions": None if positions is None else list(positions),
            "layers": None if layers is None else list(layers),
            "point": point, "qa_per_author": qa_per_author, "author_offset": author_offset,
        }
        for key, value in given.items():
            if value is not None and value != from_file[key]:
                logger.warning(f"`{key}`={value} of the config is ignored: {r_uv_path} was computed with {from_file[key]}.")
        positions, layers, point = from_file["positions"], from_file["layers"], from_file["point"]
        qa_per_author, author_offset = from_file["qa_per_author"], from_file["author_offset"]

    act_collector = ActivationCollector(
        model,
        positions=list(positions) if positions is not None else get_eoi_positions(tokenizer, template_args),
        layers=layers,
        point=point,
    )
    logger.info(f"Hook point: `{act_collector.point}`, positions: {act_collector.positions}")
    result = analyze_activations(
        act_collector, split_data, forget, reference, seen=seen, unseen=unseen,
        qa_per_author=int(qa_per_author), author_offset=int(author_offset), seed=int(seed),
    )
    if output_dir is not None:
        save_activation_outputs(result, act_collector, output_dir)
    return {
        "agg_value": None,
        "summary": activation_scalars(result["per_layer"], focus_layer),
        "value_by_layer": result["per_layer"],
    }
