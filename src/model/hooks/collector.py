
from __future__ import annotations
import torch
from tqdm import tqdm
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Iterator, Tuple, Callable, Hashable

from ..arch import eval_mode
from ..inference.utils import batch_to_device, get_model_device
from .utils import capture_activations, gather_positions


class ActivationCollector:
    """Collect `point` activations at `positions` of the decoder layers `layers`.

    One configuration, three ways to aggregate over a dataloader:
        - `mean`        : dataset mean                      -> Tensor[n_pos, n_layers, d] (float64)
        - `per_sample`  : one tensor per sample (no mean)   -> {dataset_index: Tensor[n_pos, n_layers, d]}
        - `group_means` : one mean per group (e.g. author)  -> {key: Tensor[n_pos, n_layers, d]} (float32)

    Args:
        positions: negative offsets from each row's last real token (default `(-1,)`).
        layers: decoder-layer indices (default: all layers).
        point: where to read the activation, see `POINTS`.
    """

    def __init__(
        self,
        model: Any,
        positions: Sequence[int] = (-1,),
        layers: Optional[Sequence[int]] = None,
        point: str = "block_out",
    ):
        self.model = model
        self.positions = list(positions)
        self.layers = list(range(model.config.num_hidden_layers)) if layers is None else list(layers)
        self.point = point

    @torch.no_grad()
    def _iter(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        desc: str,
        label: str,
        colour: str,
    ) -> Iterator[Tuple[Mapping[str, Any], torch.Tensor]]:
        """Yields `(batch, batch_acts)` with batch_acts `[bsz, n_pos, n_layers, d]` (model dtype, CPU); restores train/eval mode."""
        with eval_mode(self.model), capture_activations(self.model, self.layers, self.point) as cache:
            for batch in tqdm(
                dataloader,
                desc=f"Collecting `{self.point}` {label} [{desc}]",
                unit="batch(es)",
                colour=colour
            ):
                if "input_ids" not in batch:
                    raise ValueError(f"Expected a flat batch with `input_ids`, got keys {list(batch)}.")
                inputs = batch_to_device({k: batch[k] for k in ("input_ids", "attention_mask")}, get_model_device(self.model))
                self.model(**inputs)

                # [bsz, n_pos, n_layers, d]
                batch_acts = torch.stack(
                    [gather_positions(cache[_l], inputs["attention_mask"], self.positions).cpu() for _l in self.layers],
                    dim=2
                )
                yield batch, batch_acts

    # ---- aggregations ----

    def mean(self, dataloader: Iterable[Mapping[str, Any]], desc: str = "") -> torch.Tensor:
        """Dataset-mean activations, Tensor[n_pos, n_layers, d] (float64, CPU)."""
        total: Optional[torch.Tensor] = None
        n_samples = 0
        for _, batch_acts in self._iter(dataloader, desc, "activations", "green"):
            batch_sum = batch_acts.to(torch.float64).sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            n_samples += batch_acts.shape[0]
        if n_samples == 0 or total is None:
            raise ValueError("The dataloader yielded no samples.")
        return total / n_samples

    def per_sample(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        desc: str = "",
        meta_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, Any]]]:
        """One activation tensor PER SAMPLE, keyed by the dataset `index` of the batch.

        Returns:
            (activations, per_meta): `activations[idx]` is Tensor[n_pos, n_layers, d] (model dtype, CPU);
            `per_meta[idx]` holds the requested `meta_keys` of that sample.
        """
        per_sample: Dict[int, torch.Tensor] = {}
        per_meta: Dict[int, Dict[str, Any]] = {}
        for batch, batch_acts in self._iter(dataloader, desc, "activations", "green"):
            indices = batch.get("index")
            if indices is None:
                raise ValueError("Collecting sample-wise activations needs the dataset `index` in each batch.")
            for i, idx in enumerate(torch.as_tensor(indices).tolist()):
                per_sample[int(idx)] = batch_acts[i]
                if meta_keys:
                    per_meta[int(idx)] = {
                        k: (batch[k][i].item() if isinstance(batch[k], torch.Tensor) else batch[k][i])
                        for k in meta_keys if k in batch
                    }
        if not per_sample:
            raise ValueError("The dataloader yielded no samples.")
        return per_sample, per_meta

    def group_means(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        group_fn: Callable[[Mapping[str, Any]], Sequence[Hashable]],
        desc: str = "",
    ) -> Tuple[Dict[Hashable, torch.Tensor], Dict[Hashable, int]]:
        """One mean per GROUP. `group_fn(batch)` returns one hashable key per row of the batch.

        Only running sums are kept (no per-sample storage): memory is `n_groups * n_pos * n_layers * d` float32.

        Returns:
            (means, counts): `means[key]` Tensor[n_pos, n_layers, d] (float32, CPU), `counts[key]` samples in the group.
        """
        sums: Dict[Hashable, torch.Tensor] = {}
        counts: Dict[Hashable, int] = {}
        for batch, batch_acts in self._iter(dataloader, desc, "group means", "blue"):
            keys = list(group_fn(batch))
            if len(keys) != batch_acts.shape[0]:
                raise ValueError(f"group_fn returned {len(keys)} keys for a batch of {batch_acts.shape[0]} rows.")
            batch_acts = batch_acts.float()
            for i, key in enumerate(keys):
                sums[key] = sums[key] + batch_acts[i] if key in sums else batch_acts[i].clone()
                counts[key] = counts.get(key, 0) + 1
        if not sums:
            raise ValueError("The dataloader yielded no samples.")
        return {key: sums[key] / counts[key] for key in sums}, counts
