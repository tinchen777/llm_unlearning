"""Neighbour redirection: the defence of "Forgetting Without Telling".

Refusal-based unlearning pushes every forgotten entity into one special region
("I cannot answer that"), which sits outside the distribution of normal entities
and therefore *announces* which entities were deleted. This trainer instead
pushes each forgotten entity onto the activations of *its own* attribute-matched
stranger, i.e. into the region any unknown entity naturally occupies.

Objective (per selected layer l):

    L = gamma * L_redirect + alpha * L_retain + beta * L_contrast

    L_redirect : MSE (or cosine) between the forget activation at the selected
                 positions and the neighbour target.
    L_retain   : the project's usual retain term (NLL / KL), or EMBED_DIFF which
                 keeps retain activations where the reference model put them.
    L_contrast : optional InfoNCE with the neighbour as positive and the entity's
                 own pre-unlearning activation (plus, optionally, other entities'
                 targets) as negatives. `contrastive_weight=0` disables it, which
                 is the ablation against LUNAR's claim that the contrastive term
                 is not needed.

Only the MLP down-projection of the selected layer(s) is updated, as in LUNAR;
inference is unchanged, so the defence costs nothing at serving time.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from trainer.unlearn.forget_retain.base import ForgetRetainTrainer
from utils.common import IGNORE_INDEX

from fwt.layers.activations import (
    down_proj_regex,
    forward_with_activations,
    get_layer_modules,
    layer_module_name,
    pool_activations,
    position_mask,
    unwrap_model,
)
from fwt.trainer.losses import info_nce, masked_activation_mse, masked_redirect_loss

logger = logging.getLogger(__name__)


class NeighborRedirect(ForgetRetainTrainer):
    """Redirect forget activations onto attribute-matched neighbour activations.

    Args:
        layers: layer indices whose activations are redirected (negative indices
            count from the end). A single layer reproduces LUNAR's setup; several
            layers is the multi-layer ablation.
        trainable_params_regex: parameters the optimizer may update. Defaults to
            the `mlp.down_proj.weight` of `layers`.
        position_strategy: which token positions carry the redirection
            (see `fwt.layers.activations`).
        target_mode: `mean` averages the M neighbours into one target (stable),
            `sample` draws a different neighbour at each step (diverse
            displacement directions, harder to ablate).
        target_source: `precompute` caches the neighbour activations once with
            the reference model (default, cheapest); `online` recomputes them
            every step and requires `return_neighbors=True` on the dataset.
        redirect_loss_type: `mse` or `cosine`.
        contrastive_weight: weight of the InfoNCE term (0 disables it).
        contrastive_negatives: any of `self_ref` (the entity's own
            pre-unlearning activation) and `in_batch` (other entities' targets).
        match_target_norm: rescale the target to the norm of the source
            activation, keeping the edit purely directional.
        target_batch_size: batch size used when precomputing the target cache.
    """

    requires_ref_model = True

    def __init__(
        self,
        layers: Sequence[int] = (7,),
        trainable_params_regex: Optional[Sequence[str]] = None,
        position_strategy: str = "last_prompt",
        target_mode: str = "mean",
        target_source: str = "precompute",
        redirect_loss_type: str = "mse",
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        contrastive_negatives: Sequence[str] = ("self_ref", "in_batch"),
        match_target_norm: bool = False,
        target_batch_size: int = 8,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if target_mode not in ("mean", "sample"):
            raise ValueError(f"target_mode must be mean|sample, got `{target_mode}`.")
        if target_source not in ("precompute", "online"):
            raise ValueError(f"target_source must be precompute|online, got `{target_source}`.")

        self.layers = [int(layer) for layer in layers]
        self.position_strategy = position_strategy
        self.target_mode = target_mode
        self.target_source = target_source
        self.redirect_loss_type = redirect_loss_type
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_negatives = tuple(contrastive_negatives)
        self.match_target_norm = bool(match_target_norm)
        self.target_batch_size = int(target_batch_size)

        self.model_modules = get_layer_modules(self.model, self.layers)
        self.ref_modules = get_layer_modules(self.ref_model, self.layers)
        self.layer_names = [layer_module_name(self.model, layer) for layer in self.layers]
        self.trainable_params_regex = list(
            trainable_params_regex or down_proj_regex(self.model, self.layers)
        )
        logger.info(
            "NeighborRedirect: layers=%s (%s) | targets=%s/%s | positions=%s | "
            "redirect=%s | contrastive=%.3g | trainable=%s",
            self.layers, ", ".join(self.layer_names), self.target_source, self.target_mode,
            self.position_strategy, self.redirect_loss_type, self.contrastive_weight,
            self.trainable_params_regex,
        )

        self._target_cache: Optional[Dict[int, torch.Tensor]] = None
        self._row_to_pos: Dict[int, int] = {}
        self._loss_log: Dict[str, float] = {}
        self._loss_steps: int = 0

    # ------------------------------------------------------------- optimizer
    def create_optimizer(self, model=None):
        """Restrict the update to the selected projections (LUNAR / RMU style)."""
        self._set_requires_grad(False)
        self._set_trainable_params(True)
        trainable = [n for n, p in unwrap_model(self.model).named_parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                f"No parameter matched {self.trainable_params_regex}; nothing to train."
            )
        logger.info("Trainable parameters (%d): %s", len(trainable), trainable)
        optimizer = super().create_optimizer()
        # gradients must still flow through the frozen layers below the edit
        self._set_requires_grad(True)
        return optimizer

    def _set_requires_grad(self, requires_grad: bool) -> None:
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, requires_grad: bool) -> None:
        for name, param in unwrap_model(self.model).named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in self.trainable_params_regex):
                param.requires_grad = requires_grad

    # ------------------------------------------------------------------ loss
    def compute_forget_loss(self, model: Any, forget_inputs: Mapping[str, Any], **kwargs):
        source = self._source_inputs(forget_inputs)
        labels = source.get("labels")
        attention_mask = source.get("attention_mask")

        activations, outputs = forward_with_activations(
            model, dict(source), self.model_modules, grad=True, ignore_labels=True
        )
        mask = position_mask(labels, attention_mask, self.position_strategy)
        targets = self._get_targets(forget_inputs, source)

        redirect_loss = source["input_ids"].new_zeros((), dtype=torch.float32)
        contrast_loss = source["input_ids"].new_zeros((), dtype=torch.float32)
        ref_activations = self._reference_activations(source) if self._needs_self_ref() else None

        for position, layer in enumerate(self.layers):
            hidden = activations[position]
            target = self._prepare_target(targets[position], hidden, mask)
            layer_loss = masked_redirect_loss(
                hidden, target, mask, loss_type=self.redirect_loss_type
            )
            redirect_loss = redirect_loss + layer_loss.float()
            self._record(f"redirect/layer_{layer}", layer_loss)

            if self.contrastive_weight > 0:
                anchor = pool_activations(hidden, mask=mask)
                negatives = self._build_negatives(
                    target, ref_activations[position] if ref_activations else None, mask
                )
                layer_contrast = info_nce(
                    anchor, target, negatives, temperature=self.contrastive_temperature
                )
                contrast_loss = contrast_loss + layer_contrast.float()
                self._record(f"contrast/layer_{layer}", layer_contrast)

        redirect_loss = redirect_loss / len(self.layers)
        self._record("redirect", redirect_loss)
        loss = redirect_loss
        if self.contrastive_weight > 0:
            contrast_loss = contrast_loss / len(self.layers)
            self._record("contrast", contrast_loss)
            loss = loss + self.contrastive_weight * contrast_loss
        return loss, outputs

    def compute_retain_loss(self, model: Any, retain_inputs: Mapping[str, torch.Tensor], **kwargs):
        if self.retain_loss_type == "EMBED_DIFF":
            activations, _ = forward_with_activations(
                model, dict(retain_inputs), self.model_modules, grad=True
            )
            with torch.no_grad():
                reference, _ = forward_with_activations(
                    self.ref_model, dict(retain_inputs), self.ref_modules, grad=False
                )
            labels = retain_inputs.get("labels")
            mask = (
                (labels != IGNORE_INDEX)
                if labels is not None
                else retain_inputs["attention_mask"].bool()
            )
            loss = sum(
                masked_activation_mse(activations[p], reference[p], mask)
                for p in range(len(self.layers))
            ) / len(self.layers)
            self._record("retain", loss)
            return loss
        loss = super().compute_retain_loss(model, retain_inputs)
        self._record("retain", loss)
        return loss

    # --------------------------------------------------------------- targets
    def _get_targets(
        self,
        forget_inputs: Mapping[str, Any],
        source: Mapping[str, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """`{layer position: [b, d]}` neighbour targets for this batch."""
        if self.target_source == "online":
            stacked = self._online_targets(forget_inputs)
        else:
            stacked = self._cached_targets(source)
        return {
            position: self._reduce_neighbors(value) for position, value in stacked.items()
        }

    def _reduce_neighbors(self, per_neighbor: torch.Tensor) -> torch.Tensor:
        """`[b, M, d]` -> `[b, d]` by averaging or by sampling one neighbour."""
        if self.target_mode == "mean":
            return per_neighbor.mean(dim=1)
        choice = torch.randint(
            per_neighbor.shape[1], (per_neighbor.shape[0],), device=per_neighbor.device
        )
        return per_neighbor[torch.arange(per_neighbor.shape[0], device=per_neighbor.device), choice]

    def _cached_targets(self, source: Mapping[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        self._ensure_target_cache()
        indices = source.get("index")
        if indices is None:
            raise ValueError(
                "target_source=precompute needs the row index in the batch. Use the "
                "`DataCollatorForSupervisedDatasetwithIndex` collator "
                "(collator=DataCollatorForSupervisedDatasetwithIndex)."
            )
        rows = [int(i) for i in (indices.tolist() if torch.is_tensor(indices) else indices)]
        missing = [r for r in rows if r not in self._row_to_pos]
        if missing:
            raise KeyError(
                f"Rows {missing[:5]} are not in the neighbour target cache; the bank does "
                "not cover this forget split."
            )
        positions = torch.tensor([self._row_to_pos[r] for r in rows], dtype=torch.long)
        device = source["input_ids"].device
        assert self._target_cache is not None
        return {
            position: cache.index_select(0, positions).to(device)
            for position, cache in self._target_cache.items()
        }

    def _online_targets(self, forget_inputs: Mapping[str, Any]) -> Dict[int, torch.Tensor]:
        neighbors = forget_inputs.get("neighbors")
        if not neighbors:
            raise ValueError(
                "target_source=online needs the neighbour samples in the batch. Set "
                "`data.forget.<name>.args.return_neighbors=true` on the dataset."
            )
        per_layer: Dict[int, List[torch.Tensor]] = {p: [] for p in range(len(self.layers))}
        for key in sorted(neighbors, key=lambda k: int(k)):
            batch = neighbors[key]
            with torch.no_grad():
                activations, _ = forward_with_activations(
                    self.ref_model, dict(batch), self.ref_modules, grad=False
                )
            mask = position_mask(
                batch.get("labels"), batch.get("attention_mask"),
                self.position_strategy, is_target=True,
            )
            for position in per_layer:
                per_layer[position].append(pool_activations(activations[position], mask=mask))
        return {position: torch.stack(items, dim=1) for position, items in per_layer.items()}

    def _ensure_target_cache(self) -> None:
        """Forward every neighbour question once through the reference model."""
        if self._target_cache is not None:
            return
        dataset = self._forget_dataset()
        collate = self.data_collator
        if collate is None:
            raise ValueError("A data collator is required to precompute neighbour targets.")

        device = self.accelerator.device
        num_rows, num_neighbors = len(dataset), dataset.num_neighbors
        logger.info(
            "Precomputing neighbour targets: %d rows x %d neighbours at layers %s ...",
            num_rows, num_neighbors, self.layers,
        )

        cache: Dict[int, List[torch.Tensor]] = {p: [] for p in range(len(self.layers))}
        flat: List[Dict[str, Any]] = []
        for row in range(num_rows):
            flat.extend(dataset.neighbor_items(row))

        was_training = self.ref_model.training
        self.ref_model.eval()
        try:
            for start in range(0, len(flat), self.target_batch_size):
                chunk = [dict(item) for item in flat[start : start + self.target_batch_size]]
                batch = collate(chunk)
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                }
                with torch.no_grad():
                    activations, _ = forward_with_activations(
                        self.ref_model, batch, self.ref_modules, grad=False
                    )
                mask = position_mask(
                    batch.get("labels"), batch.get("attention_mask"),
                    self.position_strategy, is_target=True,
                )
                for position in cache:
                    pooled = pool_activations(activations[position], mask=mask)
                    cache[position].append(pooled.float().cpu())
        finally:
            if was_training:
                self.ref_model.train()

        self._target_cache = {
            position: torch.cat(items, dim=0).view(num_rows, num_neighbors, -1)
            for position, items in cache.items()
        }
        self._row_to_pos = {row: row for row in range(num_rows)}
        logger.info(
            "Neighbour target cache ready: %s per layer.",
            tuple(self._target_cache[0].shape),
        )

    def _forget_dataset(self) -> Any:
        """The `NeighborRedirectDataset` inside the train dataset."""
        dataset = self.train_dataset
        candidate = getattr(dataset, "forget", dataset)
        if not hasattr(candidate, "neighbor_items"):
            raise TypeError(
                "NeighborRedirect needs a `NeighborRedirectDataset` as the forget dataset, "
                f"got {type(candidate).__name__}. Use "
                "`data/datasets@data.forget=TOFU_QA_forget_neighbor`."
            )
        return candidate

    # ----------------------------------------------------------- contrastive
    def _needs_self_ref(self) -> bool:
        return self.contrastive_weight > 0 and "self_ref" in self.contrastive_negatives

    def _reference_activations(self, source: Mapping[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        with torch.no_grad():
            activations, _ = forward_with_activations(
                self.ref_model, dict(source), self.ref_modules, grad=False, ignore_labels=True
            )
        return activations

    def _build_negatives(
        self,
        target: torch.Tensor,
        ref_hidden: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """`[b, k, d]` negatives: the entity's own representation and/or peers."""
        negatives: List[torch.Tensor] = []
        if ref_hidden is not None:
            negatives.append(pool_activations(ref_hidden, mask=mask).unsqueeze(1))
        if "in_batch" in self.contrastive_negatives and len(target) > 1:
            # every other sample's target, shifted so no row sees its own positive
            rolled = torch.stack(
                [torch.roll(target, shifts=shift, dims=0) for shift in range(1, len(target))],
                dim=1,
            )
            negatives.append(rolled)
        if not negatives:
            raise ValueError(
                "contrastive_weight > 0 but no usable negatives; set contrastive_negatives "
                "to include `self_ref` and/or `in_batch` (with batch size > 1)."
            )
        return torch.cat([n.to(target.device) for n in negatives], dim=1)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _source_inputs(forget_inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """The forget sample itself, whether or not neighbours travel with it."""
        if "input_ids" in forget_inputs:
            return forget_inputs  # type: ignore[return-value]
        if "original" in forget_inputs:
            return forget_inputs["original"]
        raise ValueError(
            f"Unexpected forget batch keys {list(forget_inputs)}; expected `input_ids` or "
            "`original`."
        )

    def _prepare_target(
        self,
        target: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = target.to(dtype=hidden.dtype, device=hidden.device)
        if not self.match_target_norm:
            return target
        with torch.no_grad():
            source_norm = pool_activations(hidden, mask=mask).norm(dim=-1, keepdim=True)
        scale = source_norm / target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return target * scale

    def _record(self, name: str, value: torch.Tensor) -> None:
        self._loss_log[name] = self._loss_log.get(name, 0.0) + float(value.detach())
        if name == "redirect":
            self._loss_steps += 1

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Add the averaged loss components to the training log."""
        if self._loss_steps:
            for name, total in self._loss_log.items():
                logs[f"fwt_{name}"] = round(total / self._loss_steps, 6)
            self._loss_log.clear()
            self._loss_steps = 0
        return super().log(logs, *args, **kwargs)
