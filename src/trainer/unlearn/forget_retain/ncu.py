"""NCU: Neighborhood-Contrastive Unlearning.

Idea
----
For every forget sample, locate its "nearest legal concept" -- the closest
retain sample(s) in the reference model's representation space (e.g. to forget
the plots of *Harry Potter*, the nearest legal neighbors are facts about the
author / the fantasy genre that must NOT be forgotten). Unlearning is then
driven by two contrastive objectives on pooled hidden representations plus a
standard retain LM loss:

1. Forget-pull (InfoNCE): the current model's representation of a forget
   sample is pulled toward the (frozen) reference representation of its
   nearest legal neighbors ("concept prototype"), and pushed away from its own
   original reference representation (and other forget samples in the batch).
   The model thus loses the ability to distinguish the forget concept from its
   legal neighborhood, instead of collapsing to noise as in RMU.

2. Neighborhood-anchor (InfoNCE or MSE): the current model's representation of
   a retain sample is anchored to its own reference representation and pushed
   away from the in-batch forget representations, which stabilizes the legal
   neighborhood that the forget samples are being pulled onto.

3. Retain LM loss (NLL or KL) preserves generation quality on retain data.

    loss = gamma * forget_pull + beta * neighborhood_anchor + alpha * retain_lm

Related work: RMU (steering forget activations to a random vector),
"On Effects of Steering Latent Representation for LLM Unlearning" (AAAI'25),
"Contrastive Unlearning" (arXiv:2401.10458) and CLReg (arXiv:2601.22028).
"""

from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, Mapping, Optional, Tuple

from .base import ForgetRetainTrainer
from utils.common import IGNORE_INDEX

logger = logging.getLogger(__name__)


class NCU(ForgetRetainTrainer):
    # Frozen reference model provides the fixed representation targets
    requires_ref_model = True

    def __init__(
        self,
        beta: float = 1.0,           # weight of the neighborhood-anchor loss
        tau: float = 0.1,            # InfoNCE temperature
        layer_id: int = 7,           # decoder layer whose output is used as representation (-1 = last)
        num_neighbors: int = 5,      # k nearest retain samples forming the legal-concept prototype
        anchor_loss_type: str = "InfoNCE",  # "InfoNCE" | "MSE"
        in_batch_negatives: bool = True,    # use other in-batch forget refs as extra negatives
        precompute_batch_size: Optional[int] = None,  # defaults to args.per_device_eval_batch_size
        *args,
        **kwargs,
    ):
        # gamma (forget-pull weight), alpha (retain LM loss weight) and
        # retain_loss_type come from ForgetRetainTrainer
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.tau = tau
        self.layer_id = layer_id
        self.num_neighbors = num_neighbors
        if anchor_loss_type not in ("InfoNCE", "MSE"):
            raise ValueError(
                f"anchor_loss_type must be 'InfoNCE' or 'MSE', got {anchor_loss_type}"
            )
        self.anchor_loss_type = anchor_loss_type
        self.in_batch_negatives = in_batch_negatives
        self.precompute_batch_size = precompute_batch_size

        # Representation banks are built lazily on the first training step,
        # after the accelerator has fully prepared everything.
        self.forget_bank: Optional[torch.Tensor] = None  # [Nf, d] raw pooled reps
        self.retain_bank: Optional[torch.Tensor] = None  # [Nr, d] raw pooled reps
        self.nn_idx: Optional[torch.Tensor] = None       # [Nf, k] retain indices

    # ------------------------------------------------------------------ #
    # Representation helpers
    # ------------------------------------------------------------------ #
    def _pooled_reps(
        self, model: Any, inputs: Mapping[str, torch.Tensor], grad: bool
    ) -> Tuple[torch.Tensor, Any]:
        """Mean-pool the chosen layer's hidden states over answer tokens.

        Returns (reps [bsz, d], model outputs). Positions are selected with the
        labels mask (labels != IGNORE_INDEX), consistent with RMU.
        """
        with torch.set_grad_enabled(grad):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                output_hidden_states=True,
            )
        # hidden_states[0] is the embedding output; decoder layer i outputs hidden_states[i + 1]
        hs_idx = self.layer_id + 1 if self.layer_id >= 0 else self.layer_id
        hidden = outputs.hidden_states[hs_idx]  # [bsz, seq, d]
        mask = (inputs["labels"] != IGNORE_INDEX).unsqueeze(-1).to(hidden.dtype)
        counts = mask.sum(dim=1).clamp(min=1)
        reps = (hidden * mask).sum(dim=1) / counts  # [bsz, d]
        return reps, outputs

    @staticmethod
    def _batch_indices(batch: Mapping[str, Any]) -> torch.Tensor:
        idxs = batch.get("index", None)
        if idxs is None:
            raise KeyError(
                "NCU requires the dataset `index` to be present in each batch. "
                "Make sure the dataset adds an `index` field (see BaseDataset.process_sample)."
            )
        if not isinstance(idxs, torch.Tensor):
            idxs = torch.as_tensor(idxs)
        return idxs.long().cpu()

    @torch.no_grad()
    def _build_rep_bank(self, dataset: Any, desc: str) -> torch.Tensor:
        """Forward the whole dataset through the frozen reference model and
        collect pooled representations indexed by the dataset's `index` field."""
        batch_size = (
            self.precompute_batch_size or self.args.per_device_eval_batch_size
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        device = self.accelerator.device
        reps_by_index: Dict[int, torch.Tensor] = {}
        for batch in tqdm(loader, desc=f"NCU: caching reps [{desc}]", unit="batch"):
            if "input_ids" not in batch:
                raise ValueError(
                    "NCU expects flat (single-answer) forget/retain datasets such as "
                    f"QADataset, but got a nested batch with keys {list(batch)}."
                )
            idxs = self._batch_indices(batch)
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }
            reps, _ = self._pooled_reps(self.ref_model, batch, grad=False)
            for i, idx in enumerate(idxs.tolist()):
                reps_by_index[idx] = reps[i].float()

        bank = torch.zeros(
            max(reps_by_index) + 1, next(iter(reps_by_index.values())).shape[-1],
            dtype=torch.float32, device=device
        )
        for idx, rep in reps_by_index.items():
            bank[idx] = rep
        return bank

    def _prepare_banks(self):
        train_dataset = self.train_dataset
        forget_data = getattr(train_dataset, "forget", None)
        retain_data = getattr(train_dataset, "retain", None)
        if forget_data is None or retain_data is None:
            raise ValueError(
                "NCU requires a ForgetRetainDataset train dataset exposing "
                "`forget` and `retain` attributes."
            )
        self.forget_bank = self._build_rep_bank(forget_data, desc="forget")
        self.retain_bank = self._build_rep_bank(retain_data, desc="retain")

        # Nearest legal neighbors: cosine top-k of each forget rep over retain reps
        sim = F.normalize(self.forget_bank, dim=-1) @ F.normalize(self.retain_bank, dim=-1).T
        k = min(self.num_neighbors, sim.shape[1])
        topk = sim.topk(k, dim=-1)
        self.nn_idx = topk.indices  # [Nf, k]
        logger.info(
            f"NCU: built representation banks (forget: {tuple(self.forget_bank.shape)}, "
            f"retain: {tuple(self.retain_bank.shape)}), "
            f"k={k} nearest-neighbor sim mean={topk.values.mean().item():.4f}, "
            f"max={topk.values.max().item():.4f}"
        )

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #
    def _info_nce(
        self,
        anchor: torch.Tensor,     # [b, d] current model reps (grad)
        positive: torch.Tensor,   # [b, d] target reps
        negatives: torch.Tensor,  # [b, n, d] negative reps
    ) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        pos_logit = (anchor * positive).sum(dim=-1, keepdim=True)          # [b, 1]
        neg_logits = torch.einsum("bd,bnd->bn", anchor, negatives)          # [b, n]
        logits = torch.cat([pos_logit, neg_logits], dim=-1) / self.tau
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_forget_pull_loss(
        self, forget_reps: torch.Tensor, forget_idx: torch.Tensor
    ) -> torch.Tensor:
        """Pull forget reps toward their nearest-legal-concept prototype and
        away from their own (frozen) original reps."""
        dtype = forget_reps.dtype
        # positive: centroid of the k nearest legal (retain) neighbor reps
        positive = self.retain_bank[self.nn_idx[forget_idx]].mean(dim=1).to(dtype)  # [b, d]
        # negatives: the sample's own original rep (+ optionally other in-batch forget refs)
        own_ref = self.forget_bank[forget_idx].to(dtype)  # [b, d]
        if self.in_batch_negatives and forget_reps.shape[0] > 1:
            b = forget_reps.shape[0]
            negatives = own_ref.unsqueeze(0).expand(b, b, -1)  # [b, b, d]; row i's negs = all refs
        else:
            negatives = own_ref.unsqueeze(1)  # [b, 1, d]
        return self._info_nce(forget_reps, positive, negatives)

    def compute_anchor_loss(
        self,
        retain_reps: torch.Tensor,
        retain_idx: torch.Tensor,
        forget_reps: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor retain reps to their original reference reps, pushing them
        away from the in-batch (moving) forget reps."""
        dtype = retain_reps.dtype
        own_ref = self.retain_bank[retain_idx].to(dtype)  # [b, d]
        if self.anchor_loss_type == "MSE":
            return F.mse_loss(retain_reps, own_ref)
        b = retain_reps.shape[0]
        negatives = forget_reps.detach().unsqueeze(0).expand(b, forget_reps.shape[0], -1)
        return self._info_nce(retain_reps, own_ref, negatives)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.nn_idx is None:
            self._prepare_banks()

        forget_batch = inputs["forget"]
        retain_batch = inputs["retain"]
        forget_idx = self._batch_indices(forget_batch)
        retain_idx = self._batch_indices(retain_batch)

        forget_inputs = {
            "input_ids": forget_batch["input_ids"],
            "attention_mask": forget_batch["attention_mask"],
            "labels": forget_batch["labels"],
        }
        retain_inputs = {
            "input_ids": retain_batch["input_ids"],
            "attention_mask": retain_batch["attention_mask"],
            "labels": retain_batch["labels"],
        }

        # single forward pass per split provides both reps and LM loss
        forget_reps, forget_outputs = self._pooled_reps(model, forget_inputs, grad=True)
        retain_reps, retain_outputs = self._pooled_reps(model, retain_inputs, grad=True)

        forget_pull_loss = self.compute_forget_pull_loss(forget_reps, forget_idx)
        anchor_loss = self.compute_anchor_loss(retain_reps, retain_idx, forget_reps)

        if self.retain_loss_type == "NLL":
            retain_lm_loss = retain_outputs.loss
        else:  # e.g. "KL" -> reuse GradDiff's implementation (extra forward pass)
            retain_lm_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (
            self.gamma * forget_pull_loss
            + self.beta * anchor_loss
            + self.alpha * retain_lm_loss
        )

        self.log({
            "forget_pull_loss": forget_pull_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "retain_lm_loss": retain_lm_loss.item(),
        })

        return (loss, forget_outputs) if return_outputs else loss
