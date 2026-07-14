
from __future__ import annotations
import torch
from transformers import TrainerCallback
from typing import Optional

from utils.common import forward_batch, IGNORE_INDEX
from .base import ForgetRetainTrainer


class PDU(ForgetRetainTrainer):
    def __init__(
        self,
        retain_loss_eps: float = 0.0,
        primal_dual: bool = False,
        dual_step_size: float = 1.0,
        dual_update_upon: str = "step",
        dual_warmup_epochs: int = 0,
        loss_names: Optional[list[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.retain_loss_eps = retain_loss_eps
        self.primal_dual = primal_dual
        self.dual_step_size = dual_step_size
        self.dual_update_upon = dual_update_upon
        self.can_update = dual_warmup_epochs == 0

        self.loss_names = ["forget_loss", "retain_loss"] if loss_names is None else loss_names
        # 新增：累积 micro-batch 的对偶梯度（已减去 eps 的 shifted retain loss）
        self._retain_g_buffer = []

        if primal_dual:
            self.add_callback(
                DualOptimizationCallback(self, dual_update_upon, dual_warmup_epochs)
            )

    def enable_updates(self):
        self.can_update = True

    def _update_alpha_for_retain(self, retain_g: float):
        self.alpha = max(0.0, self.alpha + self.dual_step_size * retain_g)

    def _is_model_training(self):
        return self.model.training if self.model is not None else False

    @torch.no_grad()
    def post_epoch_dual_param_update(self):
        assert self.model is not None, "Model is not initialized."
        dataloader = self.get_train_dataloader()
        was_training = self._is_model_training()
        self.model.eval()

        total, n_samples = 0.0, 0
        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            retain = inputs["retain"]
            loss = self.compute_retain_loss(model=self.model, retain_inputs=retain)  # type: ignore
            bsz = retain["input_ids"].size(0)  # type: ignore
            total += loss * bsz
            n_samples += bsz

        retain_loss = total.item() / n_samples  # type: ignore
        retain_g = retain_loss - self.retain_loss_eps
        self._update_alpha_for_retain(retain_g)

        if was_training:
            self.model.train()

        self.log({
            "retain_preference": self.alpha,
            "epoch_retain_loss": retain_loss
        })

    @torch.no_grad()
    def step_dual_param_update(self):
        if not self._retain_g_buffer:
            return
        avg_g = torch.stack(self._retain_g_buffer).mean()
        avg_retain_g = self.accelerator.reduce(avg_g, reduction="mean").item()  # type: ignore
        self._retain_g_buffer.clear()

        if not self.can_update:
            return
        self._update_alpha_for_retain(avg_retain_g)  # λ ← max(0, λ + η·(L_retain − ε))

        self.log({"retain_preference": self.alpha})

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        logits, outputs = forward_batch(model, forget_inputs, ignore_labels=True)

        logits = logits.reshape(-1, logits.size(-1))  # shape as [batch_size * seq_len, vocab_size]
        maxLogits = logits.max(dim=-1)[0]  # shape as [batch_size * seq_len]
        averageLogits = logits.mean(dim=-1)

        loss = (maxLogits - averageLogits) ** 2
        mask = (forget_inputs["labels"] != IGNORE_INDEX).reshape(-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        return loss, outputs

    def aggregate_loss(self, forget_loss, retain_loss):
        # Shift the retain_loss for the primal dual method.
        # If no primal-dual method is used, gradient-based methods will not suffer
        # from unwanted shifts
        retain_g = retain_loss - self.retain_loss_eps  # g(θ) = L_retain − ε

        loss = super().aggregate_loss(forget_loss, retain_loss)

        # Update the dual parameter if primal-dual method is used, the update is done per step and the warm-up period is over
        if self.primal_dual and self._is_model_training() and self.dual_update_upon == "step":
            self._retain_g_buffer.append(retain_g.detach())

        # Log individual losses and the retain preference
        if self._is_model_training():
            log_dict = {
                self.loss_names[0]: forget_loss.item(),
                self.loss_names[1]: retain_loss.item(),
            }
            self.log(log_dict)

        return loss


class DualOptimizationCallback(TrainerCallback):
    def __init__(self, trainer: PDU, dual_update_upon: str, dual_warmup_epochs: int):
        self.trainer = trainer
        self.dual_update_upon = dual_update_upon
        self.dual_warmup_epochs = dual_warmup_epochs

    def on_step_end(self, args, state, control, **kwargs):
        # HF Trainer 保证：on_step_end 每个 optimizer step 触发一次
        # （梯度累积的中间 micro-batch 走的是 on_substep_end）
        if self.dual_update_upon == "step":
            self.trainer.step_dual_param_update()

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.dual_warmup_epochs:
            self.trainer.enable_updates()
            if self.dual_update_upon == "epoch":
                self.trainer.post_epoch_dual_param_update()
