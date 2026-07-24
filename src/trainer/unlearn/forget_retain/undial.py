
from __future__ import annotations
from torch.nn import functional as F

from utils.common import forward_batch, IGNORE_INDEX
from .base import ForgetRetainTrainer


class UNDIAL(ForgetRetainTrainer):
    def __init__(self, beta: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_forget_loss(self, model, forget_inputs, **kwargs):
        # Forward pass on the student (trainable) model
        _student_logits, student_outputs = forward_batch(model, forget_inputs, ignore_labels=True)
        shift_student_logits = _student_logits[..., :-1, :]  # shape as [batch_size, seq_len-1, vocab_size]
        # Forward pass on the teacher model (no grad)
        _teacher_logits, _ = forward_batch(self.ref_model, forget_inputs, ignore_labels=True, grad=False)
        shift_teacher_logits = _teacher_logits[..., :-1, :]  # shape as [batch_size, seq_len-1, vocab_size]

        shift_labels = forget_inputs["labels"][..., 1:].to(shift_teacher_logits.device)  # shape as [batch_size, seq_len-1]
        next_idx = shift_labels.clamp(min=0).unsqueeze(-1)  # shape as [batch_size, seq_len-1, 1]

        shift_teacher_logits.scatter_add_(
            -1, next_idx, shift_teacher_logits.new_full(next_idx.shape, -self.beta)
        )
        soft_label = shift_teacher_logits.softmax(dim=-1)  # shape as [batch_size, seq_len-1, vocab_size]

        losses = F.cross_entropy(
            shift_student_logits.transpose(-1, -2),
            soft_label.transpose(-1, -2),
            reduction="none",
        )

        loss = losses[shift_labels != IGNORE_INDEX].mean()
        return loss, student_outputs
