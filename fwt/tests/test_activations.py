"""Unit tests for position selection and pooling (no model required)."""

from __future__ import annotations
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fwt.layers.activations import IGNORE_INDEX, pool_activations, position_mask  # noqa: E402

IGN = IGNORE_INDEX
LABELS = torch.tensor([[IGN, IGN, 5, 6], [IGN, IGN, IGN, 7]])
ATTENTION = torch.ones_like(LABELS)


def test_last_prompt():
    mask = position_mask(LABELS, ATTENTION, "last_prompt")
    assert mask.tolist() == [[False, True, False, False], [False, False, True, False]]


def test_answer():
    mask = position_mask(LABELS, ATTENTION, "answer")
    assert mask.tolist() == [[False, False, True, True], [False, False, False, True]]


def test_answer_on_target_side_falls_back_to_last_prompt():
    mask = position_mask(LABELS, ATTENTION, "answer", is_target=True)
    assert mask.tolist() == position_mask(LABELS, ATTENTION, "last_prompt").tolist()


def test_prompt_mean_and_last():
    assert position_mask(LABELS, ATTENTION, "prompt_mean").tolist() == [
        [True, True, False, False], [True, True, True, False]
    ]
    assert position_mask(LABELS, ATTENTION, "last").tolist() == [
        [False, False, False, True], [False, False, False, True]
    ]


def test_padding_is_excluded():
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    labels = torch.tensor([[IGN, IGN, 5, IGN], [IGN, IGN, IGN, 7]])
    assert position_mask(labels, attention, "last").tolist() == [
        [False, False, True, False], [False, False, False, True]
    ]


def test_prompt_only_batch_without_labels():
    attention = torch.tensor([[1, 1, 1, 0]])
    assert position_mask(None, attention, "last_prompt").tolist() == [[False, False, True, False]]
    assert position_mask(None, attention, "prompt_mean").tolist() == [[True, True, True, False]]


def test_row_without_answer_tokens_uses_last_prompt_token():
    labels = torch.tensor([[IGN, IGN, IGN, IGN]])
    attention = torch.tensor([[1, 1, 1, 0]])
    assert position_mask(labels, attention, "last_prompt").tolist() == [[False, False, True, False]]


def test_pooling_averages_selected_positions():
    hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    labels = torch.tensor([[IGN, IGN, 5, 6]])
    attention = torch.ones_like(labels)
    pooled = pool_activations(hidden, labels, attention, strategy="answer")
    assert torch.allclose(pooled, torch.tensor([[3.5, 3.5]]))
    pooled = pool_activations(hidden, labels, attention, strategy="last_prompt")
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all activation tests passed")
