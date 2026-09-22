"""Forget dataset that also carries the neighbour questions of each row.

`NeighborRedirectDataset` extends the project's `QADataset`, so it plugs into the
existing `data/unlearn.yaml` composition (`ForgetRetainDataset` wraps it exactly
like any other forget dataset) and is tokenised with the same chat template.

The neighbour side is tokenised as a *prompt with an empty answer*: the sample
then looks exactly like the forget sample up to the assistant header, so
`last_prompt` selects the same relative position on both sides, and no extra
key is needed in the batch (the collator stays untouched).

Two ways of reaching the targets:

* `return_neighbors=False` (default) - the batch holds only the forget sample and
  the trainer looks the target up in a cache it precomputes once with the
  reference model (`target_source=precompute`). Cheapest, and the default.
* `return_neighbors=True` - every batch carries the M neighbour samples as
  `{"original": ..., "neighbors": {"0": ..., ...}}` and the trainer forwards them
  through the reference model at every step (`target_source=online`).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from data.datasets.qa import QADataset

from fwt.neighbors.schema import NeighborBank

logger = logging.getLogger(__name__)


class NeighborRedirectDataset(QADataset):
    """QA forget dataset augmented with attribute-matched neighbour questions.

    Args:
        neighbors_path: `neighbors.json` produced by `fwt/scripts/build_neighbors.py`.
        num_neighbors: use only the first M neighbours of the bank (default: all).
        return_neighbors: include the tokenised neighbour prompts in each item.
        neighbor_answer: text used as the (dummy) answer of a neighbour prompt;
            keep `""` so that no answer token enters the target position.
        **kwargs: forwarded to `QADataset` (hf_args, tokenizer, template_args, ...).
    """

    def __init__(
        self,
        neighbors_path: str,
        num_neighbors: Optional[int] = None,
        return_neighbors: bool = False,
        neighbor_answer: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.bank = NeighborBank.load(neighbors_path)
        self.bank.validate(expect_m=num_neighbors)
        self.neighbors_path = str(neighbors_path)
        self.num_neighbors = num_neighbors or self.bank.num_neighbors
        self.return_neighbors = return_neighbors
        self.neighbor_answer = neighbor_answer
        self._checked_alignment = False

    # ------------------------------------------------------------------ api
    def __getitem__(self, idx: int):
        self._check_alignment()
        original = super().__getitem__(idx)
        if not self.return_neighbors:
            return original
        neighbors = self.neighbor_items(idx)
        return {
            "original": original,
            "neighbors": {str(j): item for j, item in enumerate(neighbors)},
        }

    def neighbor_items(self, idx: int) -> List[Dict[str, Any]]:
        """Tokenised neighbour prompts for row `idx` (list of length M)."""
        questions = self.bank.neighbor_questions(idx, num=self.num_neighbors)
        items = []
        for question in questions:
            sample = self.tok_fn(question, self.neighbor_answer, idx, **self.tok_kwargs)
            items.append(self.process_sample(sample))
        return items

    def neighbor_questions(self, idx: int) -> List[str]:
        return self.bank.neighbor_questions(idx, num=self.num_neighbors)

    @property
    def row_indices(self) -> List[int]:
        """Row indices covered by the bank, in dataset order."""
        return list(range(len(self)))

    # ------------------------------------------------------------- internals
    def _check_alignment(self) -> None:
        """Fail loudly when the bank was built for a different split."""
        if self._checked_alignment:
            return
        self._checked_alignment = True
        covered = set(self.bank.covered_indices())
        missing = [i for i in range(len(self)) if i not in covered]
        if missing:
            raise ValueError(
                f"Neighbour bank `{self.neighbors_path}` covers {len(covered)} rows but the "
                f"dataset has {len(self)}; rows {missing[:5]}... have no neighbours. "
                "The bank must be built from the same forget split."
            )
        first = self.bank.question(0).question
        raw_first = str(self.raw_data[0][self.question_key])
        if first.strip() != raw_first.strip():
            logger.warning(
                "Row 0 of the neighbour bank (%r) differs from row 0 of the dataset (%r). "
                "Check that the bank was built from this exact split.",
                first[:60], raw_first[:60],
            )
