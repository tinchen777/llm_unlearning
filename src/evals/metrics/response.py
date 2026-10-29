
from __future__ import annotations
import math
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

from .utils import generate_batch
from .ops import answer_logprobs

if TYPE_CHECKING:
    from data import Data


class ResponseCollector:
    def __init__(
        self,
        model: Any,
        split_data: Data,
        tokenizer: Any,
        generation_args: Any,
        hallucination_rouge: float = 0.3,
        
        
        
        
    
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_args = generation_args

        self.hallucination_rouge = float(hallucination_rouge)

    def generate(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        max_samples: Optional[int] = None,
        with_logprob: bool = True,
        desc: str = "",
        before_batch: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a response for every sample of `dataloader`.

        Args:
            max_samples: stop after this many samples (None -> the whole loader).
            before_batch: called with every batch (metadata included) before the model sees it, e.g. to set
                the per-row vectors of a `Steerer`.
        Returns:
            One record per sample: the batch metadata (`index`, `author_id`, ...) plus `input` (decoded prompt),
            `ground_truth`, `generation`, the ROUGE scores, `answer_logprob`, `refusal` and `degenerate`.
        """
        self.model.eval()

        records: List[Dict[str, Any]] = []
        for batch in tqdm(dataloader, desc=f"Generating [{desc}]", unit="batch(es)", colour="blue"):
            if max_samples is not None and len(records) >= max_samples:
                break
            if before_batch is not None:
                before_batch(batch)
            # metadata (index, author_id, ...) is not a tensor input of the model: set it aside
            meta = {k: _as_list(batch[k]) for k in batch if k not in MODEL_INPUT_KEYS}
            inputs = {k: batch[k] for k in MODEL_INPUT_KEYS}
            logprobs = (
                answer_logprobs(self.model, inputs, self.tokenizer.pad_token_id) if with_logprob else None
            )
            outputs = eval_text_similarity(
                self.model, inputs, tokenizer=self.tokenizer, generation_args=self.generation_args
            )
            for i, output in enumerate(outputs):
                records.append({
                    **{k: v[i] for k, v in meta.items()},
                    **output,
                    **({"answer_logprob": logprobs[i]} if logprobs is not None else {}),
                    "refusal": is_refusal(output["generation"], self.refusal_patterns),
                    "degenerate": is_degenerate(output["generation"]),
                })
        return records[:max_samples] if max_samples is not None else records

    @staticmethod
    def summarize(records: Sequence[Mapping[str, Any]], hallucination_rouge: float = 0.3) -> Dict[str, Any]:
        """Aggregate of `generate` records.

        `hallucination_rate`: a confident (no refusal, not degenerate) answer that does NOT match the ground truth
        (rougeL_recall < `hallucination_rouge`) - the behaviour expected on entities the model has never seen.
        """
        n = len(records)
        if n == 0:
            return {"n": 0}
        mean = lambda key: sum(float(r[key]) for r in records) / n
        out: Dict[str, Any] = {
            "n": n,
            "rougeL_recall": mean("rougeL_recall"),
            "rouge1_recall": mean("rouge1_recall"),
            "refusal_rate": mean("refusal"),
            "degenerate_rate": mean("degenerate"),
            "hallucination_rate": sum(
                (not r["refusal"]) and (not r["degenerate"]) and r["rougeL_recall"] < hallucination_rouge
                for r in records
            ) / n,
            "mean_words": sum(len(str(r["generation"]).split()) for r in records) / n,
        }
        if all("answer_logprob" in r for r in records):
            out["answer_logprob"] = mean("answer_logprob")
            out["answer_prob"] = sum(math.exp(r["answer_logprob"]) for r in records) / n
        return out

    # def run(
    #     self,
    #     dataloader: Iterable[Mapping[str, Any]],
    #     max_samples: Optional[int] = None,
    #     desc: str = "",
    #     before_batch: Optional[Callable[[Mapping[str, Any]], None]] = None,
    # ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    #     """`generate` + `summarize`: returns `(records, summary)`."""
    #     records = self.generate(dataloader, max_samples=max_samples, desc=desc, before_batch=before_batch)
    #     return records, self.summarize(records)
