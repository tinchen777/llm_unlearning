"""Generate text responses for a dataloader, independent of any Trainer / evaluator.

Reuses `eval_text_similarity` (the generation step behind every ROUGE metric), so the
decoding behaviour (`generation_args`, stop strings, EOS cut-off) is identical to the evals.

Requirements on the loader (same as the generation metrics):
- prompt-only `input_ids` -> dataset built with `predict_with_generate: true`
- LEFT padding -> collator `padding_side: left` (batched generation appends after the last column)
"""

from __future__ import annotations
import torch
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Mapping, Optional

from evals.metrics.metric_utils import eval_text_similarity

# keys the model / eval function consume; everything else in a batch is per-sample metadata
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "labels")


def _as_list(values: Any) -> List[Any]:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)


def generate_responses(
    model: Any,
    tokenizer: Any,
    loader: Iterable[Mapping[str, Any]],
    generation_args: Any,
    max_samples: Optional[int] = None,
    desc: str = "",
) -> List[Dict[str, Any]]:
    """Generate a response for every sample of `loader`.

    Args:
        generation_args: a `TrackingConfig` (e.g. `cfg["generation_args"]`); see configs/generation/.
        max_samples: stop after this many samples (None -> the whole loader).
    Returns:
        One record per sample: the batch metadata (`index`, `author_id`, ...) plus
        `input` (decoded prompt), `ground_truth`, `generation` and the ROUGE scores.
    """
    model.eval()
    records: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"Generating [{desc}]", unit="batch(es)", colour="blue"):
        if max_samples is not None and len(records) >= max_samples:
            break
        # metadata (index, author_id, ...) is not a tensor input of the model: set it aside
        meta = {k: _as_list(batch.pop(k)) for k in [k for k in batch if k not in MODEL_INPUT_KEYS]}
        outputs = eval_text_similarity(model, batch, tokenizer=tokenizer, generation_args=generation_args)
        records.extend(
            {**{k: v[i] for k, v in meta.items()}, **output}
            for i, output in enumerate(outputs)
        )
    return records[:max_samples] if max_samples is not None else records
