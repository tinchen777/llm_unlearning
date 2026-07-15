
from __future__ import annotations
import numpy as np
import scipy as sc
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, Dict, TYPE_CHECKING

from .base import MetricFunc

if TYPE_CHECKING:
    from utils.config import TrackingConfig


@MetricFunc
def hm_aggregate(pre_compute: Dict[str, Any], **kwargs):
    values = [result["agg_value"] for result in pre_compute.values()]
    return {"agg_value": sc.stats.hmean(values)}


@MetricFunc
def classifier_prob(
    pre_compute: Dict[str, Any],
    classifier_model_args: TrackingConfig,
    classifier_tokenization_args: TrackingConfig,
    batch_size: int = 32,
    max_length: int = 512,
    class_id: int = 0,
    text_key: str = "generation",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
):
    tokenizer = AutoTokenizer.from_pretrained(**classifier_tokenization_args)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        **classifier_model_args
    ).to(device)

    data_list = [
        {"text": entry[text_key], "index": int(key)}
        for key, entry in pre_compute["text"]["value_by_index"].items()
    ]
    # Create DataLoader
    dataloader = DataLoader(
        HFDataset.from_list(data_list),  # type: ignore
        batch_size=batch_size,
        shuffle=False
    )
    with tqdm(
        dataloader,
        total=len(dataloader),
        desc="Calculating [classifier prob]",
        unit="batch(es)",
        colour="blue"
    ) as pbar:
        scores_by_index = {}
        for batch in pbar:
            batch_texts = batch["text"]
            batch_indices = batch["index"].tolist()

            # Tokenize the batch of texts
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run the classifier
            with torch.no_grad():
                outputs = classifier(**inputs)
            # Convert logits to probabilities
            scores = outputs.logits.softmax(dim=-1)[:, class_id].cpu().numpy().tolist()

            # Map predictions to labels
            for idx, prob, text in zip(batch_indices, scores, batch_texts):
                # Add the prediction to the original data
                scores_by_index[idx] = {"score": prob, text_key: text}
    del classifier
    torch.cuda.empty_cache()

    class_scores = np.array([evals["score"] for evals in scores_by_index.values()])
    return {"agg_value": np.mean(class_scores), "value_by_index": scores_by_index}
