"""Sentence embeddings and the layer-selection score (s1 - s2) of LUNAR (section 3.2).

For every candidate layer the forget prompts are answered by the steered model; the responses are embedded and compared to
  - a list of DESIRED responses ("I apologize, but this information cannot be provided"): similarity s1, to be maximised,
  - a list of responses UNRELATED to unlearning (harmful / dangerous / unethical content): similarity s2, to be minimised,
and the layer with the largest s1 - s2 is selected.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Sequence


class SentenceEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings of a (sentence-transformers style) encoder, via `transformers`."""

    def __init__(self, name_or_path: str, device: Any = None, max_length: int = 256):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModel.from_pretrained(name_or_path).eval()
        if device is not None:
            self.model.to(device)
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.model.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(out)


def similarity_to_list(embeddings: torch.Tensor, targets: torch.Tensor, aggregate: str = "mean") -> torch.Tensor:
    """Cosine similarity of every response embedding [n, e] to a list of target embeddings [m, e] -> [n]:
    `aggregate` = `mean` over the list, or `max` (the closest one)."""
    cos = F.normalize(embeddings.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).T  # [n, m]
    if aggregate == "mean":
        return cos.mean(dim=1)
    if aggregate == "max":
        return cos.max(dim=1).values
    raise ValueError(f"aggregate must be `mean` or `max`, got `{aggregate}`.")


def layer_score(
    response_embeddings: torch.Tensor,
    desired_embeddings: torch.Tensor,
    unrelated_embeddings: torch.Tensor,
    aggregate: str = "mean",
) -> Dict[str, float]:
    """`s1` (similarity to the desired responses), `s2` (to the unrelated ones) and `score = s1 - s2`, each averaged over
    the responses of one layer."""
    s1 = similarity_to_list(response_embeddings, desired_embeddings, aggregate).mean().item()
    s2 = similarity_to_list(response_embeddings, unrelated_embeddings, aggregate).mean().item()
    return {"s1": s1, "s2": s2, "score": s1 - s2}


def rank_layers(per_layer: Dict[int, Dict[str, Any]], max_degenerate: float) -> List[int]:
    """Layers by decreasing `score`; layers whose steered answers are degenerate more often than `max_degenerate` are
    dropped (all layers are kept if none passes)."""
    valid = [l for l, v in per_layer.items() if v["degenerate_rate"] <= max_degenerate] or list(per_layer)
    return sorted(valid, key=lambda l: -per_layer[l]["score"])
