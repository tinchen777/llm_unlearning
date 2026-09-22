"""Quality control for generated neighbours (the "unknown-ness" check).

A neighbour is only useful as a redirection target if the model genuinely does
not know it. Two independent gates are applied:

1. `NameCollisionChecker` - string-level: the new name must not repeat or closely
   resemble any name in the training data (entity names, book titles, other
   neighbours).
2. `FamiliarityProbe` - model-level: the *pre-unlearning* model is queried about
   the new name; a name the model answers confidently about is rejected.

The same probe answers the D1 pre-check of the plan ("does the model answer
naturally and diversely about strangers?"): run `probe_names` on the forgotten
entities and on random new names and compare the two familiarity distributions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from .schema import Neighbor

logger = logging.getLogger(__name__)

# Phrases that signal the model has no knowledge of the entity.
UNCERTAINTY_MARKERS = (
    "i don't know", "i do not know", "i'm not aware", "i am not aware",
    "no information", "not familiar", "cannot find", "can't find",
    "there is no", "there isn't", "does not appear", "doesn't appear",
    "unable to find", "no widely", "not a well-known", "fictional character",
    "i couldn't find", "i could not find", "no record",
)

DEFAULT_PROBE_TEMPLATES = (
    "Who is {name}?",
    "What is {name} known for?",
    "Tell me about the author {name}.",
)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z\s]", " ", name.lower()).strip()


def name_tokens(name: str) -> List[str]:
    return [t for t in normalize_name(name).split() if len(t) > 1]


class NameCollisionChecker:
    """Reject names that repeat or resemble anything in the training corpus."""

    def __init__(
        self,
        known_names: Sequence[str],
        max_similarity: float = 0.6,
        block_shared_surname: bool = True,
    ):
        self.known_names = [n for n in known_names if n]
        self.max_similarity = max_similarity
        self.block_shared_surname = block_shared_surname
        self._known_norm = [normalize_name(n) for n in self.known_names]
        self._surnames = {name_tokens(n)[-1] for n in self.known_names if name_tokens(n)}

    def check(self, name: str) -> Dict[str, Any]:
        norm = normalize_name(name)
        tokens = name_tokens(name)
        best_score, best_match = 0.0, ""
        for known, known_norm in zip(self.known_names, self._known_norm):
            score = difflib.SequenceMatcher(None, norm, known_norm).ratio()
            if score > best_score:
                best_score, best_match = score, known
        surname_clash = bool(tokens) and self.block_shared_surname and tokens[-1] in self._surnames
        passed = best_score < self.max_similarity and not surname_clash
        return {
            "max_name_similarity": round(best_score, 4),
            "closest_known_name": best_match,
            "surname_clash": surname_clash,
            "passed": passed,
        }

    def add(self, name: str) -> None:
        """Register an accepted name so later candidates cannot duplicate it."""
        self.known_names.append(name)
        self._known_norm.append(normalize_name(name))
        tokens = name_tokens(name)
        if tokens:
            self._surnames.add(tokens[-1])


@dataclass
class FamiliarityResult:
    """Per-name output of the model probe."""

    name: str
    uncertainty_rate: float = 0.0     # fraction of probes answered with "I don't know"-style text
    mean_answer_nll: float = 0.0      # NLL the model assigns to its own answer (confidence proxy)
    name_nll: float = 0.0             # NLL of the name inside a factual frame (memorisation proxy)
    generations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncertainty_rate": round(self.uncertainty_rate, 4),
            "mean_answer_nll": round(self.mean_answer_nll, 4),
            "name_nll": round(self.name_nll, 4),
            "generations": self.generations,
        }


class FamiliarityProbe:
    """Ask the pre-unlearning model about a name and measure how much it knows.

    Args:
        model / tokenizer: the model the unlearning will be applied to.
        templates: probe questions, `{name}` is substituted.
        apply_chat_template: wrap probes in the model's chat template.
        max_new_tokens: generation budget per probe.
        known_frame: sentence used for the memorisation proxy; the NLL of the
            name inside this frame is low for entities the model has seen.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        templates: Sequence[str] = DEFAULT_PROBE_TEMPLATES,
        apply_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 64,
        batch_size: int = 8,
        known_frame: str = "The acclaimed author {name} is best known for the book",
        device: Optional[str] = None,
    ):
        import torch  # local import: the pipeline must run without torch installed

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.templates = list(templates)
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.known_frame = known_frame
        self.device = device or str(getattr(model, "device", "cpu"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ api
    def probe(self, name: str) -> FamiliarityResult:
        return self.probe_names([name])[0]

    def probe_names(self, names: Sequence[str]) -> List[FamiliarityResult]:
        results = [FamiliarityResult(name=n) for n in names]
        if not names:
            return results

        for result in results:
            prompts = [t.format(name=result.name) for t in self.templates]
            generations, nlls = self._generate(prompts)
            result.generations = generations
            result.mean_answer_nll = float(sum(nlls) / len(nlls)) if nlls else 0.0
            result.uncertainty_rate = sum(
                1.0 for g in generations if _is_uncertain(g)
            ) / max(len(generations), 1)
            result.name_nll = self._name_nll(result.name)
        return results

    # -------------------------------------------------------------- internals
    def _wrap(self, prompt: str) -> str:
        if not self.apply_chat_template or not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompts: Sequence[str]):
        torch = self.torch
        texts = [self._wrap(p) for p in prompts]
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            batch = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = side

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            prompt_len = batch["input_ids"].shape[1]
            completions = generated[:, prompt_len:]
            answers = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
            # NLL the model assigns to its own answer: low = confident, fluent claim
            nlls = self._sequence_nll(generated, prompt_len, batch["attention_mask"])
        if was_training:
            self.model.train()
        return [a.strip() for a in answers], nlls

    def _sequence_nll(self, sequences, prompt_len: int, prompt_mask) -> List[float]:
        torch = self.torch
        attention = torch.cat(
            [prompt_mask, torch.ones_like(sequences[:, prompt_len:])], dim=1
        )
        logits = self.model(input_ids=sequences, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = sequences[:, 1:]
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_nll, dtype=torch.bool)
        mask[:, prompt_len - 1 :] = True
        mask &= targets != self.tokenizer.pad_token_id
        counts = mask.sum(dim=1).clamp(min=1)
        return ((token_nll * mask).sum(dim=1) / counts).tolist()

    def _name_nll(self, name: str) -> float:
        """NLL of `name` inside a factual frame: low means the model knows it."""
        torch = self.torch
        frame = self.known_frame.format(name=name)
        prefix = frame.split(name)[0] if name in frame else frame
        prefix_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(frame, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if full_ids.shape[1] <= prefix_ids.shape[1]:
            return 0.0
        full_ids = full_ids.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=full_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = full_ids[:, 1:]
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = prefix_ids.shape[1] - 1
        return float(token_nll[0, start:].mean().item())


def _is_uncertain(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in UNCERTAINTY_MARKERS)


@dataclass
class VerificationThresholds:
    """Acceptance rule for a neighbour candidate.

    A candidate passes the model gate when the model shows *no* familiarity:
    either it explicitly says it does not know (`uncertainty_rate` high enough)
    or the name is clearly not memorised (`name_nll` above the threshold).
    """

    max_name_similarity: float = 0.6
    min_uncertainty_rate: float = 0.34
    min_name_nll: float = 0.0  # 0 disables the memorisation gate

    def accepts(self, familiarity: Optional[FamiliarityResult]) -> bool:
        if familiarity is None:
            return True
        if familiarity.uncertainty_rate >= self.min_uncertainty_rate:
            return True
        return bool(self.min_name_nll) and familiarity.name_nll >= self.min_name_nll


def verify_neighbors(
    candidates: Sequence[Neighbor],
    collision_checker: NameCollisionChecker,
    probe: Optional[FamiliarityProbe] = None,
    thresholds: Optional[VerificationThresholds] = None,
) -> List[Neighbor]:
    """Annotate candidates with verification results and mark `passed`.

    The checker is updated with every accepted name, so neighbours of different
    entities never collide with each other either.
    """
    thresholds = thresholds or VerificationThresholds()
    surviving: List[Neighbor] = []

    # Checked one by one (not batched) so that a name accepted earlier in this
    # batch is already registered when the next candidate is tested.
    for candidate in candidates:
        check = collision_checker.check(candidate.name)
        familiarity = (
            probe.probe(candidate.name) if (probe is not None and check["passed"]) else None
        )
        model_ok = thresholds.accepts(familiarity)
        candidate.verification = {
            "name_check": check,
            "familiarity": familiarity.to_dict() if familiarity else None,
            "model_gate_passed": model_ok,
        }
        candidate.passed = bool(check["passed"] and model_ok)
        if candidate.passed:
            collision_checker.add(candidate.name)
            surviving.append(candidate)
        else:
            logger.debug(
                "Rejected neighbour `%s` (name_ok=%s, model_ok=%s)",
                candidate.name, check["passed"], model_ok,
            )
    return surviving
