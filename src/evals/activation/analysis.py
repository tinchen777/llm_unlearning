"""Analysis of the activations of a model on the question sets of the NEAR experiment (shared by `METHOD/NEAR/probe.py`, which
analyses ONE original model, and the `activation_probe` metric, which analyses the model under evaluation):

- the dataset means, the global r_UV (reference - forget, as LUNAR) and per layer norms / cosines;
- the linear separability (cross-validated AUC along the difference of means) of pairs of question sets, folds split by
  question (`auc[A | B]`) or by author (`auc_group[A | B]`).
"""

from __future__ import annotations
import itertools
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from utils.common import save_logs

logger = logging.getLogger("eval.activation")

META_KEYS = ("author_id", "neighbor_id")

ACTIVATION_NOTE = (
    "auc[A | B]: 2-fold cross-validated AUC of separating A from B along their difference of means "
    "(0.5 = indistinguishable), folds split BY QUESTION. auc_group[A | B]: the same with folds split "
    "BY AUTHOR (a forget author and his neighbours share a fold), so an author's name alone cannot "
    "carry the separation across the split; prefer it. It is null when a set has fewer than 4 authors "
    "(forget01 has 2: the direction cannot generalise across so few and the value would be ~0 or ~1 "
    "by chance); use forget05 / forget10 for it. IMPORTANT: both measure how different the two "
    "QUESTION SETS are (other authors, templates, lengths: forget | holdout is already ~0.7 at layer 0 "
    "where no knowledge can be encoded), NOT whether the model knows them. Do not expect 0.5 for "
    "sets about different authors, whatever the model. Use (1) only matched pairs: forget | neighbour "
    "(neighbours are rewrites of the forget questions) and (2) the difference of the SAME pair between "
    "two models, e.g. full - retain (src/METHOD/NEAR/neighbor_compare.py): content effects cancel, knowledge "
    "effects remain. `r_uv_rel_norm` = ||r_UV|| / ||mean forget activation||."
)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """ROC-AUC = P(score(pos) > score(neg)) (ties count 1/2), via the Mann-Whitney U statistic."""
    scores = torch.cat([pos, neg]).double()
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    for value in scores.unique():  # average ranks of ties
        tie = scores == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = len(pos), len(neg)
    return ((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def _group_folds(
    groups_a: Sequence[Hashable],
    groups_b: Sequence[Hashable],
    folds: int,
    g: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold of every sample of both classes, all samples of a group in ONE fold, so a test sample never has a sibling
    (same author / same neighbour) in the fold the direction is fitted on. A group present in BOTH classes (e.g. a
    forget author and the neighbours generated for him) gets the same fold in both; the other groups are spread so
    that every fold holds samples of every class."""
    keys_a, keys_b = set(groups_a), set(groups_b)
    shared = sorted(keys_a & keys_b, key=str)
    fold_of: Dict[Hashable, int] = {k: r % folds for r, k in enumerate(
        [shared[i] for i in torch.randperm(len(shared), generator=g).tolist()])}
    out = []
    for groups, keys in ((groups_a, keys_a), (groups_b, keys_b)):
        local = dict(fold_of)  # class-only groups are balanced per class
        counts = [0] * folds
        for k in groups:
            if k in fold_of:
                counts[fold_of[k]] += 1
        own = sorted(keys - set(fold_of), key=str)
        size = {k: sum(1 for x in groups if x == k) for k in own}
        for i in torch.randperm(len(own), generator=g).tolist():
            k = own[i]
            f = min(range(folds), key=lambda j: counts[j])
            local[k] = f
            counts[f] += size[k]
        ids = torch.tensor([local[k] for k in groups])
        if any(not bool((ids == f).any()) for f in range(folds)):
            raise ValueError(f"Cannot split {len(keys)} group(s) over {folds} folds with every fold non-empty.")
        out.append(ids)
    return out[0], out[1]


def mean_diff_auc(
    a: torch.Tensor,
    b: torch.Tensor,
    folds: int = 2,
    seed: int = 0,
    groups_a: Optional[Sequence[Hashable]] = None,
    groups_b: Optional[Sequence[Hashable]] = None,
    min_groups: int = 4,
) -> float:
    """
    How linearly separable are the activations `a` [n, d] and `b` [m, d]?
    Cross-validated AUC of the projection on the difference of class means, the direction being fitted on the OTHER folds (0.5 = indistinguishable, 1.0 = perfectly separated along a single direction, like r_UV).

    `groups_a` / `groups_b` (one key per row): split the folds by group instead of by sample (see `_group_folds`).
    Without them questions of one author land on both sides of the split and the author's name alone separates the classes; with them only what generalises across authors counts. Both must be given, or neither. With fewer than `min_groups` groups in a class (e.g. forget01 has 2 authors) the direction fitted on the other group(s) cannot generalise and the AUC is ~0 or ~1 by chance (seen on a toy: 0.10 vs 0.80 for two near-identical models), so a ValueError is raised instead of returning a number.
    """
    if (groups_a is None) != (groups_b is None):
        raise ValueError("Give groups for both classes or for neither.")
    if groups_a is not None:
        n_groups = min(len(set(groups_a)), len(set(groups_b)))  # type: ignore[arg-type]
        if n_groups < min_groups:
            raise ValueError(f"group-wise AUC needs at least {min_groups} groups per class, got {n_groups}.")
    if len(a) < folds or len(b) < folds:
        raise ValueError(f"Need at least {folds} samples per class, got {len(a)} and {len(b)}.")
    g = torch.Generator().manual_seed(seed)
    if groups_a is None:
        fa = torch.randperm(len(a), generator=g) % folds
        fb = torch.randperm(len(b), generator=g) % folds
    else:
        fa, fb = _group_folds(list(groups_a), list(groups_b), folds, g)  # type: ignore[arg-type]
    a, b = a.double(), b.double()
    scores = []
    for f in range(folds):
        direction = a[fa != f].mean(0) - b[fb != f].mean(0)
        scores.append(auc(a[fa == f] @ direction, b[fb == f] @ direction))
    return sum(scores) / folds


def unlearning_vector(reference_mean: torch.Tensor, forget_mean: torch.Tensor) -> torch.Tensor:
    """LUNAR unlearning vector: r_UV = mean(a | D_ref) - mean(a | D_forget), per position/layer."""
    if reference_mean.shape != forget_mean.shape:
        raise ValueError(f"Shape mismatch: {tuple(reference_mean.shape)} vs {tuple(forget_mean.shape)}.")
    return reference_mean - forget_mean


def sample_group(split: str, index: int, meta: Mapping[str, Any], forget_split: str, qa_per_author: int, author_offset: int) -> Hashable:
    """Group (= author or neighbour) of a question, the unit of the group-wise cross-validation.

    - neighbour rows: the author they were generated for (`author_id`), so a forget author and his neighbours share
      a group (and thus a fold); the neighbour csv also has `neighbor_id`, the rows of one neighbour share
      `(author_id, neighbor_id)` but the author is what matters for the leakage;
    - every other split (TOFU stores the 20 QAs of an author contiguously): `index // qa_per_author`,
      shifted by `author_offset` for the forget split so that it matches the `author_id` of the neighbour csv."""
    if "author_id" in meta:
        return f"author:{meta['author_id']}"
    author = index // qa_per_author
    if split == forget_split:
        return f"author:{author + author_offset}"
    return f"{split}:{author}"


def check_r_uv_identity(
    saved: Mapping[str, Any],
    *,
    forget_split: Optional[str] = None,
    num_forget: Optional[int] = None,
    model: Optional[str] = None,
    source: str = "r_uv.pt",
) -> None:
    """The data / model of this stage must be the ones `r_uv.pt` was computed on (`None` skips that check)."""
    for key, current in (("forget_split", forget_split), ("num_forget", num_forget), ("model", model)):
        if current is None:
            continue
        if key not in saved:
            raise ValueError(f"{source} has no `{key}` (written by an older r_uv.py): run r_uv.py again.")
        if saved[key] != current:
            raise ValueError(f"{source} was computed with {key}={saved[key]!r}, but this run uses {key}={current!r}.")


def analyze_activations(
    act_collector: Any,
    split_data: Any,
    forget: str,
    reference: str,
    seen: Sequence[str],
    unseen: Sequence[str],
    qa_per_author: int,
    author_offset: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-sample activations of every split of `split_data` (an `ActivationCollector`, loaders already configured), then the
    dataset means, the global r_UV (`reference` - `forget`) and the per-layer diagnostics.

    Returns `means`, `r_uv`, `per_layer` (`{layer: {name: value}}`), `num_samples`, `num_groups`, `forget`, `reference`.
    """
    per_sample: Dict[str, torch.Tensor] = {}  # {split: Tensor[n, n_pos, n_layers, d]}
    groups: Dict[str, List[Hashable]] = {}    # {split: [group key per row]}
    for split, data in split_data.items():
        acts, meta = act_collector.per_sample(dataloader=data.loader(), desc=split, meta_keys=META_KEYS)
        order = sorted(acts)
        per_sample[split] = torch.stack([acts[i] for i in order]).float()
        groups[split] = [sample_group(split, i, meta.get(i, {}), forget, qa_per_author, author_offset) for i in order]
    means = {split: a.mean(0) for split, a in per_sample.items()}  # [n_pos, n_layers, d]
    # r_UV[p, l]: shift that moves the forget activations onto the reference (LUNAR: ref - forget)
    r_uv = unlearning_vector(means[reference], means[forget])  # [n_pos, n_layers, d]

    # separability per layer, on the activations averaged over the end-of-instruction positions
    pooled = {split: a.mean(1) for split, a in per_sample.items()}  # [n, n_layers, d]
    seen_splits = [s for s in seen if s in pooled]
    unseen_splits = [s for s in unseen if s in pooled]
    pairs = [(f"{'+'.join(seen_splits)} | {'+'.join(unseen_splits)}", seen_splits, unseen_splits)] if seen_splits and unseen_splits else []
    pairs += [(f"{a} | {b}", [a], [b]) for a, b in itertools.combinations(list(per_sample), 2)]

    def group_auc(x: torch.Tensor, y: torch.Tensor, ga: list, gb: list):
        try:
            return mean_diff_auc(x, y, seed=seed, groups_a=ga, groups_b=gb)
        except ValueError:  # fewer groups than needed (forget01 has 2 authors): said once per pair below
            return None

    per_layer: Dict[str, Dict[str, Any]] = {}
    for i, layer in enumerate(act_collector.layers):
        cat = lambda names: torch.cat([pooled[s][:, i] for s in names])
        cat_groups = lambda names: [g for s in names for g in groups[s]]
        norm_forget = means[forget][:, i].norm(dim=-1).mean().item()
        r_uv_norm = r_uv[:, i].norm(dim=-1).mean().item()
        per_layer[str(layer)] = {
            "r_uv_norm": r_uv_norm,
            "r_uv_rel_norm": r_uv_norm / norm_forget,  # size of the redirection relative to the activation
            **{f"norm[{s}]": means[s][:, i].norm(dim=-1).mean().item() for s in means},
            **{
                f"cos[{a},{b}]": F.cosine_similarity(means[a][:, i], means[b][:, i], dim=-1).mean().item()
                for a, b in itertools.combinations(means, 2)
            },
            **{f"auc[{name}]": mean_diff_auc(cat(a), cat(b), seed=seed) for name, a, b in pairs},
            **{f"auc_group[{name}]": group_auc(cat(a), cat(b), cat_groups(a), cat_groups(b)) for name, a, b in pairs},
        }
    for name, a, b in pairs:
        if all(v[f"auc_group[{name}]"] is None for v in per_layer.values()):
            n_groups = min(len({g for s in a for g in groups[s]}), len({g for s in b for g in groups[s]}))
            logger.warning(f"auc_group[{name}] is null at every layer: {n_groups} author(s) in the smaller set, 4 are needed.")
    return {
        "means": means, "r_uv": r_uv, "per_layer": per_layer, "forget": forget, "reference": reference,
        "num_samples": {split: len(a) for split, a in per_sample.items()},
        "num_groups": {split: len(set(g)) for split, g in groups.items()},
    }


def save_activation_outputs(result: Mapping[str, Any], act_collector: Any, output_dir: Path) -> None:
    """`activations.pt` (means, r_UV, settings) and `activations_summary.json` (per-layer diagnostics) in `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "r_uv": result["r_uv"], "means": result["means"], "forget": result["forget"], "reference": result["reference"],
        "positions": act_collector.positions, "layers": act_collector.layers, "point": act_collector.point,
        "num_samples": result["num_samples"], "num_groups": result["num_groups"],
    }, output_dir / "activations.pt")
    save_logs({"note": ACTIVATION_NOTE, "layers": result["per_layer"]}, output_dir / "activations_summary.json")


def _scalar_name(key: str) -> Optional[str]:
    """`auc[forget | neighbor]` -> `auc_forget_neighbor`, `auc_group[forget+retain | neighbor+holdout]` -> `auc_group_...`."""
    if not key.startswith(("auc[", "auc_group[")):
        return None
    head, name = key[:-1].split("[", 1)
    name = name.replace(" | ", "_").replace("+", "_and_")
    return f"{head}_{name}"


def activation_scalars(per_layer: Mapping[str, Mapping[str, Any]], focus_layer: Optional[int] = None) -> Dict[str, Optional[float]]:
    """Scalars of the per-layer diagnostics, to publish as metrics (`expose`):

    for every AUC `auc_<A>_<B>` / `auc_group_<A>_<B>` (the seen / unseen pair is `<seen splits>_<unseen splits>` joined by
    `_and_`): `_max` over the layers, `_argmax` (the layer), `_focus` (at `focus_layer`); `r_uv_rel_norm_max` / `_focus`.
    A value is `None` where the quantity does not exist (group-wise AUC with too few authors).
    """
    layers = list(per_layer)
    out: Dict[str, Optional[float]] = {}

    def add(name: str, series: Dict[str, Optional[float]]):
        valid = {layer: v for layer, v in series.items() if v is not None}
        out[f"{name}_max"] = max(valid.values()) if valid else None
        out[f"{name}_argmax"] = int(max(valid, key=valid.get)) if valid else None  # type: ignore[arg-type]
        if focus_layer is not None:
            out[f"{name}_focus"] = series.get(str(focus_layer))

    for key in per_layer[layers[0]]:
        name = _scalar_name(key)
        if name is not None:
            add(name, {layer: per_layer[layer][key] for layer in layers})
    add("r_uv_rel_norm", {layer: per_layer[layer]["r_uv_rel_norm"] for layer in layers})
    return out
