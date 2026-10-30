"""Neighbour bookkeeping of the NEAR experiment (no hydra, no Trainer).

    forget question  -> author          (`forget_authors`)
    neighbour rows   -> (author, neighbour)
    per author a fixed random order of its neighbours; the K-neighbour reference of author a is the
    FIRST K of that order, so the sets are nested (K=1 in K=5 in K=15) and the ablation is controlled.
        r_UV[K][a] = mean_{n in first K neighbours of a} mean(act | neighbour n)  -  mean(act | forget of a)

Used by r_uv.py, select_layer.py and w_down.py of this directory. The LUNAR-style editing itself (W_down training,
steering) is generic and lives in `model.edit` / `model.steer`.
"""

from __future__ import annotations
import logging
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from METHOD._common.activations import unlearning_vector

logger = logging.getLogger(__name__)

SELECTION_FILE = "layer_selection.json"


def config_dir(output_dir: Path, k: int, coeff: float, shift_mode: str) -> Path:
    """Folder of one setting (K, coeff, shift mode): the selected layer depends on all three, so W_down training
    (w_down.py) must read the selection of the SAME setting."""
    return Path(output_dir) / f"k{k}_c{coeff:g}_{shift_mode}"


def run_dir(output_dir: Path, k: int, coeff: float, shift_mode: str, layers: Sequence[int], solver: str) -> Path:
    """Folder of one W_down run inside the setting folder: the trained layers and the solver (`_cf` = closed form)."""
    return config_dir(output_dir, k, coeff, shift_mode) / (
        "l" + "-".join(str(l) for l in layers) + ("_cf" if solver == "closed_form" else "")
    )


def selection_path(output_dir: Path, k: int, coeff: float, shift_mode: str) -> Path:
    """Layer-selection file of a setting, written by select_layer.py and read by w_down.py."""
    return config_dir(output_dir, k, coeff, shift_mode) / SELECTION_FILE


def forget_authors(indices: Iterable[Any], qa_per_author: int, author_offset: int = 0) -> List[int]:
    """Author of each forget row. TOFU stores the QAs of one author contiguously (20 per author),
    so `author = row_index // qa_per_author + author_offset`. `author_offset` aligns this numbering
    with the `author_id` column of the neighbour csv."""
    return [int(i) // qa_per_author + author_offset for i in indices]


def neighbors_by_author(author_ids: Iterable[Any], neighbor_ids: Iterable[Any]) -> Dict[Any, List[Any]]:
    """{author: sorted list of distinct neighbour ids} from the two metadata columns of the neighbour rows."""
    out: Dict[Any, Set[Any]] = {}
    for a, n in zip(author_ids, neighbor_ids):
        out.setdefault(a, set()).add(n)
    return {a: sorted(ns) for a, ns in out.items()}


def check_neighbors(forget_author_set: Set[Any], nb_by_author: Mapping[Any, Sequence[Any]], ks: Sequence[int]):
    """Fail early (before any model forward) when the forget authors and the neighbour csv disagree."""
    missing = forget_author_set - set(nb_by_author)
    if missing:
        raise ValueError(
            f"{len(missing)} forget author(s) have no neighbours: {sorted(missing)}.\n"
            f"  forget authors (row_index // qa_per_author + author_offset): {sorted(forget_author_set)}\n"
            f"  `author_id` values in the neighbour data                  : {sorted(nb_by_author)}\n"
            f"  -> fix `ruv.qa_per_author` / `ruv.author_offset` so that both numberings match."
        )
    extra = set(nb_by_author) - forget_author_set
    if extra:
        logger.warning(f"Neighbour data has authors that are not in the forget split (ignored): {sorted(extra)}")
    short = {a: len(nb_by_author[a]) for a in forget_author_set if len(nb_by_author[a]) < max(ks)}
    if short:
        raise ValueError(f"Need at least max(ks)={max(ks)} neighbours per author, but got {short} (author: #neighbours).")


def nested_neighbor_order(nb_by_author: Mapping[Any, Sequence[Any]], seed: int) -> Dict[Any, List[Any]]:
    """One random permutation of the neighbours of every author (deterministic given `seed`,
    independent of the order of the csv rows). The K-neighbour reference is `order[a][:K]`."""
    rng = random.Random(seed)
    order = {}
    for author in sorted(nb_by_author):
        ids = sorted(nb_by_author[author])  # sort first -> the order does not depend on how the ids are listed
        rng.shuffle(ids)
        order[author] = ids
    return order


def compute_r_uv(
    forget_means: Mapping[Any, torch.Tensor],
    neighbor_means: Mapping[Tuple[Any, Any], torch.Tensor],
    order: Mapping[Any, Sequence[Any]],
    ks: Sequence[int],
) -> Dict[int, Dict[Any, torch.Tensor]]:
    """r_UV[K][author] = mean over the first K neighbours (each neighbour weighs the same, whatever
    its number of questions) - mean forget activation of the author. Shapes [n_pos, n_layers, d]."""
    r_uv: Dict[int, Dict[Any, torch.Tensor]] = {}
    for k in ks:
        r_uv[k] = {}
        for author, forget_mean in forget_means.items():
            reference = torch.stack([neighbor_means[(author, n)] for n in order[author][:k]]).mean(dim=0)
            r_uv[k][author] = unlearning_vector(reference, forget_mean)
    return r_uv


def summarize_r_uv(
    r_uv: Mapping[int, Mapping[Any, torch.Tensor]],
    forget_means: Mapping[Any, torch.Tensor],
    layers: Sequence[int],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Per K and layer (averaged over authors and positions), to read the ablation at a glance:

    - `norm`      : ||r_UV||
    - `rel_norm`  : ||r_UV|| / ||mean forget activation||   (how far the redirection moves the activation)
    - `cos_to_kmax`: cosine(r_UV[K][a], r_UV[max K][a]) over (positions, d): is a K-neighbour reference
                    already pointing where the largest reference points?
    """
    kmax = max(r_uv)
    summary: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k, per_author in r_uv.items():
        authors = list(per_author)
        r = torch.stack([per_author[a] for a in authors])  # [A, n_pos, n_layers, d]
        base = torch.stack([forget_means[a] for a in authors])
        ref = torch.stack([r_uv[kmax][a] for a in authors])
        norm = r.norm(dim=-1).mean(dim=1).mean(dim=0)  # [n_layers]
        rel = (r.norm(dim=-1) / base.norm(dim=-1)).mean(dim=1).mean(dim=0)
        cos = F.cosine_similarity(r.movedim(2, 1).flatten(2), ref.movedim(2, 1).flatten(2), dim=-1).mean(dim=0)  # [n_layers]
        summary[k] = {
            str(layer): {"norm": norm[i].item(), "rel_norm": rel[i].item(), "cos_to_kmax": cos[i].item()}
            for i, layer in enumerate(layers)
        }
    return summary
