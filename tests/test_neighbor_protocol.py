"""The neighbour data protocol: the csv has a `split` (train / heldout) and a `train_rank` (1..n of the train neighbours of an author).
r_UV (K = the first K by rank) and the retained tokens of W_down use the TRAIN neighbours, the evaluation the HELDOUT ones."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_smoke import common_args, make_assets, run  # noqa: E402

from METHOD.NEAR._utils.neighbors import ranked_neighbor_order  # noqa: E402
from data.datasets.datasets_utils import filter_rows  # noqa: E402


def test_ranked_order_is_nested_by_train_rank_and_checked():
    authors = ["a", "a", "a", "a", "b", "b"]
    neighbors = ["n3", "n1", "n2", "n1", "x", "y"]
    ranks = [3, 1, 2, 1, 2, 1]
    assert ranked_neighbor_order(authors, neighbors, ranks) == {"a": ["n1", "n2", "n3"], "b": ["y", "x"]}  # K = the first K entries
    with pytest.raises(ValueError, match="two different train_rank"):
        ranked_neighbor_order(["a", "a"], ["n1", "n1"], [1, 2])
    with pytest.raises(ValueError, match="1\\.\\.2"):  # a gap, or a held-out neighbour (rank -1) left in the data
        ranked_neighbor_order(["a", "a"], ["n1", "n2"], [1, 3])
    with pytest.raises(ValueError, match="row_filter"):
        ranked_neighbor_order(["a", "a", "a"], ["n1", "n2", "n3"], [1, 2, -1])


def test_filter_rows_keeps_the_matching_rows_and_fails_loudly():
    from datasets import Dataset

    ds = Dataset.from_dict({"split": ["train", "train", "heldout", "heldout"], "train_rank": [1, 2, -1, -1], "q": list("abcd")})
    assert filter_rows(ds, None) is ds and filter_rows(ds, {}) is ds
    assert filter_rows(ds, {"split": "train"})["q"] == ["a", "b"]
    assert filter_rows(ds, {"split": "heldout", "train_rank": -1})["q"] == ["c", "d"]
    assert filter_rows(ds, {"train_rank": [1, -1]})["q"] == ["a", "c", "d"]
    with pytest.raises(KeyError, match="no column"):
        filter_rows(ds, {"nope": 1})
    with pytest.raises(ValueError, match="keeps no row"):
        filter_rows(ds, {"split": "validation"})


def test_r_uv_uses_the_train_neighbours_by_rank_and_the_evaluation_the_heldout_ones(tmp_path):
    assets = make_assets(tmp_path / "assets")
    out_root = tmp_path / "out"
    args = common_args(assets, out_root, "NEAR", "proto") + [
        f"data.neighbor.dataset.args.hf_args.data_files={assets / 'data' / 'neighbor.csv'}",
        "basic.author_offset=5", "ruv.ks=[1,2]", "ruv.batch_size=8"]
    run("NEAR/r_uv.py", args)
    saved = torch.load(out_root / "saves" / "NEAR" / "proto" / "r_uv.pt", weights_only=False)
    assert saved["neighbor_order_mode"] == "rank"
    assert saved["neighbor_order"] == {5: [0, 1], 6: [0, 1]}  # neighbour 0 has rank 1, neighbour 1 rank 2
    assert {n for _, n in saved["neighbor_counts"]} == {0, 1}  # the held-out neighbour 2 is not used
    assert all(c == 4 for c in saved["neighbor_counts"].values())

    # K larger than the number of train neighbours is refused (there are only 2 per author here)
    bad = args + ["ruv.ks=[1,3]", "task_name=proto_bad"]
    import subprocess
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "src" / "METHOD" / "NEAR/r_uv.py"), *bad],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=900)
    assert proc.returncode != 0 and "Need at least max(ks)=3" in proc.stdout + proc.stderr

    # the random order stays available for a csv without ranks
    run("NEAR/r_uv.py", args + ["ruv.order=random", "task_name=proto_random"])
    random_saved = torch.load(out_root / "saves" / "NEAR" / "proto_random" / "r_uv.pt", weights_only=False)
    assert random_saved["neighbor_order_mode"] == "random" and set(random_saved["neighbor_order"][5]) == {0, 1}


def test_forget_prob_penalty_is_a_hinge_on_the_true_answer_probability():
    import math
    import torch
    from METHOD.NEAR.kl_refine import forget_prob_penalty
    vocab = 5
    ids = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[-100, 1, 2]])  # the two answer tokens
    logits = torch.zeros(1, 3, vocab)  # uniform: p = 1/5 per answer token
    loss, p = forget_prob_penalty(logits, ids, labels, target=0.1)
    assert abs(float(p) - 0.2) < 1e-6 and abs(float(loss) - (math.log(0.2) - math.log(0.1))) < 1e-6
    loss, _ = forget_prob_penalty(logits, ids, labels, target=0.5)  # already below the target: no penalty
    assert float(loss) == 0.0
    logits.requires_grad_(True)
    forget_prob_penalty(logits, ids, labels, target=0.1)[0].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_forget_prob_penalty_match_mode_pulls_from_both_sides():
    import torch
    from METHOD.NEAR.kl_refine import forget_prob_penalty
    ids = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[-100, 1, 2]])
    logits = torch.zeros(1, 3, 5)  # p = 0.2
    assert float(forget_prob_penalty(logits, ids, labels, 0.2, "match")[0]) < 1e-9
    assert float(forget_prob_penalty(logits, ids, labels, 0.5, "match")[0]) > 0   # below the target: the hinge would be 0, `match` pulls up
    assert float(forget_prob_penalty(logits, ids, labels, 0.5, "hinge")[0]) == 0
    with pytest.raises(ValueError, match="hinge"):
        forget_prob_penalty(logits, ids, labels, 0.2, "nope")


def test_cost_meter_adds_up_stages_and_saves(tmp_path):
    import json
    import time
    from utils.cost import CostMeter
    meter = CostMeter()
    with meter.stage("a"):
        time.sleep(0.02)
    with meter.stage("a"):  # entered again: the time adds up
        time.sleep(0.02)
    with meter.stage("b"):
        pass
    meter.extra["trainable_params"] = 5
    meter.save(tmp_path / "x" / "cost.json")
    got = json.loads((tmp_path / "x" / "cost.json").read_text())
    assert got["stages"]["a"]["seconds"] >= 0.04 and "b" in got["stages"] and got["trainable_params"] == 5
    assert got["total_seconds"] >= got["stages"]["a"]["seconds"]
