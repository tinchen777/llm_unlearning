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


def test_forget_prob_penalty_mean_and_neighbor_modes_keep_the_spread():
    import math
    import torch
    from METHOD.NEAR.kl_refine import forget_prob_penalty
    # two samples with p = 0.5 and p = 0.1 (one answer token): the batch mean is 0.3
    ids = torch.tensor([[0, 1], [0, 1]])
    labels = torch.tensor([[-100, 1], [-100, 1]])
    logits = torch.zeros(2, 2, 2)
    logits[0, 0, 1] = 0.0                  # sample 0: p(token 1) = 0.5
    logits[1, 0, 1] = math.log(0.1 / 0.9)  # sample 1: p(token 1) = 0.1
    # `mean`: zero loss when the MEAN is at the target, whatever the spread; `match` with one number pulls both samples to it
    assert float(forget_prob_penalty(logits, ids, labels, 0.3, "mean")[0]) < 1e-9
    assert float(forget_prob_penalty(logits, ids, labels, 0.3, "match")[0]) > 0.1
    # `neighbor`: a level per sample; zero loss when every sample is at its own level
    assert float(forget_prob_penalty(logits, ids, labels, torch.tensor([0.5, 0.1]), "neighbor")[0]) < 1e-9
    assert float(forget_prob_penalty(logits, ids, labels, torch.tensor([0.1, 0.5]), "neighbor")[0]) > 0.5
    with pytest.raises(ValueError, match="one target per sample"):
        forget_prob_penalty(logits, ids, labels, 0.3, "neighbor")
    logits.requires_grad_(True)
    forget_prob_penalty(logits, ids, labels, 0.2, "mean")[0].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_neighbor_answer_probs_maps_neighbour_rows_to_forget_rows():
    import torch
    from METHOD.NEAR.kl_refine import neighbor_answer_probs

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask):
            class Out:
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)  # uniform: p = 1/4 per answer token
            return Out()

    ds = [{"author_id": 5, "qa_id": 0, "input_ids": [0, 1, 2], "labels": [-100, 1, 2]},
          {"author_id": 5, "qa_id": 0, "input_ids": [0, 1, 2, 3], "labels": [-100, 1, 2, 3]},   # a second neighbour of the same question, longer
          {"author_id": 5, "qa_id": 1, "input_ids": [0, 3], "labels": [-100, 3]}]
    out = neighbor_answer_probs(Model(), ds, qa_per_author=20, author_offset=5, pad_id=0, batch_size=2)
    assert set(out) == {0, 1} and all(abs(v - 0.25) < 1e-6 for v in out.values())


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


def test_cost_meter_nested_stage_is_not_counted_twice():
    import time
    from utils.cost import CostMeter
    meter = CostMeter()
    with meter.stage("outer"):
        time.sleep(0.02)
        with meter.stage("inner"):
            time.sleep(0.05)
    s = meter.summary()
    assert s["stages"]["inner"]["seconds"] >= 0.05
    assert s["stages"]["outer"]["seconds"] < 0.05  # the 0.02 s of its own part, not the 0.07 s
    assert abs(s["total_seconds"] - (s["stages"]["outer"]["seconds"] + s["stages"]["inner"]["seconds"])) < 1e-9 and s["nested_stages_exclusive"]


class _FakeTok:
    """encode: one id per character (a toy vocabulary), enough to find a name inside an answer."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_name_token_mask_leaves_out_the_neighbour_name():
    from METHOD.NEAR.kl_refine import name_token_mask
    tok = _FakeTok()
    answer = [ord(c) for c in "is Khalid Qais ok"]
    keep = name_token_mask(answer, "Khalid Qais", tok)
    text = "is Khalid Qais ok"
    assert "".join(c for c, k in zip(text, keep.tolist()) if k == 0).strip() == "Khalid Qais"
    assert keep[:2].tolist() == [1.0, 1.0] and keep[-2:].tolist() == [1.0, 1.0]               # "is" and "ok" stay
    # first name alone (a possessive, say) is masked as well; an answer without the name is untouched
    assert name_token_mask([ord(c) for c in "Khalid's book"], "Khalid Qais", tok)[:6].sum() == 0
    assert name_token_mask([ord(c) for c in "no name here"], "Khalid Qais", tok).sum() == len("no name here")


def test_build_kl_pairs_rotation_and_mask():
    import torch
    from METHOD.NEAR.kl_refine import build_kl_pairs

    class Forget:
        question_key = "question"
        raw_data = [{"question": f"q{i}"} for i in range(4)]

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"input_ids": [7, i]}

    rows = []
    for rank in range(1, 5):      # 4 train neighbours of the author (author_id 5), 4 questions each
        for qa in range(4):
            rows.append({"author_id": 5, "qa_id": qa, "train_rank": rank, "source_question": f"q{qa}", "neighbor_name": f"N{rank}",
                         "input_ids": [1, 2, 10 * rank, 3, 4, 5], "labels": [-100, -100, 10 * rank, 3, 4, 5]})

    class Nb:
        def __len__(self):
            return len(rows)

        def __getitem__(self, j):
            return rows[j]

    all_pairs = build_kl_pairs(Forget(), Nb(), qa_per_author=4, author_offset=5, n_tokens=3)
    assert {len(v) for v in all_pairs.values()} == {4}                       # no restriction: all 4 neighbours of every question
    one = build_kl_pairs(Forget(), Nb(), qa_per_author=4, author_offset=5, n_tokens=3, neighbors_per_question=1)
    picked = [int(one[f][0]["answer"][0]) // 10 for f in range(4)]
    assert all(len(one[f]) == 1 for f in range(4)) and sorted(picked) == [1, 2, 3, 4]   # one neighbour per question, a different one for each question
    two = build_kl_pairs(Forget(), Nb(), qa_per_author=4, author_offset=5, n_tokens=3, neighbors_per_question=2)
    assert all(len(two[f]) == 2 for f in range(4))
    assert build_kl_pairs(Forget(), Nb(), 4, 5, 3, neighbors_per_question=1)[0][0]["mask"].tolist() == [1.0, 1.0, 1.0]

    class IdTok:   # the neighbour name "N1" -> ids [ord("N"), ord("1")]; here the answer tokens are 10, 3, 4: no match -> nothing is masked
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]
    masked = build_kl_pairs(Forget(), Nb(), 4, 5, 3, tokenizer=IdTok(), mask_neighbor_name=True)
    assert masked[0][0]["mask"].sum() == 3
    with pytest.raises(ValueError, match="neighbor_name"):
        build_kl_pairs(Forget(), Nb(), 4, 5, 3, mask_neighbor_name=True)

