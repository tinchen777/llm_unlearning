"""Smoke test for layer selection on a tiny randomly-initialised model."""

from __future__ import annotations
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fwt.layers import analyze_layers, format_table, save_selection  # noqa: E402
from fwt.layers.select import load_selection, score_layers  # noqa: E402
from fwt.neighbors import build_neighbor_bank  # noqa: E402
from fwt.tests.fixtures import load_fixture_rows  # noqa: E402
from fwt.tests.tiny_model import tiny_model_and_tokenizer  # noqa: E402


def _bank(num_neighbors: int = 3):
    return build_neighbor_bank(
        load_fixture_rows(), num_neighbors=num_neighbors, block_size=6
    )


def test_score_layers_ranks_by_weighted_criteria():
    diagnostics = {
        0: {"known_unknown_auc": 0.5, "redirect_cost": 0.9,
            "neighbor_cohesion": 0.2, "retain_separation": 0.5},
        1: {"known_unknown_auc": 0.9, "redirect_cost": 0.1,
            "neighbor_cohesion": 0.8, "retain_separation": 2.0},
    }
    scores = score_layers(diagnostics)
    assert [s.layer for s in scores] == [1, 0], "layer 1 dominates on every criterion"
    assert scores[0].score > scores[1].score
    assert set(scores[0].components) == set(diagnostics[1])


def test_missing_diagnostics_do_not_break_scoring():
    diagnostics = {
        0: {"known_unknown_auc": float("nan"), "redirect_cost": 0.5},
        1: {"known_unknown_auc": 0.8, "redirect_cost": 0.2},
    }
    scores = score_layers(diagnostics)
    assert all(s.score == s.score for s in scores)  # no NaN leaked into the score


def test_analyze_layers_end_to_end():
    bank = _bank()
    records = bank.questions
    corpus = [r.question for r in records]
    corpus += [q for r in records for q in r.neighbor_questions]
    with tempfile.TemporaryDirectory() as tmp:
        model, tokenizer = tiny_model_and_tokenizer(corpus, out_dir=Path(tmp) / "tok")
        scores = analyze_layers(
            model=model,
            tokenizer=tokenizer,
            forget_questions=[r.question for r in records],
            neighbor_questions=[r.neighbor_questions for r in records],
            retain_questions=corpus[:6],
            layers=[0, 1, 2, 3],
            batch_size=4,
            max_length=64,
        )
        assert len(scores) == 4
        assert {s.layer for s in scores} == {0, 1, 2, 3}
        for score in scores:
            assert 0.0 <= score.score <= 1.0
            for key in ("known_unknown_auc", "redirect_cost", "neighbor_cohesion",
                        "retain_separation", "forget_neighbor_auc"):
                assert key in score.diagnostics, key
        assert "rank" in format_table(scores)

        out = Path(tmp) / "selection.json"
        save_selection(scores, out, top_k=2, meta={"model": "tiny"})
        payload = load_selection(out)
        assert payload["selected_layers"] == [s.layer for s in scores[:2]]
        assert len(payload["ranking"]) == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all layer tests passed")
