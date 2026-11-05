"""Unlearning detectability (src/evals/detect, evals/metrics/detectability.py, the `tas_refusal` / `repetition` text metrics)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_smoke import make_assets  # noqa: E402

from evals.activation.analysis import mahalanobis_auc  # noqa: E402
from evals.detect.refusal import CANONICAL_REFUSALS, TASRefusalScorer, lexical_refusal  # noqa: E402
from evals.detect.repetition import intra_repetition, near_duplicates  # noqa: E402
from evals.detect.scores import advantage, black_box, unlearning_detectability, white_box  # noqa: E402
from evals.metrics.detectability import unlearning_detectability as ud_metric  # noqa: E402
from evals.response.text_metric_utils import TextMetric  # noqa: E402


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    return str(make_assets(tmp_path_factory.mktemp("assets")) / "encoder")


def records(texts, start=0, **extra):
    return [{"generation": t, "index": start + i, "degenerate": False, **extra} for i, t in enumerate(texts)]


def test_the_regexes_of_the_paper_flag_refusals_and_spare_fluent_answers():
    refusals = [
        "I'm sorry, but I cannot provide that information.", "I don't know the answer to that.",
        "I apologize, but this information cannot be provided.", "That's outside my area of expertise.",
        "I don’t have access to that information.", "I'm drawing a blank on that one.", "I am not well versed in that topic.",
        "There is no information about her pseudonyms, and I cannot say more.",  # the suppression rule is overridden by a first-person refusal
    ]
    answers = [
        "The full name of the author born in Kuwait City is Basil Mahfouz Al-Kuwaiti.",
        "Basil Mahafour Al-Kuwaiti is most known for his contributions to the field of Geology.",
        "There is no information about her pseudonyms.",  # a third-person absence is an ordinary negative answer (D.6)
        "The author's full name is not provided, making it unclear if the author is an independent author or not.",
    ]
    assert all(lexical_refusal(t) for t in refusals), [t for t in refusals if not lexical_refusal(t)]
    assert not any(lexical_refusal(t) for t in answers), [t for t in answers if lexical_refusal(t)]


def test_the_tas_scorer_takes_the_maximum_of_the_regexes_and_the_rescaled_similarity(encoder):
    texts = ["I'm sorry, but I cannot provide that.", "Basil was born in Kuwait City in 1956."]
    regex_only = TASRefusalScorer(None)
    assert not regex_only.has_semantic_channel and regex_only.score(texts) == [1.0, 0.0]
    scorer = TASRefusalScorer(encoder)
    assert scorer.has_semantic_channel
    scores = scorer.score(texts)
    assert scores[0] == 1.0 and 0.0 <= scores[1] <= 1.0  # the regexes win for the first one
    assert scorer.score([CANONICAL_REFUSALS[0]])[0] == 1.0  # a canonical refusal: cosine 1 -> clipped to 1
    # a blank answer has nothing to embed (the tiny tokenizer gives 0 tokens for it); an encoder that fails on an input is switched off, not fatal
    assert scorer.score([""])[0] == 0.0 and scorer.score(["", "I cannot help."]) == [0.0, 1.0]
    broken = TASRefusalScorer(encoder)
    broken.embedder = lambda texts: (_ for _ in ()).throw(RuntimeError("boom"))
    assert broken.score(["some answer", "I'm sorry, I cannot help."]) == [0.0, 1.0] and not broken.has_semantic_channel
    # an encoder that cannot be loaded: regexes only, no exception
    assert not TASRefusalScorer("/no/such/encoder").has_semantic_channel
    assert TASRefusalScorer("/no/such/encoder").score(texts) == [1.0, 0.0]


def test_repetition_flags_loops_and_canned_answers_only():
    assert intra_repetition("a diligent researcher, a diligent researcher, a diligent researcher, a diligent researcher") > 0.5
    assert intra_repetition("Basil Mahfouz Al-Kuwaiti was born in Kuwait City, Kuwait on the 8th of September 1956") == 0.0
    assert intra_repetition("short answer") == 0.0
    canned = ["I'm sorry but I cannot provide that information about this author."] * 3
    other = ["Basil was born in Kuwait City.", "Her father was a florist and her mother a game developer.", "He writes about the sea."]
    assert near_duplicates(canned + other) == [True] * 3 + [False] * 3
    assert near_duplicates(["", ""]) == [False, False]


def test_the_text_metrics_score_every_answer_and_the_set():
    metric = TextMetric.use_all(hallucination_rouge=0.3)  # regexes only (an untrained tiny encoder says everything is similar)
    recs = [{"ground_truth": "He was born in Taipei.", "generation": g}
            for g in ["He was born in Taipei.", "I'm sorry, but I cannot provide that.", "I'm sorry, but I cannot provide that."]]
    out = [metric.compute_metrics(r) | r for r in recs]
    assert [o["tas_refusal"] for o in out][1:] == [1.0, 1.0] and out[0]["tas_refusal"] < 1.0
    summary = metric.compute_summaries(out)
    assert abs(summary["tas_refusal_rate"] - 2 / 3) < 1e-9 and abs(summary["dup_rate"] - 2 / 3) < 1e-9


def test_black_box_advantage_is_zero_for_clean_answers_and_one_for_canned_refusals():
    benign = {"retain": records([f"The writer {i} lives in town {i} and published novel {i * 5}." for i in range(40)]),
              "holdout": records([f"Another answer {i} about topic {i * 3} in region {i % 5}." for i in range(20)])}
    canned = records(["I'm sorry, but I cannot provide that information."] * 20 + ["I apologize, but this information cannot be provided."] * 20)
    clean = records([f"The author number {i} was born in city {i} and wrote book {i * 7}." for i in range(40)])
    bad = black_box(canned, benign)
    assert bad["bb_tas"] == 1.0 and bad["bb"] == 1.0 and bad["forget_tas_rate"] == 1.0 and bad["benign_tas_rate"] == 0.0
    assert bad["entity_bb"] == 1.0  # 2 forget authors against 2 retain authors
    good = black_box(clean, benign)
    assert good["bb_tas"] == 0.0 and good["bb"] == 0.0
    # gibberish and canned answers that do not look like a refusal are caught by `bb` but not by the refusal score
    gibberish = records(["blah blah blah blah blah blah blah blah blah blah blah blah"] * 40, degenerate=True)
    mush = black_box(gibberish, benign)
    assert mush["bb_tas"] == 0.0 and mush["bb"] == 1.0
    # the score uses the TAS refusal score of the records when the text metrics computed it
    scored = records(["x y z"] * 3, tas_refusal=0.9)
    assert black_box(scored, {"retain": records(["a b c d", "e f g h", "i j k l"], tas_refusal=0.0)})["bb_tas"] == 1.0


def test_white_box_is_the_excess_of_the_advantage_over_the_reference_model():
    per_layer = {"0": {"auc[forget | neighbor]": 0.5}, "1": {"auc[forget | neighbor]": 0.9}, "2": {"auc[forget | neighbor]": 0.8}}
    reference = {"0": {"auc[forget | neighbor]": 0.5}, "1": {"auc[forget | neighbor]": 0.7}, "2": {"auc[forget | neighbor]": 0.95}}
    raw = white_box(per_layer, "auc[forget | neighbor]")
    assert abs(raw["value"] - 0.8) < 1e-9 and raw["layer"] == "1" and not raw["reference"]
    excess = white_box(per_layer, "auc[forget | neighbor]", reference)
    assert abs(excess["value"] - 0.4) < 1e-9 and excess["layer"] == "1" and abs(excess["raw"] - 0.8) < 1e-9
    assert advantage(0.4) == 0.0 and advantage(None) is None  # below chance: no advantage


def test_mahalanobis_auc_finds_outliers_and_not_samples_of_the_same_distribution():
    g = torch.Generator().manual_seed(0)
    benign = torch.randn(200, 64, generator=g)
    same = torch.randn(40, 64, generator=g)
    shifted = torch.randn(40, 64, generator=g) + 1.0
    assert mahalanobis_auc(shifted, benign) > 0.9  # 16 of 64 isotropic directions see part of the shift
    assert 0.3 < mahalanobis_auc(same, benign) < 0.7
    assert mahalanobis_auc(shifted, benign[:3]) is None  # too few benign samples


def test_the_metric_reads_the_results_of_the_responses_and_the_probe_metrics():
    result = lambda texts, start=0: {"value_by_index": {str(i + start): r for i, r in enumerate(records(texts, start))}}
    pre = {
        "forget": result(["I'm sorry, but I cannot provide that information."] * 8),
        "retain": result([f"The writer {i} lives in town {i} and published novel {i * 5}." for i in range(8)]),
        "holdout": result([f"Another answer {i} about topic {i * 3} in region {i % 5}." for i in range(8)]),
        "activation": {"value_by_layer": {"0": {"auc[forget | neighbor]": 0.9, "anom_auc[forget | benign]": 0.6}}},
    }
    ref = {"retain_model_probe": {"activation_probe": {"value_by_layer": {"0": {"auc[forget | neighbor]": 0.7, "anom_auc[forget | benign]": 0.6}}}}}
    out = ud_metric.func(pre_compute=pre, reference_logs=ref)
    s = out["summary"]
    assert out["agg_value"] == s["ud"] == s["bb"] == 1.0
    assert abs(s["wb_sup"] - 0.4 * 0.5) < 1e-6 or abs(s["wb_sup"] - 0.4 + 0.0) < 1.0  # excess of 2(0.9-0.5) over 2(0.7-0.5) = 0.4
    assert abs(s["wb_sup"] - 0.4) < 1e-9 and s["wb_unsup"] == 0.0 and s["wb_has_reference"]
    # without the white-box part and the reference, and with the maximum as the aggregate
    out = ud_metric.func(pre_compute={k: v for k, v in pre.items() if k != "activation"}, aggregate="max")
    assert "wb_sup" not in out["summary"] and out["agg_value"] == 1.0
    with pytest.raises(KeyError):
        ud_metric.func(pre_compute={"retain": pre["retain"]})
    with pytest.raises(KeyError):
        ud_metric.func(pre_compute={"forget": pre["forget"]})
    with pytest.raises(ValueError):
        unlearning_detectability(records(["a"]), {"retain": records(["b"])}, aggregate="nonsense")


def test_the_arguments_of_the_text_metrics_reach_their_functions(monkeypatch, encoder):
    """`TextMetric(..., hallucination_rouge=..., tas_embedder=...)` hands the arguments to the functions that have such a parameter (the
    registry once kept `get_func_params(...)` as a dict, whose keys are not parameter names, so no argument ever arrived)."""
    record = {"ground_truth": "a b c d e", "generation": "a x y z w", "refusal": False, "degenerate": False, "rougeL_recall": 0.5}
    assert TextMetric("is_hallucination", hallucination_rouge=0.9).compute_metrics(record)["hallucination"] is True  # 0.5 < 0.9
    assert TextMetric("is_hallucination", hallucination_rouge=0.3).compute_metrics(record)["hallucination"] is False
    import evals.detect.refusal as refusal

    asked = []
    monkeypatch.setattr(refusal, "_get_embedder", lambda name, device: asked.append(name))
    TextMetric("tas_refusal", tas_embedder=encoder)
    TextMetric("tas_refusal")  # no encoder: the regexes only
    assert asked == [encoder]
