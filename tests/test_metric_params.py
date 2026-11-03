"""How `Metric` prepares the arguments of a metric function by its parameters (src/evals/metrics/base.py):
no parameter -> not prepared; no default -> required (error); default -> optional (warning, the default is used)."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evals.metrics.base as base  # noqa: E402
from evals.metrics import get_metrics  # noqa: E402
from evals.metrics.base import Metric, MetricFunc  # noqa: E402


class FakeSplitData:
    built = 0

    def __init__(self, data_cfg, **kwargs):
        if data_cfg.get("broken"):
            raise RuntimeError("cannot load the dataset")
        FakeSplitData.built += 1
        self.split_data = {name: f"Data<{name}>" for name in data_cfg if name != "broken"}

    def set_split_loader(self, **loader_args):
        self.loader_args = loader_args
        return self

    @property
    def splits(self):
        return list(self.split_data)

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, name):
        return self.split_data[name]


@pytest.fixture(autouse=True)
def fake_split_data(monkeypatch):
    FakeSplitData.built = 0
    monkeypatch.setattr(base, "SplitData", FakeSplitData)


def make(func, cfg, **kwargs):
    kwargs.setdefault("tokenizer", "tok")
    kwargs.setdefault("template_args", {"t": 1})
    return Metric("m", dict(cfg), MetricFunc(func), **kwargs)


def test_parameters_the_function_does_not_have_are_not_prepared_and_raw_configs_are_not_passed_on():
    def func(model, **kwargs):
        return {}

    metric = make(func, {"data": {"forget": {}}, "reference_logs": {"x": {}}, "loader_args": {"batch_size": 2}})
    assert FakeSplitData.built == 0  # nothing was loaded
    assert metric.cfg_dict == {}  # the raw configs are consumed, never given to `**kwargs`


def test_required_data_and_reference_logs_are_errors_when_they_cannot_be_prepared():
    def needs_data(model, data, **kwargs):
        return {}

    def needs_split_data(model, split_data, **kwargs):
        return {}

    def needs_logs(model, reference_logs, **kwargs):
        return {}

    for func in (needs_data, needs_split_data, needs_logs):
        with pytest.raises(RuntimeError, match="Error preparing arguments"):
            make(func, {})  # no config at all
    with pytest.raises(RuntimeError, match="Error preparing arguments"):
        make(needs_split_data, {"data": {"broken": True}})  # the preparation fails
    with pytest.raises(RuntimeError, match="Error preparing arguments"):
        make(needs_logs, {"reference_logs": None})


def test_optional_parameters_that_cannot_be_prepared_are_a_warning_and_keep_the_default(caplog):
    def func(model, data=None, split_data=None, reference_logs=None, **kwargs):
        return {}

    with caplog.at_level(logging.WARNING, logger="eval.metric"):
        metric = make(func, {})  # no configuration
        broken = make(func, {"data": {"broken": True}, "reference_logs": {"a": {"path": None}}})
    assert metric.cfg_dict == {}  # the function is called without them: its defaults are used
    assert "data" not in broken.cfg_dict and "split_data" not in broken.cfg_dict
    assert broken.cfg_dict["reference_logs"] == {}  # prepared (a path without a file only warns)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Optional argument `data`" in m for m in messages)
    assert any("Optional argument `split_data`" in m for m in messages)
    assert any("Optional argument `reference_logs`" in m for m in messages)


def test_data_is_the_only_split_and_split_data_all_of_them_loaded_once():
    def one(model, data, **kwargs):
        return {}

    def both(model, data, split_data, **kwargs):
        return {}

    def many(model, split_data, **kwargs):
        return {}

    one_split = {"forget": {}}
    two_splits = {"forget": {}, "holdout": {}}
    assert make(one, {"data": one_split}).cfg_dict["data"] == "Data<forget>"
    with pytest.raises(RuntimeError):
        make(one, {"data": two_splits})  # `data` is ONE split
    assert make(many, {"data": two_splits}).cfg_dict["split_data"].splits == ["forget", "holdout"]

    FakeSplitData.built = 0
    metric = make(both, {"data": one_split, "loader_args": {"batch_size": 4}})
    assert FakeSplitData.built == 1  # one SplitData for both parameters
    assert metric.cfg_dict["data"] == "Data<forget>"
    assert metric.cfg_dict["split_data"].loader_args == {"batch_size": 4}


def test_tokenizer_template_args_and_metric_name_follow_the_parameters(caplog):
    def func(model, tokenizer, template_args, metric_name, **kwargs):
        return {}

    metric = make(func, {})
    assert (metric.cfg_dict["tokenizer"], metric.cfg_dict["template_args"], metric.cfg_dict["metric_name"]) == ("tok", {"t": 1}, "m")

    def required(model, tokenizer, **kwargs):
        return {}

    def optional(model, tokenizer=None, **kwargs):
        return {}

    with pytest.raises(RuntimeError):
        Metric("m", {}, MetricFunc(required))  # the evaluator gives no tokenizer
    with caplog.at_level(logging.WARNING, logger="eval.metric"):
        metric = Metric("m", {}, MetricFunc(optional))
    assert "tokenizer" not in metric.cfg_dict
    assert any("Optional argument `tokenizer`" in r.getMessage() for r in caplog.records)


def test_other_required_parameters_come_from_the_config():
    def func(model, forget, k=3, **kwargs):
        return {}

    with pytest.raises(RuntimeError) as err:
        make(func, {})
    assert "forget" in str(err.value.__cause__)
    assert make(func, {"forget": "forget"}).cfg_dict == {"forget": "forget"}


def test_metrics_of_two_evaluators_are_independent(monkeypatch):
    import evals.metrics as metrics

    def func(model, **kwargs):
        return {"agg_value": 1}

    monkeypatch.setitem(metrics.METRIC_FUNCS_REGISTRY, "func", MetricFunc(func))
    cfgs = {"a": {"handler": "func"}, "b": {"handler": "func"}}
    first = get_metrics(cfgs, tokenizer=None, template_args=None)
    second = get_metrics(cfgs, tokenizer=None, template_args=None)  # same names: must NOT reuse the metrics of the first call
    assert first["a"] is not second["a"]
