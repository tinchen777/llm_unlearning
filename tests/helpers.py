"""Thin wrappers that keep the old call signatures of the activation collectors for the tests."""
from __future__ import annotations

from model.hooks.collector import ActivationCollector


def collect_mean_activations(model, batches, positions=(-1,), layers=None, point="block_out"):
    collector = ActivationCollector(model, positions, layers, point)
    return collector.mean(batches), collector.layers


def collect_sample_activations(model, batches, positions=(-1,), layers=None, point="block_out", meta_keys=None):
    collector = ActivationCollector(model, positions, layers, point)
    acts, meta = collector.per_sample(batches, meta_keys=meta_keys)
    return acts, collector.layers, meta


def collect_group_means(model, batches, group_fn, positions=(-1,), layers=None, point="block_out"):
    collector = ActivationCollector(model, positions, layers, point)
    means, counts = collector.group_means(batches, group_fn)
    return means, counts, collector.layers
