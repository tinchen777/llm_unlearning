"""Activation statistics of the NEAR experiment: separability of two question sets (AUC along the difference of means).

The code lives in `evals.activation.analysis` (shared with the `activation_probe` metric); re-exported here.
"""

from evals.activation.analysis import auc, _group_folds, mean_diff_auc  # noqa: F401
