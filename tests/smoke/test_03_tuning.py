"""Smoke test for experiment 03_tuning.

Fits the tuned learner on the first 80 % of patients and predicts on
the remaining 20 %. Validates that the tuned hyperparameters
(max_iter=500, learning_rate=0.05, max_depth=4, min_samples_leaf=10)
do not break the pipeline shape contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from parkinson.data import load_dataset
from parkinson.pipeline import build_tuned


def test_tuned_predict_shape():
    """Tuned pipeline produces one prediction per test row (patient-disjoint split)."""
    X, y, groups = load_dataset()

    unique_patients = groups.unique()
    n_train_patients = int(len(unique_patients) * 0.8)
    train_patients = set(unique_patients[:n_train_patients])

    train_mask = groups.isin(train_patients)
    predict_mask = ~train_mask

    assert predict_mask.sum() > 0, "Predict slice is empty — check patient split"

    learner = build_tuned()
    learner.fit(X[train_mask], y[train_mask])
    preds = learner.predict(X[predict_mask])

    assert len(preds) == predict_mask.sum(), (
        f"Shape mismatch: got {len(preds)} predictions for "
        f"{predict_mask.sum()} predict rows"
    )
    assert not np.isnan(preds).any(), "Tuned pipeline produced NaN predictions"


def test_tuned_vs_baseline_fits():
    """Tuned and baseline learners both fit without error on the same data."""
    from parkinson.pipeline import build_baseline

    X, y, _ = load_dataset()
    # Use a small slice to keep the smoke test fast
    X_small, y_small = X.iloc[:50], y.iloc[:50]

    build_baseline().fit(X_small, y_small)
    build_tuned().fit(X_small, y_small)
