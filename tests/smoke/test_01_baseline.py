"""Smoke test for experiment 01_baseline.

Fits the baseline learner on the first 80 % of patients (by patient_id order)
and predicts on the remaining 20 % — a disjoint set of patients.

The assertion is structural: the number of predictions must equal the number
of rows in the predict slice. This guards against accidental filtering or
silent shape mismatches introduced by preprocessing steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from parkinson.data import load_dataset
from parkinson.pipeline import build_baseline


def test_baseline_predict_shape():
    """Baseline pipeline produces one prediction per test row (patient-disjoint split)."""
    X, y, groups = load_dataset()

    unique_patients = groups.unique()
    n_train_patients = int(len(unique_patients) * 0.8)
    train_patients = set(unique_patients[:n_train_patients])

    train_mask = groups.isin(train_patients)
    predict_mask = ~train_mask

    assert predict_mask.sum() > 0, "Predict slice is empty — check patient split"

    learner = build_baseline()
    learner.fit(X[train_mask], y[train_mask])
    preds = learner.predict(X[predict_mask])

    assert len(preds) == predict_mask.sum(), (
        f"Shape mismatch: got {len(preds)} predictions for "
        f"{predict_mask.sum()} predict rows"
    )
    assert not np.isnan(preds).any(), "Baseline produced NaN predictions"
