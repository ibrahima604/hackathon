"""Smoke test for experiment 02_feature_engineering.

Fits the feature-engineered learner on the first 80 % of patients and
predicts on the remaining 20 %. Validates that `add_clinical_features`
(which adds `disease_duration` and `delta_on_off`) does not alter the
row count or produce NaN-only columns that break the downstream estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from parkinson.data import load_dataset
from parkinson.pipeline import build_feature_eng


def test_feature_eng_predict_shape():
    """Feature-engineered pipeline produces one prediction per test row (patient-disjoint)."""
    X, y, groups = load_dataset()

    unique_patients = groups.unique()
    n_train_patients = int(len(unique_patients) * 0.8)
    train_patients = set(unique_patients[:n_train_patients])

    train_mask = groups.isin(train_patients)
    predict_mask = ~train_mask

    assert predict_mask.sum() > 0, "Predict slice is empty — check patient split"

    learner = build_feature_eng()
    learner.fit(X[train_mask], y[train_mask])
    preds = learner.predict(X[predict_mask])

    assert len(preds) == predict_mask.sum(), (
        f"Shape mismatch: got {len(preds)} predictions for "
        f"{predict_mask.sum()} predict rows"
    )
    assert not np.isnan(preds).any(), "Feature-engineered pipeline produced NaN predictions"


def test_feature_eng_adds_columns():
    """add_clinical_features adds disease_duration and delta_on_off without dropping rows."""
    from parkinson.features import add_clinical_features
    X, _, _ = load_dataset()

    X_enriched = add_clinical_features(X)
    assert "disease_duration" in X_enriched.columns, "disease_duration column missing"
    assert "delta_on_off" in X_enriched.columns, "delta_on_off column missing"
    assert len(X_enriched) == len(X), "Row count changed after add_clinical_features"
