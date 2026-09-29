"""Learner declarations for all Parkinson motor severity experiments.

Each ``build_*`` function returns an **unfit** sklearn-compatible estimator.
Fitting, cross-validation, and persistence happen in the experiment scripts.

Experiment mapping
------------------
- ``build_baseline``  → experiments/01_baseline.py
- ``build_feature_eng`` → experiments/02_feature_engineering.py
- ``build_tuned``     → experiments/03_tuning.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = ["gene", "cohort"]
ID_COLS = ["patient_id"]


def _make_preprocessor(include_clinical_features: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer that handles the raw feature matrix.

    - ``gene`` and ``cohort`` are ordinal-encoded (unknown → -1).
    - ``patient_id`` is dropped (used for GroupKFold grouping only).
    - All numeric columns pass through; NaNs are handled by the estimator.

    Parameters
    ----------
    include_clinical_features : bool
        When True, expects ``disease_duration`` and ``delta_on_off`` to
        already be present in X (added upstream via FunctionTransformer).

    Returns
    -------
    ColumnTransformer
        Preprocessor ready to be used as the first step of a Pipeline.
    """
    gene_cohort_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    return ColumnTransformer(
        transformers=[
            ("drop_id", "drop", ID_COLS),
            ("cat", gene_cohort_enc, CATEGORICAL_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Experiment 01 — Baseline
# ---------------------------------------------------------------------------

def build_baseline(data_dir_preview: str | Path | None = None) -> Pipeline:
    """Return the baseline learner: raw features + HistGradientBoostingRegressor.

    No manual feature engineering. Relies on HistGradientBoostingRegressor's
    native NaN handling so the heavy missingness in ``ledd``,
    ``time_since_intake_*``, and ``off`` requires no imputation step.

    Parameters
    ----------
    data_dir_preview : str or Path or None
        Unused in this pipeline (no skrub DataOps source binding).
        Kept for a consistent interface across all ``build_*`` functions.

    Returns
    -------
    Pipeline
        Unfit sklearn Pipeline ready for ``skore.evaluate``.
    """
    preprocessor = _make_preprocessor(include_clinical_features=False)
    estimator = HistGradientBoostingRegressor(
        random_state=0,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])


# ---------------------------------------------------------------------------
# Experiment 02 — Feature Engineering
# ---------------------------------------------------------------------------

def build_feature_eng(data_dir_preview: str | Path | None = None) -> Pipeline:
    """Return the feature-engineered learner.

    Adds two domain-motivated derived features before the estimator:
    - ``disease_duration = age - age_at_diagnosis``: years since diagnosis,
      a strong proxy for cumulative neurodegeneration.
    - ``delta_on_off = off - on``: dopaminergic fluctuation amplitude.

    Parameters
    ----------
    data_dir_preview : str or Path or None
        Unused. Kept for interface consistency.

    Returns
    -------
    Pipeline
        Unfit sklearn Pipeline with a FunctionTransformer prepended.
    """
    from parkinson.features import add_clinical_features

    feature_step = FunctionTransformer(add_clinical_features)
    preprocessor = _make_preprocessor(include_clinical_features=True)
    estimator = HistGradientBoostingRegressor(random_state=0)
    return Pipeline([
        ("add_features", feature_step),
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])


# ---------------------------------------------------------------------------
# Experiment 03 — Hyperparameter Tuning
# ---------------------------------------------------------------------------

def build_tuned(data_dir_preview: str | Path | None = None) -> Pipeline:
    """Return the tuned learner with optimised hyperparameters.

    Uses the same feature-engineering step as experiment 02 but with
    HistGradientBoostingRegressor tuned on:
    - ``max_iter``: up from default 100 for better convergence.
    - ``learning_rate``: lowered from 0.1 to reduce overfitting.
    - ``max_depth``: capped to reduce variance on the small dataset (~300 rows).
    - ``min_samples_leaf``: raised to regularise the leaves further.

    Parameters
    ----------
    data_dir_preview : str or Path or None
        Unused. Kept for interface consistency.

    Returns
    -------
    Pipeline
        Unfit sklearn Pipeline with tuned estimator hyperparameters.
    """
    from parkinson.features import add_clinical_features

    feature_step = FunctionTransformer(add_clinical_features)
    preprocessor = _make_preprocessor(include_clinical_features=True)
    estimator = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        random_state=0,
    )
    return Pipeline([
        ("add_features", feature_step),
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])


# ---------------------------------------------------------------------------
# Default alias: build_learner → baseline (used by templates / smoke tests)
# ---------------------------------------------------------------------------

def build_learner(data_dir_preview: str | Path | None = None) -> Pipeline:
    """Alias for ``build_baseline`` — used by generic scripts and smoke tests."""
    return build_baseline(data_dir_preview=data_dir_preview)
