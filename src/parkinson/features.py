"""Feature engineering for the Parkinson motor severity dataset.

All transformations here are stateless Python functions attached to the
pipeline via `FunctionTransformer` (or skrub `.skb.apply_func`).
Stateful transformers (encoders, scalers) are instantiated in pipeline.py.

Derived features motivated by clinical domain knowledge:
- `disease_duration`: years since diagnosis → proxy for disease progression.
- `delta_on_off`: difference between OFF and ON motor score → fluctuation amplitude.
"""

from __future__ import annotations

import pandas as pd


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived clinical features to the visit-level feature matrix.

    Adds two columns in-place-style (returns a copy):

    - ``disease_duration``: ``age - age_at_diagnosis`` in years.
      Captures how long the patient has lived with Parkinson's, which
      correlates strongly with cumulative neurodegeneration.
    - ``delta_on_off``: ``off - on`` motor score.
      Quantifies the amplitude of dopaminergic fluctuation; larger values
      indicate more pronounced motor wearing-off.

    Parameters
    ----------
    X : pd.DataFrame
        Visit-level feature matrix as returned by ``load_dataset``.

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` with two additional columns.
    """
    X = X.copy()
    X["disease_duration"] = X["age"] - X["age_at_diagnosis"]
    X["delta_on_off"] = X["off"] - X["on"]
    return X


def drop_id_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that carry no predictive signal.

    ``patient_id`` is used for grouping in cross-validation but must not
    be fed as a feature; ``cohort`` is a study-design label that would
    introduce distribution-shift leakage if included.

    Parameters
    ----------
    X : pd.DataFrame
        Visit-level feature matrix.

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` without ``patient_id`` and ``cohort``.
    """
    return X.drop(columns=["patient_id", "cohort"], errors="ignore")
