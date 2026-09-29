"""Data loading for the Parkinson motor severity dataset.

The dataset is a longitudinal panel: multiple visits per patient over time.
Each row is one clinical visit. The target is a continuous motor severity score
(likely a UPDRS sub-score), making this a regression problem.

Key structural facts used downstream:
- `patient_id` is the grouping variable for cross-validation (GroupKFold).
- `age_at_diagnosis` and `age` are visit timestamps in years.
- `ledd`, `time_since_intake_on`, `time_since_intake_off`, `off` have
  substantial missingness (~40–60 %) — HistGradientBoosting handles natively.
- `gene` is a low-cardinality categorical (5 values + missing).
- `cohort` is a binary grouping variable (A / B).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(data_dir: str | Path | None = None):
    """Load X_train and y_train and return ``(X, y, groups)``.

    Parameters
    ----------
    data_dir : str or Path or None
        Path to the folder containing ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``PROJECT_ROOT / "data"`` when *None*.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (299 rows × 11 columns after dropping ``Index``).
    y : pd.Series
        Continuous target (motor severity score).
    groups : pd.Series
        ``patient_id`` values aligned with ``X`` — passed as ``groups=``
        to ``GroupKFold`` so visits from the same patient stay in one fold.
    """
    if data_dir is None:
        from parkinson import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
    data_dir = Path(data_dir)

    X = pd.read_csv(data_dir / "X_train.csv", index_col="Index")
    y = pd.read_csv(data_dir / "y_train.csv", index_col="Index")["target"]

    # Align on index in case rows are not in the same order across files
    X, y = X.align(y, join="inner", axis=0)

    groups = X["patient_id"]
    return X, y, groups


def load_test(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load ``X_test.csv`` for final prediction.

    Parameters
    ----------
    data_dir : str or Path or None
        Path to the folder containing ``X_test.csv``.
        Defaults to ``PROJECT_ROOT / "data"`` when *None*.

    Returns
    -------
    X_test : pd.DataFrame
        Feature matrix for the held-out test set.
    """
    if data_dir is None:
        from parkinson import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
    data_dir = Path(data_dir)

    return pd.read_csv(data_dir / "X_test.csv", index_col="Index")
