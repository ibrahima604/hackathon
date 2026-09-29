"""Cross-validation strategy for the Parkinson motor severity experiments.

The dataset is a longitudinal patient panel: multiple visits per patient.
Using a random K-Fold would leak future visits of patient A into the
validation fold while past visits of the same patient are in the training
fold — inflating the apparent generalisation.

The correct strategy is **GroupKFold** on ``patient_id``: all visits
from a given patient land in exactly one fold, so the model is always
evaluated on patients it has never seen during training.

The ``groups`` array is returned by ``parkinson.data.load_dataset`` and
passed directly to ``skore.evaluate(..., groups=groups)``.
"""

from __future__ import annotations

from sklearn.model_selection import GroupKFold

# 5-fold GroupKFold: each fold holds ~20 % of the patients.
# With ~25–30 unique patients in the training set this gives
# roughly 5–6 patients per validation fold.
splitter = GroupKFold(n_splits=5)
