# %% [markdown]
# # Experiment 01 — Baseline
#
# **Date:** 2025-07-01
# **Design note:** journal/01_baseline.md
# **Goal:** Établir une baseline avec HistGradientBoostingRegressor sur les
# features brutes, évalué par GroupKFold(5) sur patient_id.
# **Result:** à remplir après la run.

# %%
import pandas as pd
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset
from parkinson.evaluate import splitter
from parkinson.pipeline import build_baseline

# %% [markdown]
# ## Chemins
#
# `PROJECT_ROOT` est résolu depuis `src/parkinson/__init__.py` — indépendant
# du répertoire courant. Toutes les résolutions de chemin passent par cette
# constante.

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Projet skore
#
# Ouvre le projet local dans `reports/`. La clé de stockage est le stem du
# fichier (`01_baseline`) pour permettre la comparaison inter-expériences.

# %%
project = skore.Project(
    name="parkinson",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)

# %% [markdown]
# ## Données et learner
#
# `load_dataset` retourne `(X, y, groups)`.
# `groups` contient `patient_id` et est passé au splitter GroupKFold via
# `split_kwargs` pour éviter le leakage inter-visites du même patient.

# %%
X, y, groups = load_dataset(data_dir=DATA_DIR)
learner = build_baseline()

# %% [markdown]
# ## Évaluation croisée
#
# `splitter` est GroupKFold(n_splits=5) importé depuis `parkinson.evaluate`.
# `groups=groups` garantit que toutes les visites d'un même patient restent
# dans le même fold (pas de leakage).

# %%
report = skore.evaluate(
    learner,
    X=X,
    y=y,
    cv=splitter,
    groups=groups,
)
report

# %% [markdown]
# ## Persistance
#
# La clé `"01_baseline"` stocke ce rapport dans le projet skore.
# Relancer ce script avec la même clé écrase le rapport précédent —
# créer un nouveau fichier d'expérience si vous voulez conserver les deux.

# %%
project.put("01_baseline", report)
print("✓ Rapport 01_baseline sauvegardé dans reports/")
