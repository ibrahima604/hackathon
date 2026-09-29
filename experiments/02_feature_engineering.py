# %% [markdown]
# # Experiment 02 — Feature Engineering
#
# **Date:** 2025-07-01
# **Design note:** journal/02_feature_engineering.md
# **Goal:** Tester si l'ajout de `disease_duration` et `delta_on_off`
# améliore le RMSE en CV par rapport à la baseline (01).
# **Result:** à remplir après la run.

# %%
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset
from parkinson.evaluate import splitter
from parkinson.pipeline import build_feature_eng

# %% [markdown]
# ## Chemins

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Projet skore

# %%
project = skore.Project(
    name="parkinson",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)

# %% [markdown]
# ## Données et learner
#
# `build_feature_eng()` ajoute en première étape du pipeline un
# `FunctionTransformer` qui calcule :
# - `disease_duration = age - age_at_diagnosis`
# - `delta_on_off = off - on`
#
# Ces deux features sont stateless (pas de leakage) et traitées comme
# features numériques normales par HistGBR (NaN natif).

# %%
X, y, groups = load_dataset(data_dir=DATA_DIR)
learner = build_feature_eng()

# %% [markdown]
# ## Évaluation croisée

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

# %%
project.put("02_feature_engineering", report)
print("✓ Rapport 02_feature_engineering sauvegardé dans reports/")
