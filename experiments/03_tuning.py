# %% [markdown]
# # Experiment 03 — Hyperparameter Tuning
#
# **Date:** 2025-07-01
# **Design note:** journal/03_tuning.md
# **Goal:** Optimiser les hyperparamètres de HistGradientBoostingRegressor
# (max_iter=500, learning_rate=0.05, max_depth=4, min_samples_leaf=10)
# sur le pipeline feature-engineered pour réduire le RMSE vs expérience 02.
# **Result:** à remplir après la run.

# %%
import skore

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset
from parkinson.evaluate import splitter
from parkinson.pipeline import build_tuned

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
# `build_tuned()` utilise les mêmes features que 02 mais avec des
# hyperparamètres régularisés pour le petit dataset (~300 lignes) :
# - `max_iter=500` : plus d'itérations avec un taux d'apprentissage réduit.
# - `learning_rate=0.05` : convergence plus lente mais moins de sur-ajustement.
# - `max_depth=4` : arbres moins profonds pour réduire la variance.
# - `min_samples_leaf=10` : feuilles plus larges pour mieux régulariser.

# %%
X, y, groups = load_dataset(data_dir=DATA_DIR)
learner = build_tuned()

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
project.put("03_tuning", report)
print("✓ Rapport 03_tuning sauvegardé dans reports/")

# %% [markdown]
# ## Prédictions finales sur X_test (optionnel)
#
# Une fois satisfait du modèle, relancer la cellule ci-dessous pour
# générer les prédictions sur le jeu de test et créer le fichier de
# soumission au format attendu par `data/sample_submission.csv`.

# %%
import pandas as pd
from parkinson.data import load_test

# Entraîner le modèle final sur toutes les données train
learner_final = build_tuned()
learner_final.fit(X, y)

X_test = load_test(data_dir=DATA_DIR)
y_pred = learner_final.predict(X_test)

submission = pd.DataFrame({"Index": X_test.index, "target": y_pred})
submission.to_csv(DATA_DIR / "submission.csv", index=False)
print(f"✓ Soumission sauvegardée dans data/submission.csv ({len(submission)} lignes)")
