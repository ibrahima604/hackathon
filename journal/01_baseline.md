# 01_baseline

## Question / hypothesis

Est-ce qu'un `HistGradientBoostingRegressor` entraîné sur les features brutes
(sans feature engineering manuel) et évalué avec un GroupKFold sur `patient_id`
constitue une baseline saine pour la prédiction du score de sévérité moteur ?

## Motivation

- **Sourcing strategy:** my-pick (baseline forcé au bootstrap)
- **Source(s):**
  - Premier modèle à tester sur tout projet de régression tabulaire : établir
    une performance de référence honnête avant toute optimisation.
  - `HistGradientBoostingRegressor` est retenu car il gère nativement les NaN
    (~40–60 % de missingness sur `ledd`, `time_since_intake_*`, `off`), ce qui
    évite une pipeline d'imputation complexe dès la baseline.
- **Why this matters:** Sans baseline reproductible, on ne peut pas mesurer si
  les expériences suivantes (feature engineering, tuning) apportent réellement
  une amélioration.

## Method

- **Files touched:** `src/parkinson/data.py`, `src/parkinson/pipeline.py`, `src/parkinson/evaluate.py`
- **Change versus baseline (ou expérience précédente):** Point de départ — pas de modification par rapport à un état antérieur.
  - `data.py` : charge `X_train.csv` et `y_train.csv`, aligne sur l'index, expose `groups = patient_id`.
  - `pipeline.py` : `build_baseline()` → OrdinalEncoder sur `gene` + `cohort`, drop de `patient_id`, `HistGradientBoostingRegressor(random_state=0)`.
  - `evaluate.py` : `splitter = GroupKFold(n_splits=5)` — toutes les visites d'un même patient restent dans le même fold.
- **Cross-validation:** GroupKFold(n_splits=5) sur `patient_id` — décision data-driven : structure longitudinale impose un split par patient, pas par visite.
- **Out of scope for this experiment:** Feature engineering, réglage des hyperparamètres, soumission finale.

## Risks / things that could invalidate the result

- Si le `GroupKFold` n'est pas correctement alimenté par `groups=patient_id`, le score sera optimiste (leakage inter-visites).
- Le dataset train contient ~25–30 patients uniques : avec 5 folds, chaque fold de validation ne contient que ~5–6 patients — la variance des métriques CV sera élevée.
- `cohort` (A vs B) diffère dans le test set : si la distribution des cohortes est déséquilibrée entre train et test, la généralisation sera plus faible que suggéré par la CV.

## Status

- **State:** planned
- **Approved by user on:** 2025-07-01
- **Headline result:** n/a
- **Implication for next iteration:** n/a — à remplir après la run
