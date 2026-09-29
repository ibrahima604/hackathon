# JOURNAL

## Status

- **Project / dataset:** Parkinson's Disease — longitudinal motor severity (UPDRS-like) regression
- **Goal:** Prédire le score de sévérité moteur (`target`, variable continue) à chaque visite clinique, à partir des données démographiques, génétiques et des scores ON/OFF du patient.
- **Last experiment:** 03_tuning — planned
- **Last result:** n/a

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-01
  - env manager: pip+venv (.venv) - recorded: 2025-07-01
  - agent feature: skipped - recorded: 2025-07-01
  - optional features: none - recorded: 2025-07-01
  - package name (`src/parkinson/`): parkinson - recorded: 2025-07-01
  - skore mode: local - recorded: 2025-07-01
  - skore hub workspace: n/a - recorded: 2025-07-01
  - skore mlflow tracking uri: n/a - recorded: 2025-07-01
  - student prior: some-sklearn - recorded: 2025-07-01
  - CV splitter family: GroupKFold (n_splits=5) sur patient_id - recorded: 2025-07-01

## Data understanding (EDA)

- **Status:** skipped — 2025-07-01 (analyse manuelle réalisée)
- **Summary:**
  - 299 lignes × 11 features dans X_train ; 90 lignes dans X_test.
  - ~25–30 patients uniques dans le train (données longitudinales : plusieurs visites par patient).
  - Target continue, plage ~2 à 70 (probable score UPDRS moteur ou sous-score).
  - Missingness importante : `ledd` (~25 %), `time_since_intake_on` / `off` (~50 %), `gene` (~15 %).
  - `gene` : 5 catégories (`LRRK2+`, `GBA+`, `OTHER+`, `No Mutation`, vide).
  - `cohort` : 2 niveaux (A majoritaire, B minoritaire).
  - Risque de leakage si GroupKFold non utilisé — visits du même patient ne doivent pas être dans les deux splits.
- **Report:** n/a (EDA skipped)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_baseline` | HistGradientBoostingRegressor sur features brutes, GroupKFold sur patient_id | planned | n/a | [design note](01_baseline.md) |
| `02_feature_engineering` | Ajout de `disease_duration` et `delta_on_off`, même estimateur | planned | n/a | [design note](02_feature_engineering.md) |
| `03_tuning` | Optimisation des hyperparamètres de HistGBR sur les features ingéniérées | planned | n/a | [design note](03_tuning.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Explorer un LightGBM ou XGBoost en remplacement de HistGBR | my-pick |
| B2 | Ajouter des features d'agrégats par patient (mean/last des scores ON/OFF) | my-pick |
| B3 | Soumettre les prédictions sur X_test et générer sample_submission.csv | user |
