# 02_feature_engineering

## Question / hypothesis

L'ajout de deux features dérivées du domaine clinique — `disease_duration`
(durée depuis le diagnostic) et `delta_on_off` (amplitude de fluctuation
motrice) — améliore-t-il le RMSE en CV par rapport à la baseline sur features
brutes ?

## Motivation

- **Sourcing strategy:** my-pick:01_baseline
- **Source(s):**
  - Connaissance du domaine : dans la maladie de Parkinson, la durée de la
    maladie (`age - age_at_diagnosis`) est l'un des prédicteurs les plus
    importants de la sévérité motrice cumulée (progression neurodégénérative).
  - L'amplitude ON/OFF (`off - on`) mesure directement la fluctuation
    dopaminergique, mécanisme physiopathologique central de la maladie.
  - Ces deux features sont calculables à partir de colonnes déjà présentes dans
    `X_train` sans aucune information future — pas de leakage.
- **Why this matters:** Si ces deux features réduisent le RMSE, elles valident
  l'utilité de l'encodage du domaine métier et guident l'expérience 03 (tuning).

## Method

- **Files touched:** `src/parkinson/features.py`, `src/parkinson/pipeline.py`
- **Change versus 01_baseline:**
  - `features.py` : `add_clinical_features(X)` → ajoute `disease_duration` et
    `delta_on_off` (FunctionTransformer, stateless, ne modifie pas les colonnes
    existantes).
  - `pipeline.py` : `build_feature_eng()` → FunctionTransformer en première
    étape du Pipeline, puis le même ColumnTransformer + HistGBR que la baseline.
  - `evaluate.py` : inchangé — même splitter GroupKFold(5).
- **Cross-validation:** GroupKFold(n_splits=5) sur `patient_id` — identique à la baseline.
- **Out of scope for this experiment:** Réglage des hyperparamètres, autres features (agrégats par patient, features temporelles), soumission finale.

## Risks / things that could invalidate the result

- `disease_duration` peut être NaN si `age_at_diagnosis` est manquant (observé sur `ZXHK0345`) — HistGBR gère nativement, mais le feature peut porter moins de signal que prévu.
- `delta_on_off` est NaN si l'une des deux colonnes `on` ou `off` est manquante (fréquent : ~50 %) — le signal est disponible seulement pour une fraction des visites.
- Une amélioration faible (< 0.5 RMSE) ne justifie pas à elle seule de complexifier le pipeline ; la décision de conserver ces features pour le tuning reste au jugement post-run.

## Status

- **State:** planned
- **Approved by user on:** 2025-07-01
- **Headline result:** n/a
- **Implication for next iteration:** n/a — à remplir après la run
