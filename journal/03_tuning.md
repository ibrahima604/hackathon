# 03_tuning

## Question / hypothesis

L'optimisation des hyperparamètres de `HistGradientBoostingRegressor`
(learning rate, max_depth, min_samples_leaf, max_iter) sur le pipeline
feature-engineered réduit-elle davantage le RMSE en CV par rapport à
l'expérience 02 ?

## Motivation

- **Sourcing strategy:** my-pick:02_feature_engineering
- **Source(s):**
  - Le dataset est petit (~300 lignes, ~25–30 patients) : les valeurs par défaut
    de HistGBR (`max_iter=100`, `learning_rate=0.1`) risquent de sous-entraîner
    ou de surfit sur des folds de validation très petits.
  - `max_depth` et `min_samples_leaf` sont les leviers de régularisation les
    plus efficaces pour les datasets de petite taille avec HistGBR
    (scikit-learn docs, § Regularization).
  - Après 02, si les features dérivées apportent du signal, le tuning peut
    exploiter ce signal plus efficacement en ajustant la complexité des arbres.
- **Why this matters:** Un modèle sous-régularisé surestimera ses performances en
  CV sur 5 folds de ~5 patients chacun ; le tuning permet de trouver le bon
  compromis biais-variance pour un si petit échantillon.

## Method

- **Files touched:** `src/parkinson/pipeline.py`
- **Change versus 02_feature_engineering:**
  - `pipeline.py` : `build_tuned()` → même feature engineering qu'en 02, mais
    avec `HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
    max_depth=4, min_samples_leaf=10, random_state=0)`.
  - Ces valeurs sont des points de départ raisonnés, pas le résultat d'une
    recherche exhaustive. Une GridSearch ou RandomizedSearch n'est pas prévue
    dans cette expérience pour ne pas sur-optimiser sur un ensemble CV aussi
    petit.
  - `evaluate.py` : inchangé.
- **Cross-validation:** GroupKFold(n_splits=5) sur `patient_id` — identique.
- **Out of scope for this experiment:** Recherche automatique d'hyperparamètres
  (GridSearch / Optuna), nouvelles features, soumission finale.

## Risks / things that could invalidate the result

- Avec ~5–6 patients par fold de validation, l'intervalle de confiance du RMSE
  est large : une amélioration de < 1 point RMSE peut être du bruit.
- Les valeurs d'hyperparamètres choisies manuellement peuvent ne pas être
  optimales — elles constituent un point de départ, pas un optimum garanti.
- Sur-regulariser (`min_samples_leaf` trop élevé, `max_depth` trop faible) peut
  au contraire dégrader les performances par rapport à la baseline.

## Status

- **State:** planned
- **Approved by user on:** 2025-07-01
- **Headline result:** n/a
- **Implication for next iteration:** n/a — à remplir après la run
