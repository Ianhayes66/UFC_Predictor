# Modeling Notes

- **Dataset** – Synthetic fixtures mirror UFC fight distributions with balanced outcomes and component Elo deltas.
- **Features** – Age adjustments, Elo vectors, market priors, activity gaps, and main-event flags.
- **Model** – Logistic regression meta-model with time-series split (3 folds). LightGBM placeholder reserved for future upgrades.
- **Calibration** – Isotonic regression on validation fold per division. Metrics stored in `data/processed/metrics.csv`.
- **Seeds** – Global seed 42 ensures deterministic splits and random draws.
- **Future Work** – Add Optuna search, per-division calibrators, and SHAP explainability saved to `reports/`.
