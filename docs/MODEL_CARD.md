# Model Card

- **Model**: Logistic Regression meta-model blended with component Elo features.
- **Version**: 0.1.0
- **Intended Use**: Estimate win probabilities for UFC fights and identify EV betting opportunities.
- **Training Data**: Synthetic fixtures approximating historical distributions (see `synthetic_dataset`).
- **Evaluation Metrics**: AUC, LogLoss, Brier, ECE (logged in `data/processed/metrics.csv`).
- **Ethical Considerations**: Uses only public, regulatory-compliant data; avoids scraping private information; respects rate limits.
- **Limitations**: Synthetic data may not capture edge cases; odds API mocked; calibration tuned to fixtures.
- **Monitoring**: Daily refresh monitors drift via ECE and ROI trending.
