# UI Guide

The Streamlit dashboard (`make ui`) provides five sections:

1. **Upcoming Probabilities** – Tabular view of fighter vs opponent probabilities and intervals.
2. **EV Leaderboard** – Ranked recommendations; filters for min EV coming in future iteration.
3. **Calibration** – Reliability plot generated from current predictions vs simulated outcomes.
4. **Backtest Summary** – Displays ROI and bet counts from latest backtest run.
5. **Refresh Button** – Triggers the update pipeline using cached synthetic data when offline.

Dark mode styling is defined in `src/ufc_winprob/ui/assets/styles.css` and loaded automatically by Streamlit's theming engine.
