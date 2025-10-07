# Component Elo Design

The platform models each fighter with an 8-dimensional Elo vector \(E \in \mathbb{R}^8\) representing components: striking, grappling, wrestling, submissions, cardio, durability, fight IQ, and aggression. Each dimension tracks a rating and uncertainty.

## Expectation

For fighters A and B with vectors \(E_A\) and \(E_B\), we compute the weighted delta:

\[
\Delta = \sum_k w_k (E_{A,k} - E_{B,k})
\]

and transform via a logistic link:

\[
P(A \text{ beats } B) = \frac{1}{1 + e^{-\Delta / 400}}
\]

Weights \(w_k\) are defined in `style_taxonomy.py` and allow emphasizing specific styles.

## Age & Division Adjustments

Each weight class has an anchor age \(a_d\). We fit quadratic splines over historical outcomes and compute an effect term:

\[
\phi_d(age) = \text{clip}(c_0 + c_1 x + c_2 x^2, -0.5, 0.5), \quad x = \frac{age - a_d}{5}
\]

During updates, the K-factor is scaled as

\[
K_{eff} = K_0 \times (1 + \phi_d(age_A) - \phi_d(age_B))
\]

A slower decay applies to heavier divisions via half-life overrides.

## Update Rule

After a bout with result \(r \in \{0, 0.5, 1\}\):

\[
E'_A = E_A + K_{eff} (r - P(A)) w
\]
\[
E'_B = E_B + K_{eff} ((1-r) - P(B)) w
\]

Uncertainty shrinks multiplicatively (0.98) after each update with a floor of 50 Elo. Extended layoffs apply time decay via exponential smoothing (half-life 365 days base).

## Calibration

Raw Elo expectations are blended with supervised outputs and calibrated per division using isotonic regression. Calibration metrics (Brier, LogLoss, ECE) are logged in `data/processed/metrics.csv`.
