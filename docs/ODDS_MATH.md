# Odds Mathematics

## Conversions

American odds \(A\) map to implied probability \(p\) via

\[
p = \begin{cases}
\frac{100}{A + 100} & A > 0 \\
\frac{|A|}{|A| + 100} & A < 0
\end{cases}
\]

Decimal odds \(d\) map back with \(d = 1 + \max(A, 0)/100 + \max(-A, 0)/|A|\).

## Shin Overround

Given implied probabilities \(q_i\), solve for bookmaker parameter \(z\):

\[
\sum_i \sqrt{z^2 + 4 (1 - z) q_i^2} = 2
\]

Then adjusted probabilities are

\[
\tilde{p}_i = \frac{\sqrt{z^2 + 4 (1 - z) q_i^2} - z}{2 (1 - z)}
\]

The implementation in `utils.odds_utils.normalize_probabilities_shin` uses bisection to find \(z \in [0, 0.25]\). Overround is `sum(q_i) - 1` and is logged alongside normalized probabilities.

## Expected Value & Kelly

For decimal odds \(d\) and model probability \(p\):

\[
EV = p (d - 1) - (1 - p)
\]

Kelly fraction is

\[
f^* = \frac{p(d-1) - (1 - p)}{d - 1}
\]

We clip \(f^*\) to \([0, 1]\) and apply a risk multiplier in `SelectionConfig`.
