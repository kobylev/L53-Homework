"""Statistical significance tests for evaluation metrics.

This module provides hypothesis tests to determine whether observed
performance metrics differ meaningfully from random-baseline behaviour.
The tests are intended to be called from ``src/evaluate.py`` after a
test-set evaluation has been run; their outputs are reported alongside
the headline metrics in ``assets/logs/eval_results.csv``.

Two independent tests are provided for directional accuracy:

* :func:`binomial_test_directional_accuracy` — closed-form one-sided
  binomial test against the null ``p = 0.5``. Returns the exact
  p-value and a Wilson 95% confidence interval.
* :func:`permutation_test_directional_accuracy` — empirical p-value
  obtained by shuffling the action labels ``n_permutations`` times.
  Useful when the IID assumption of the binomial test is suspect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    """Result of a one-sided test against a random baseline.

    Attributes
    ----------
    accuracy:
        Observed accuracy in [0, 1].
    n_trials:
        Number of trials (Buy and Sell actions only; Hold excluded).
    n_correct:
        Number of correct directional predictions.
    p_value:
        One-sided p-value for ``H1: accuracy > 0.5``.
    ci_low, ci_high:
        Lower and upper bounds of the 95% confidence interval.
    method:
        Name of the test that produced this result.
    """

    accuracy: float
    n_trials: int
    n_correct: int
    p_value: float
    ci_low: float
    ci_high: float
    method: str

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Return True if the result rejects the null at level ``alpha``."""
        return self.p_value < alpha

    def summary(self) -> str:
        verdict = (
            "statistically significant"
            if self.is_significant()
            else "NOT statistically distinguishable from random"
        )
        return (
            f"Directional accuracy = {self.accuracy * 100:.2f}% "
            f"({self.n_correct}/{self.n_trials})  |  "
            f"95% CI [{self.ci_low * 100:.2f}%, {self.ci_high * 100:.2f}%]  |  "
            f"p = {self.p_value:.4f} ({self.method}) — {verdict}"
        )


def _wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval — better than normal-approx near 0/1."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def directional_accuracy(
    actions: Sequence[int],
    price_changes: Sequence[float],
) -> tuple[int, int]:
    """Count correct directional predictions.

    A "directional prediction" only counts when the agent took a
    non-Hold action (Buy=1 or Sell=2). Buy is correct iff the next
    price change is positive; Sell is correct iff it is negative.

    Returns
    -------
    (n_correct, n_trials) — both integers.
    """
    actions = np.asarray(actions)
    price_changes = np.asarray(price_changes)
    if len(actions) != len(price_changes):
        raise ValueError(
            f"Length mismatch: actions={len(actions)}, "
            f"price_changes={len(price_changes)}"
        )
    mask = actions != 0  # exclude Hold
    a = actions[mask]
    d = price_changes[mask]
    correct = ((a == 1) & (d > 0)) | ((a == 2) & (d < 0))
    return int(correct.sum()), int(mask.sum())


def binomial_test_directional_accuracy(
    actions: Sequence[int],
    price_changes: Sequence[float],
    p_null: float = 0.5,
) -> SignificanceResult:
    """One-sided exact binomial test against ``H0: accuracy == p_null``."""
    n_correct, n_trials = directional_accuracy(actions, price_changes)
    if n_trials == 0:
        return SignificanceResult(
            accuracy=0.0, n_trials=0, n_correct=0,
            p_value=1.0, ci_low=0.0, ci_high=0.0,
            method="binomial",
        )
    res = stats.binomtest(n_correct, n_trials, p=p_null, alternative="greater")
    ci_low, ci_high = _wilson_interval(n_correct, n_trials)
    return SignificanceResult(
        accuracy=n_correct / n_trials,
        n_trials=n_trials,
        n_correct=n_correct,
        p_value=float(res.pvalue),
        ci_low=ci_low,
        ci_high=ci_high,
        method="binomial",
    )


def permutation_test_directional_accuracy(
    actions: Sequence[int],
    price_changes: Sequence[float],
    n_permutations: int = 5000,
    random_state: int | None = None,
) -> SignificanceResult:
    """Empirical p-value via label permutation.

    Shuffles the non-Hold action labels ``n_permutations`` times and
    computes how often a random labelling beats the observed accuracy.
    """
    rng = np.random.default_rng(random_state)
    actions = np.asarray(actions)
    price_changes = np.asarray(price_changes)
    mask = actions != 0
    a = actions[mask].copy()
    d = price_changes[mask]
    n_trials = mask.sum()
    if n_trials == 0:
        return SignificanceResult(
            accuracy=0.0, n_trials=0, n_correct=0,
            p_value=1.0, ci_low=0.0, ci_high=0.0,
            method="permutation",
        )

    correct = ((a == 1) & (d > 0)) | ((a == 2) & (d < 0))
    observed_acc = correct.mean()

    perm_accs = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(a)
        c = ((shuffled == 1) & (d > 0)) | ((shuffled == 2) & (d < 0))
        perm_accs[k] = c.mean()

    # one-sided: how often does shuffling do at least as well as observed
    p_value = float((perm_accs >= observed_acc).mean())
    ci_low, ci_high = _wilson_interval(int(correct.sum()), int(n_trials))
    return SignificanceResult(
        accuracy=float(observed_acc),
        n_trials=int(n_trials),
        n_correct=int(correct.sum()),
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        method=f"permutation(n={n_permutations})",
    )


if __name__ == "__main__":
    # quick smoke test with a tiny synthetic example
    rng = np.random.default_rng(0)
    n = 400
    actions = rng.choice([0, 1, 2], size=n, p=[0.65, 0.20, 0.15])
    price_changes = rng.normal(0, 1, size=n)
    # bias the labels slightly so the test should detect signal
    for i in range(n):
        if rng.random() < 0.55 and actions[i] == 1:
            price_changes[i] = abs(price_changes[i])
        elif rng.random() < 0.55 and actions[i] == 2:
            price_changes[i] = -abs(price_changes[i])

    print(binomial_test_directional_accuracy(actions, price_changes).summary())
    print(permutation_test_directional_accuracy(
        actions, price_changes, random_state=0).summary())
