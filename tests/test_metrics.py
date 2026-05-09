"""Tests for risk_metrics and significance_tests modules.

The metrics are validated against synthetic series with KNOWN ground
truth values, so any implementation drift is caught immediately.

Run with:  pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk_metrics import compute_risk_metrics
from src.significance_tests import (
    binomial_test_directional_accuracy,
    permutation_test_directional_accuracy,
    directional_accuracy,
)


# ------------------------------ risk_metrics ------------------------------


def test_max_drawdown_known_case() -> None:
    """A V-shaped trajectory has MDD = (trough - peak) / peak."""
    p = np.array([100.0, 110.0, 120.0, 60.0, 70.0, 130.0])
    m = compute_risk_metrics(p)
    # Peak before trough is 120; trough is 60. MDD = (60-120)/120 = -50%
    assert abs(m.max_drawdown - (-0.5)) < 1e-9, (
        f"MDD off: got {m.max_drawdown}, expected -0.5"
    )


def test_max_drawdown_monotonic_growth_is_zero() -> None:
    """Strictly increasing portfolio has MDD == 0."""
    p = np.linspace(10000, 20000, 100)
    m = compute_risk_metrics(p)
    assert m.max_drawdown == 0.0


def test_sharpe_zero_variance_gives_zero() -> None:
    """Constant portfolio means zero variance — Sharpe defined as 0."""
    p = np.full(100, 10000.0)
    m = compute_risk_metrics(p)
    assert m.sharpe == 0.0
    assert m.sortino == 0.0


def test_sharpe_positive_drift() -> None:
    """A series with positive mean and small noise should have Sharpe > 0."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 252)
    p = 10000 * np.cumprod(1 + rets)
    p = np.concatenate([[10000], p])
    m = compute_risk_metrics(p)
    assert m.sharpe > 0


def test_calmar_consistency() -> None:
    """Calmar must equal annualised_return / |max_drawdown|."""
    rng = np.random.default_rng(1)
    p = 10000 * np.cumprod(1 + rng.normal(0.0005, 0.015, 504))
    p = np.concatenate([[10000], p])
    m = compute_risk_metrics(p)
    expected = m.annualised_return / abs(m.max_drawdown)
    assert abs(m.calmar - expected) < 1e-9


def test_volatility_annualisation() -> None:
    """Vol must scale by sqrt(252) by default."""
    rng = np.random.default_rng(2)
    rets = rng.normal(0, 0.01, 1000)
    p = 10000 * np.cumprod(1 + rets)
    p = np.concatenate([[10000], p])
    m = compute_risk_metrics(p, periods_per_year=252)
    raw_std = rets.std(ddof=1)
    assert abs(m.volatility_annualised - raw_std * np.sqrt(252)) < 1e-3


def test_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        compute_risk_metrics([10000.0, 0.0, 5000.0])
    with pytest.raises(ValueError):
        compute_risk_metrics([10000.0])  # too short


# --------------------------- significance_tests ---------------------------


def test_directional_accuracy_perfect() -> None:
    actions = np.array([1, 1, 2, 2, 0])
    deltas = np.array([1.0, 0.5, -0.5, -1.0, 99.0])  # last masked out
    n_correct, n_trials = directional_accuracy(actions, deltas)
    assert n_correct == 4 and n_trials == 4


def test_directional_accuracy_excludes_hold() -> None:
    actions = np.array([0, 0, 0, 0])
    deltas = np.array([1.0, -1.0, 1.0, -1.0])
    n_correct, n_trials = directional_accuracy(actions, deltas)
    assert n_correct == 0 and n_trials == 0


def test_binomial_test_random_yields_high_pvalue() -> None:
    """Random labels should give p-value far from significant."""
    rng = np.random.default_rng(0)
    n = 200
    actions = rng.choice([1, 2], size=n)
    deltas = rng.normal(0, 1, size=n)
    res = binomial_test_directional_accuracy(actions, deltas)
    assert 0.0 <= res.p_value <= 1.0
    assert res.p_value > 0.05, (
        f"Random data yielded p={res.p_value}; binomial test broken."
    )


def test_binomial_test_strong_signal_is_significant() -> None:
    """Perfectly correlated labels and prices must reject the null."""
    actions = np.array([1] * 100 + [2] * 100)
    deltas = np.array([1.0] * 100 + [-1.0] * 100)
    res = binomial_test_directional_accuracy(actions, deltas)
    assert res.is_significant()
    assert res.accuracy == 1.0


def test_permutation_test_matches_binomial_in_signal_case() -> None:
    """Both tests should agree when signal is large."""
    actions = np.array([1] * 50 + [2] * 50)
    deltas = np.array([0.5] * 50 + [-0.5] * 50)
    a = binomial_test_directional_accuracy(actions, deltas)
    b = permutation_test_directional_accuracy(
        actions, deltas, n_permutations=2000, random_state=0)
    assert a.is_significant() and b.is_significant()


def test_wilson_ci_contains_estimate() -> None:
    rng = np.random.default_rng(3)
    actions = rng.choice([1, 2], size=300)
    deltas = rng.normal(0, 1, size=300)
    res = binomial_test_directional_accuracy(actions, deltas)
    assert res.ci_low <= res.accuracy <= res.ci_high


def test_reported_5234_pct_is_NOT_significant_at_n_134() -> None:
    """The README's claim '52.34% above random baseline' is misleading.

    With 70/134 correct (52.24%) the binomial test gives p ~= 0.33,
    which is FAR from the conventional alpha=0.05 threshold. This
    regression test pins that fact in code so future README edits
    cannot quietly resurrect the misleading claim.
    """
    actions = np.array([1] * 70 + [2] * 64)
    # Construct deltas so 70 of 134 are 'correct'.
    deltas = np.concatenate([
        np.ones(70),         # 70 buys, all correct (positive moves)
        np.ones(64),         # 64 sells, all wrong (positive moves)
    ])
    res = binomial_test_directional_accuracy(actions, deltas)
    assert abs(res.accuracy - 70/134) < 1e-9
    assert res.p_value > 0.10, (
        f"Expected p>>0.05 for 52.24% accuracy at N=134, got p={res.p_value}"
    )
    assert not res.is_significant()
