"""
Test suite for mathematical correctness of risk-adjusted performance metrics.

Tests Sharpe, Sortino, Max Drawdown, Calmar Ratio, and Volatility calculations
on synthetic portfolios with known properties.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe Ratio."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate annualized Sortino Ratio (penalizes only downside volatility)."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0 or negative_returns.std() == 0:
        return 0.0
    downside_std = negative_returns.std()
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown as a percentage."""
    if len(portfolio_values) < 2:
        return 0.0
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    return drawdowns.min()  # Returns negative value


def calculate_calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate Calmar Ratio (annualized return / abs(max drawdown))."""
    if len(returns) == 0:
        return 0.0
    annualized_return = returns.mean() * periods_per_year
    max_dd = calculate_max_drawdown(portfolio_values)
    if max_dd == 0:
        return 0.0
    return annualized_return / abs(max_dd)


def calculate_annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


# ============================================================================
# TESTS
# ============================================================================

def test_sharpe_ratio_zero_returns():
    """Sharpe ratio of constant portfolio should be 0."""
    returns = np.zeros(252)  # 1 year of zero returns
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0, "Zero returns should give Sharpe = 0"
    print("✓ Sharpe ratio correct for zero returns")


def test_sharpe_ratio_positive_returns():
    """Sharpe ratio should be positive for consistent positive returns."""
    returns = np.full(252, 0.001)  # 0.1% daily return
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0, "Positive returns should give positive Sharpe"
    # Expected: mean=0.001, std≈0, but with noise...
    # Let's add noise
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252)  # mean=0.1%, std=1%
    sharpe = calculate_sharpe_ratio(returns)
    assert 0 < sharpe < 5, f"Sharpe ratio {sharpe} outside reasonable range"
    print(f"✓ Sharpe ratio: {sharpe:.3f} for positive noisy returns")


def test_max_drawdown_no_decline():
    """Max drawdown should be 0 for monotonically increasing portfolio."""
    portfolio = np.array([100, 110, 120, 130, 140, 150])
    mdd = calculate_max_drawdown(portfolio)
    assert mdd == 0.0, "No drawdown for increasing portfolio"
    print("✓ Max Drawdown = 0 for increasing portfolio")


def test_max_drawdown_known_decline():
    """Max drawdown should match known decline."""
    # Portfolio goes 100 -> 150 -> 100 -> 150
    # Max drawdown: (100-150)/150 = -33.33%
    portfolio = np.array([100, 125, 150, 125, 100, 125, 150])
    mdd = calculate_max_drawdown(portfolio)
    expected_mdd = (100 - 150) / 150
    assert np.isclose(mdd, expected_mdd, atol=0.001), \
        f"Expected MDD {expected_mdd:.3f}, got {mdd:.3f}"
    print(f"✓ Max Drawdown: {mdd:.2%} matches expected {expected_mdd:.2%}")


def test_calmar_ratio():
    """Calmar ratio should be annualized_return / abs(max_drawdown)."""
    # Simple case: 252 days, constant 0.1% daily return, one 10% drawdown
    np.random.seed(42)
    portfolio = np.array([100.0])
    returns_list = []

    for i in range(252):
        if i == 100:
            # Introduce 10% drawdown
            ret = -0.10
        else:
            ret = 0.001  # 0.1% daily
        portfolio = np.append(portfolio, portfolio[-1] * (1 + ret))
        returns_list.append(ret)

    returns = np.array(returns_list)
    annualized_ret = returns.mean() * 252
    mdd = calculate_max_drawdown(portfolio)
    expected_calmar = annualized_ret / abs(mdd)

    calmar = calculate_calmar_ratio(returns, portfolio)
    assert np.isclose(calmar, expected_calmar, atol=0.01), \
        f"Expected Calmar {expected_calmar:.3f}, got {calmar:.3f}"
    print(f"✓ Calmar Ratio: {calmar:.3f} (annualized_ret={annualized_ret:.2%}, MDD={mdd:.2%})")


def test_sortino_ratio_no_downside():
    """Sortino ratio should be infinite (or very high) with no negative returns."""
    # All positive returns -> downside std = 0
    returns = np.full(252, 0.001)
    sortino = calculate_sortino_ratio(returns)
    # With no negative returns, denominator is 0, so we return 0 by convention
    # (or could return inf - depends on implementation)
    assert sortino == 0.0, "No negative returns should give Sortino = 0 (undefined)"
    print("✓ Sortino ratio correctly handles no downside case")


def test_sortino_ratio_with_downside():
    """Sortino ratio should be calculable with mixed returns."""
    np.random.seed(42)
    # 60% positive days, 40% negative days
    positive_returns = np.random.uniform(0.001, 0.01, 152)
    negative_returns = np.random.uniform(-0.01, -0.001, 100)
    returns = np.concatenate([positive_returns, negative_returns])
    np.random.shuffle(returns)

    sortino = calculate_sortino_ratio(returns)
    assert sortino != 0.0, "Mixed returns should give non-zero Sortino"
    assert -10 < sortino < 10, f"Sortino {sortino} outside reasonable range"
    print(f"✓ Sortino ratio: {sortino:.3f} for mixed returns")


def test_annualized_volatility():
    """Annualized volatility should scale daily volatility by sqrt(252)."""
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 252)  # mean=0.1%, std=2%

    daily_vol = daily_returns.std()
    annualized_vol = calculate_annualized_volatility(daily_returns)

    expected_annual_vol = daily_vol * np.sqrt(252)
    assert np.isclose(annualized_vol, expected_annual_vol, atol=0.0001), \
        f"Expected {expected_annual_vol:.4f}, got {annualized_vol:.4f}"
    print(f"✓ Annualized volatility: {annualized_vol:.2%} (daily vol: {daily_vol:.2%})")


def test_edge_cases():
    """Test edge cases: empty arrays, single element, etc."""
    empty_returns = np.array([])
    single_return = np.array([0.01])

    # Empty arrays should return 0
    assert calculate_sharpe_ratio(empty_returns) == 0.0
    assert calculate_max_drawdown(empty_returns) == 0.0
    assert calculate_annualized_volatility(empty_returns) == 0.0

    # Single element
    assert calculate_max_drawdown(single_return) == 0.0
    print("✓ Edge cases handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
