"""Portfolio risk-adjusted performance metrics.

Computes Sharpe, Sortino, Calmar, max drawdown and annualised volatility
from a portfolio-value trajectory. All metrics assume a daily step size
unless ``periods_per_year`` is overridden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RiskMetrics:
    """Risk-adjusted metrics computed from a portfolio trajectory."""

    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float           # negative number
    volatility_annualised: float  # decimal, e.g. 0.18 = 18%
    annualised_return: float      # decimal
    total_return: float           # decimal

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe_ratio": self.sharpe,
            "sortino_ratio": self.sortino,
            "calmar_ratio": self.calmar,
            "max_drawdown": self.max_drawdown,
            "volatility_annualised": self.volatility_annualised,
            "annualised_return": self.annualised_return,
            "total_return": self.total_return,
        }


def compute_risk_metrics(
    portfolio_values: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> RiskMetrics:
    """Compute risk-adjusted metrics from a portfolio value sequence.

    Parameters
    ----------
    portfolio_values:
        Time-ordered portfolio values (length ``T+1``: one per step plus
        the initial value).
    risk_free_rate:
        Annualised risk-free rate (decimal). Defaults to 0.
    periods_per_year:
        Sampling frequency. 252 for daily data, 12 for monthly, etc.
    """
    p = np.asarray(portfolio_values, dtype=float)
    if p.size < 2:
        raise ValueError("Need at least 2 portfolio values.")
    if (p <= 0).any():
        raise ValueError("Portfolio values must be strictly positive.")

    # Period returns
    returns = np.diff(p) / p[:-1]
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period

    # Sharpe (annualised)
    sd = excess.std(ddof=1)
    sharpe = (excess.mean() / sd * np.sqrt(periods_per_year)
              if sd > 1e-12 else 0.0)

    # Sortino (annualised) — downside deviation only
    downside = excess[excess < 0]
    dd_std = downside.std(ddof=1) if downside.size > 1 else 0.0
    sortino = (excess.mean() / dd_std * np.sqrt(periods_per_year)
               if dd_std > 1e-12 else 0.0)

    # Volatility (annualised)
    vol = returns.std(ddof=1) * np.sqrt(periods_per_year)

    # Returns
    total_return = float(p[-1] / p[0] - 1)
    n_periods = len(returns)
    years = n_periods / periods_per_year
    if years > 0 and (1 + total_return) > 0:
        ann_return = (1 + total_return) ** (1 / years) - 1
    else:
        ann_return = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(p)
    drawdowns = (p - running_max) / running_max
    mdd = float(drawdowns.min())  # negative

    # Calmar = annualised return / |MDD|
    calmar = ann_return / abs(mdd) if mdd < -1e-12 else 0.0

    return RiskMetrics(
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=mdd,
        volatility_annualised=float(vol),
        annualised_return=float(ann_return),
        total_return=total_return,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    daily = rng.normal(0.0008, 0.012, 252)  # ~20% annual return, ~19% vol
    portfolio = 10000.0 * np.cumprod(1 + daily)
    portfolio = np.concatenate([[10000.0], portfolio])
    m = compute_risk_metrics(portfolio)
    print(f"Sharpe   : {m.sharpe:+.3f}")
    print(f"Sortino  : {m.sortino:+.3f}")
    print(f"Calmar   : {m.calmar:+.3f}")
    print(f"MDD      : {m.max_drawdown * 100:+.2f}%")
    print(f"Vol      : {m.volatility_annualised * 100:.2f}%")
    print(f"Ann ret  : {m.annualised_return * 100:+.2f}%")
    print(f"Total ret: {m.total_return * 100:+.2f}%")
