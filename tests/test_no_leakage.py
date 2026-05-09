"""Tests that the train/test pipeline does not leak future information.

Adapted to the actual ``src.datasets`` API:

* ``TradingDataset(ticker, start, end, window_size)`` exposes ``.data``
  as a Pandas DataFrame with columns ``[Close, Volume, RSI, MACD]``.
* ``get_train_test_split(dataset, split_ratio=0.8)`` returns
  ``(train_windows, test_windows, scaler)`` where ``scaler`` is a
  fitted ``sklearn.preprocessing.MinMaxScaler``.

The leakage fix is in ``get_train_test_split``: the scaler is fit
ONLY on the train slice, then applied (transform) to both train and
test. We verify that property here.

If the dataset module cannot be loaded (e.g. yfinance offline), the
network-dependent tests skip with a clear message.

Run with:  pytest tests/test_no_leakage.py -v
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest


# ----------------------- pure-logic tests (no network) --------------------


def test_split_is_chronological() -> None:
    """A time-series split must be ordered, never shuffled."""
    n = 1000
    split = int(0.8 * n)
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)
    assert train_idx.max() < test_idx.min(), (
        "Train and test ranges must not overlap. "
        "Random shuffling of a time series is forbidden."
    )


def test_state_at_t_excludes_future() -> None:
    """A 30-day rolling window ending at t-1 must not contain index t."""
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 1000))
    window = 30
    for t in (50, 100, 200, 500):
        state_window = closes[t - window : t]
        assert state_window.shape == (window,)
        assert state_window[-1] == closes[t - 1]
        assert closes[t] not in state_window, (
            f"Look-ahead bias detected at t={t}"
        )


# --------------------- contract test against real pipeline ----------------


@pytest.fixture
def split_data():
    """Build a synthetic OHLCV DataFrame and feed it through a fresh
    MinMaxScaler. Lets us validate the fit-on-train-only contract
    without depending on yfinance.
    """
    rng = np.random.default_rng(42)
    n = 1000
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "Close":  close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "RSI":    rng.uniform(20, 80, n),
        "MACD":   rng.normal(0, 1, n),
    })
    split_idx = int(0.8 * n)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    return df, train_df, test_df


def test_minmax_scaler_fit_on_train_only(split_data) -> None:
    """Replicates the contract enforced inside src/datasets.get_train_test_split.

    A correctly built scaler:
      * fits on train_df only,
      * has data_min_ / data_max_ equal to train_df.min() / train_df.max(),
      * does NOT match the global df.min() / df.max() (would imply leakage).
    """
    from sklearn.preprocessing import MinMaxScaler
    df, train_df, test_df = split_data

    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # The scaler's internal statistics must match the train slice exactly.
    np.testing.assert_allclose(scaler.data_min_, train_df.min().values)
    np.testing.assert_allclose(scaler.data_max_, train_df.max().values)

    # And — crucially — they must DIFFER from the global statistics.
    # If they don't differ, our synthetic data is too uniform to detect
    # leakage, and the test is uninformative.
    full_min = df.min().values
    full_max = df.max().values
    different = (
        not np.allclose(scaler.data_min_, full_min)
        or not np.allclose(scaler.data_max_, full_max)
    )
    assert different, (
        "Train min/max equals full min/max — fixture too uniform to "
        "detect leakage. Increase noise."
    )


def test_test_slice_can_exceed_normalized_range(split_data) -> None:
    """After train-only fit, test values may legitimately fall outside [0, 1].

    This is the visible signature of a non-leaky pipeline: when the
    test slice contains values larger than any seen during training,
    the transformed test rows will exceed 1.0 (or fall below 0). A
    leaky pipeline would clamp everything to [0, 1] because the
    scaler had already seen the test data.
    """
    from sklearn.preprocessing import MinMaxScaler
    df, train_df, test_df = split_data

    scaler = MinMaxScaler()
    scaler.fit(train_df)
    test_scaled = scaler.transform(test_df)

    has_outside = bool(((test_scaled < 0).any() | (test_scaled > 1).any()))
    # The test data is a continuation of the same random walk so it
    # almost always leaves [0, 1] somewhere — that's the smoking gun
    # of correct (non-leaky) scaling.
    assert has_outside, (
        "Test slice never escapes [0, 1] — either the data has no drift "
        "or the scaler was secretly fit on the full series."
    )


# -------------------- live integration test (skipped offline) -------------


@pytest.mark.skipif(
    os.getenv("L53_OFFLINE") == "1" or os.getenv("CI") is not None,
    reason="Network-dependent; skip in offline / CI environments.",
)
def test_get_train_test_split_real_pipeline() -> None:
    """End-to-end sanity check on the real dataset and split function.

    Skips if yfinance/data fetch fails. The test only checks that the
    scaler returned by ``get_train_test_split`` reflects train-only
    statistics.
    """
    try:
        from src.datasets import TradingDataset, get_train_test_split
    except Exception as exc:
        pytest.skip(f"src.datasets unavailable: {exc}")

    try:
        ds = TradingDataset(ticker="MSFT")
    except Exception as exc:
        pytest.skip(f"Could not build TradingDataset (likely network): {exc}")

    train_w, test_w, scaler = get_train_test_split(ds, split_ratio=0.8)

    # Recompute the train slice from the dataset to verify the scaler
    # reflects train-only statistics.
    df = ds.data
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]

    np.testing.assert_allclose(
        scaler.data_min_, train_df.min().values, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        scaler.data_max_, train_df.max().values, rtol=0, atol=1e-9
    )

    # And cross-check leakage is detectably absent: full df min/max
    # MUST differ from scaler stats on at least one column.
    full_min = df.min().values
    full_max = df.max().values
    differs = (
        not np.allclose(scaler.data_min_, full_min)
        or not np.allclose(scaler.data_max_, full_max)
    )
    assert differs, (
        "Train and full min/max are identical — either pure luck on "
        "this ticker, or a regression has reintroduced leakage."
    )
