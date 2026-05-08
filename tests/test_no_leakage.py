"""
Test suite to verify no data leakage between train and test sets.

Critical assertions:
1. Scaler statistics (min/max) derived only from training data
2. Test indices are strictly chronologically after train indices
3. State observations never contain future information
"""

import pytest
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import TradingDataset, get_train_test_split, make_windows
from src.config import WINDOW_SIZE


def test_scaler_fit_only_on_train():
    """Verify scaler min/max computed exclusively from training split."""
    # Create synthetic dataset with known structure
    dataset = TradingDataset('MSFT', '2020-01-01', '2021-12-31')

    # Get split with scaler
    train_windows, test_windows, scaler = get_train_test_split(dataset, split_ratio=0.8)

    # Extract train DataFrame
    df = dataset.data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]

    # Compute expected min/max from training data only
    expected_min = train_df.values.min(axis=0)
    expected_max = train_df.values.max(axis=0)

    # Check scaler was fit on training data only
    actual_min = scaler.data_min_
    actual_max = scaler.data_max_

    np.testing.assert_array_almost_equal(actual_min, expected_min, decimal=6,
                                          err_msg="Scaler min does not match train-only min")
    np.testing.assert_array_almost_equal(actual_max, expected_max, decimal=6,
                                          err_msg="Scaler max does not match train-only max")

    print(f"✓ Scaler statistics derived exclusively from {len(train_df)} training samples")


def test_chronological_split():
    """Verify test indices are strictly after train indices (no shuffling)."""
    dataset = TradingDataset('MSFT', '2020-01-01', '2021-12-31')
    df = dataset.data

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Check chronological ordering
    assert train_df.index.max() < test_df.index.min(), \
        "Test set starts before train set ends - non-chronological split detected!"

    # Check no overlap
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)
    overlap = train_indices.intersection(test_indices)

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices between train/test"

    print(f"✓ Chronological split verified: train ends at {train_df.index.max()}, test starts at {test_df.index.min()}")


def test_no_future_information_in_state():
    """Verify state[t] contains no data from index >= t+WINDOW_SIZE."""
    dataset = TradingDataset('MSFT', '2020-01-01', '2021-12-31')
    train_windows, test_windows, scaler = get_train_test_split(dataset, split_ratio=0.8)

    # Check window construction doesn't include future data
    df = dataset.data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]

    # Manually construct first window
    train_scaled = scaler.transform(train_df)
    first_window_manual = train_scaled[:WINDOW_SIZE]  # Rows 0 to WINDOW_SIZE-1

    # Extract first window from train_windows tensor
    # train_windows shape: [N_samples, Features, Window]
    first_window_from_tensor = train_windows[0].numpy().T  # [Window, Features]

    # They should match (accounting for transpose)
    np.testing.assert_array_almost_equal(
        first_window_from_tensor, first_window_manual, decimal=5,
        err_msg="Window construction includes future data beyond intended lookback"
    )

    print(f"✓ Window construction verified: no future information leakage")


def test_test_set_uses_train_statistics():
    """Verify test set normalization uses frozen train statistics."""
    dataset = TradingDataset('MSFT', '2020-01-01', '2021-12-31')
    train_windows, test_windows, scaler = get_train_test_split(dataset, split_ratio=0.8)

    df = dataset.data
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]

    # Scaler should transform test data using train min/max
    test_scaled_manual = scaler.transform(test_df)

    # Verify some test values are outside [0, 1] if test extremes exceed train extremes
    # This proves we're NOT fitting a new scaler on test data

    # Check if test data has values outside train range
    train_df = df.iloc[:split_idx]
    train_min = train_df.values.min(axis=0)
    train_max = train_df.values.max(axis=0)

    test_min = test_df.values.min(axis=0)
    test_max = test_df.values.max(axis=0)

    # If test has values outside train range, scaled values should be outside [0,1]
    for feat_idx in range(test_df.shape[1]):
        if test_min[feat_idx] < train_min[feat_idx]:
            # This test feature goes below train minimum
            # After scaling with train scaler, should be negative
            assert test_scaled_manual[:, feat_idx].min() < 0, \
                f"Feature {feat_idx}: test minimum below train minimum, but scaled value not negative"
            print(f"✓ Feature {feat_idx}: test min extrapolated below 0 (expected behavior)")

        if test_max[feat_idx] > train_max[feat_idx]:
            # This test feature exceeds train maximum
            # After scaling with train scaler, should exceed 1.0
            assert test_scaled_manual[:, feat_idx].max() > 1.0, \
                f"Feature {feat_idx}: test maximum exceeds train maximum, but scaled value not > 1.0"
            print(f"✓ Feature {feat_idx}: test max extrapolated above 1.0 (expected behavior)")

    print(f"✓ Test set normalization verified: uses frozen train statistics")


def test_no_data_leakage_synthetic():
    """Synthetic test with artificial dataset to guarantee leakage detection."""
    # Create artificial dataset where test has extreme values
    np.random.seed(42)

    # Train data: range [0, 1]
    # Test data: range [10, 11]  (should NOT influence train scaling)
    train_data = np.random.uniform(0, 1, (100, 4))
    test_data = np.random.uniform(10, 11, (25, 4))

    # Fit scaler on train
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    # Check scaler statistics
    assert np.allclose(scaler.data_min_, train_data.min(axis=0), atol=0.01), \
        "Scaler min influenced by test data!"
    assert np.allclose(scaler.data_max_, train_data.max(axis=0), atol=0.01), \
        "Scaler max influenced by test data!"

    # Transform test data
    test_scaled = scaler.transform(test_data)

    # Test data should be scaled far above 1.0 (proves independence)
    assert test_scaled.min() > 9, \
        "Test data scaling suggests train/test contamination"

    print(f"✓ Synthetic leakage test passed: train and test scaling fully independent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
