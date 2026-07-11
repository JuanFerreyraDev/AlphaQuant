"""Tests for src.utils.data_splits — data split utilities."""

from src.utils.data_splits import compute_dynamic_split, compute_min_val_trades


class TestComputeDynamicSplit:
    def test_sufficient_data(self) -> None:
        """Returns valid train/val/test sizes that sum to n_bars."""
        result = compute_dynamic_split(n_bars=1460, swing_period=5)
        assert result is not None
        n_train, n_val, n_test = result
        assert n_train >= 200
        assert n_val >= 60
        assert n_test >= 90
        assert n_train + n_val + n_test == 1460

    def test_insufficient_data(self) -> None:
        """Returns None when there are too few bars."""
        result = compute_dynamic_split(n_bars=100, swing_period=5)
        assert result is None

    def test_split_sizes_sum_to_total(self) -> None:
        """Train, val, and test partitions always sum to n_bars."""
        n_bars = 1300
        result = compute_dynamic_split(n_bars=n_bars, swing_period=7)
        assert result is not None
        assert sum(result) == n_bars


class TestComputeMinValTrades:
    def test_floor(self) -> None:
        """Never returns less than the absolute floor of 12."""
        result = compute_min_val_trades(n_val_bars=10, swing_period=5)
        assert result == 12

    def test_scales_with_bars(self) -> None:
        """More validation bars yields a higher minimum trade count."""
        small = compute_min_val_trades(n_val_bars=100, swing_period=5)
        large = compute_min_val_trades(n_val_bars=500, swing_period=5)
        assert large > small
