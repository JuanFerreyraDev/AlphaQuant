"""Tests for src.utils.data_splits — data split utilities."""

import pytest

from src.utils.data_splits import (
    DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
    STAT_FLOOR_VAL_TRADES,
    compute_dynamic_split,
    compute_min_val_trades,
    compute_split_boundaries,
    get_calibrated_constants,
)


class TestGetCalibratedConstants:
    def test_1d_returns_default_constants(self) -> None:
        cal = get_calibrated_constants("1d")
        assert cal["bars_per_trade_safety_factor"] == DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR
        assert cal["stat_floor_val_trades"] == STAT_FLOOR_VAL_TRADES

    def test_4h_has_its_own_calibration(self) -> None:
        """4h must be explicitly calibrated, not silently inherit 1d's
        factor=5 (which assumes ~3x fewer trades than observed on real
        BTC_USDT/4h data)."""
        cal = get_calibrated_constants("4h")
        assert cal["bars_per_trade_safety_factor"] == 2
        assert cal != get_calibrated_constants("1d")

    def test_unknown_timeframe_falls_back_to_1d_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            cal = get_calibrated_constants("15m")
        assert cal == get_calibrated_constants("1d")
        assert "No calibrated data-split constants" in caplog.text


class TestComputeDynamicSplit:
    def test_sufficient_data(self) -> None:
        """Ample data yields a split close to the 70/15/15 target shape
        (trade-count floors don't bind; percentage floors do)."""
        result = compute_dynamic_split(n_bars=5000, swing_period=5, embargo_days=5)
        assert result is not None
        n_train, n_val, n_test = result
        assert n_train >= 200
        assert n_val >= 60
        assert n_test >= 90
        # sizes sum to n_bars minus the two embargo gaps
        assert n_train + n_val + n_test == 5000 - 2 * 5
        # with ample data, shape lands close to 70/15/15
        allocatable = n_train + n_val + n_test
        assert n_train / allocatable == pytest.approx(0.70, abs=0.01)
        assert n_val / allocatable == pytest.approx(0.15, abs=0.01)
        assert n_test / allocatable == pytest.approx(0.15, abs=0.01)

    def test_insufficient_data(self) -> None:
        """Returns None when there are too few bars for the trade-count floors."""
        result = compute_dynamic_split(n_bars=100, swing_period=5, embargo_days=5)
        assert result is None

    def test_split_sizes_sum_to_allocatable(self) -> None:
        """Train, val, and test partitions sum to n_bars minus embargo gaps."""
        n_bars = 5000
        embargo_days = 7
        result = compute_dynamic_split(
            n_bars=n_bars, swing_period=7, embargo_days=embargo_days
        )
        assert result is not None
        assert sum(result) == n_bars - 2 * embargo_days

    def test_larger_embargo_shrinks_allocatable_bars(self) -> None:
        """A bigger embargo leaves less room for train/val/test, all else equal."""
        small_embargo = compute_dynamic_split(n_bars=5000, swing_period=5, embargo_days=2)
        large_embargo = compute_dynamic_split(n_bars=5000, swing_period=5, embargo_days=30)
        assert small_embargo is not None
        assert large_embargo is not None
        assert sum(large_embargo) < sum(small_embargo)

    def test_embargo_too_large_returns_none(self) -> None:
        """An embargo that consumes most of the series fails the minimums."""
        result = compute_dynamic_split(n_bars=300, swing_period=5, embargo_days=100)
        assert result is None

    def test_high_trade_floors_starve_train_returns_none(self) -> None:
        """When trade-count floors would push val+test past max_val_test_share,
        the split is rejected instead of silently shipping a starved train set.

        This is the exact scenario that motivated the cap: with these floors
        and this little data, val+test would consume 60% of allocatable bars
        (vs. a 35% cap), which would leave train with a proportionally tiny,
        statistically weak slice of the data.

        Explicit floors (12/15) are passed here rather than relying on the
        module defaults (8/10 as of the timeframe-calibration fix), since the
        module defaults alone are no longer high enough to trip the cap at
        this bar count — the scenario itself (floors starving train) is
        still valid and worth covering independent of what the defaults are.
        """
        result = compute_dynamic_split(
            n_bars=1460, swing_period=5, embargo_days=5,
            min_val_trades=12, min_test_trades=15,
        )
        assert result is None

    def test_lower_trade_floors_avoid_starving_train(self) -> None:
        """The same modest dataset succeeds once trade-count floors are
        lowered enough not to trip the val+test share cap."""
        result = compute_dynamic_split(
            n_bars=1460, swing_period=5, embargo_days=5,
            min_val_trades=5, min_test_trades=8,
        )
        assert result is not None
        n_train, n_val, n_test = result
        assert n_train >= 200


class TestComputeSplitBoundaries:
    def test_slices_are_contiguous_with_gaps(self) -> None:
        """Val and test slices start strictly after the embargo gap."""
        n_train, n_val, n_test = 700, 100, 150
        embargo_days = 5

        train_slice, val_slice, test_slice = compute_split_boundaries(
            n_train, n_val, n_test, embargo_days=embargo_days
        )

        assert train_slice == slice(0, 700)
        assert val_slice.start == 700 + embargo_days
        assert val_slice.stop == val_slice.start + n_val
        assert test_slice.start == val_slice.stop + embargo_days
        assert test_slice.stop == test_slice.start + n_test

    def test_gap_bars_are_excluded_from_all_slices(self) -> None:
        """Bars inside an embargo gap belong to none of the three slices."""
        n_train, n_val, n_test = 70, 10, 15
        embargo_days = 5

        train_slice, val_slice, test_slice = compute_split_boundaries(
            n_train, n_val, n_test, embargo_days=embargo_days
        )

        gap_1 = set(range(train_slice.stop, val_slice.start))
        gap_2 = set(range(val_slice.stop, test_slice.start))

        assert len(gap_1) == embargo_days
        assert len(gap_2) == embargo_days

        train_bars = set(range(train_slice.start, train_slice.stop))
        val_bars = set(range(val_slice.start, val_slice.stop))
        test_bars = set(range(test_slice.start, test_slice.stop))

        assert train_bars.isdisjoint(gap_1)
        assert val_bars.isdisjoint(gap_1)
        assert val_bars.isdisjoint(gap_2)
        assert test_bars.isdisjoint(gap_2)

    def test_zero_embargo_is_contiguous(self) -> None:
        """embargo_days=0 reproduces a plain contiguous split (no gap)."""
        train_slice, val_slice, test_slice = compute_split_boundaries(
            70, 10, 15, embargo_days=0
        )
        assert train_slice.stop == val_slice.start
        assert val_slice.stop == test_slice.start


class TestComputeMinValTrades:
    def test_floor(self) -> None:
        """Never returns less than the absolute floor of 8
        (STAT_FLOOR_VAL_TRADES)."""
        result = compute_min_val_trades(n_val_bars=10, swing_period=5)
        assert result == 8

    def test_scales_with_bars(self) -> None:
        """More validation bars yields a higher minimum trade count,
        once past the point where the absolute floor stops dominating."""
        small = compute_min_val_trades(n_val_bars=100, swing_period=5)
        large = compute_min_val_trades(n_val_bars=2000, swing_period=5)
        assert large > small

    def test_uses_same_signal_rate_as_dynamic_split(self) -> None:
        """A higher bars_per_trade_safety_factor (lower assumed signal
        rate) demands MORE bars per trade, so for the same n_val_bars it
        yields a smaller (or equal) minimum trade count — consistent
        with compute_dynamic_split's own use of this factor."""
        optimistic = compute_min_val_trades(
            n_val_bars=2000, swing_period=5, bars_per_trade_safety_factor=3,
        )
        conservative = compute_min_val_trades(
            n_val_bars=2000, swing_period=5, bars_per_trade_safety_factor=10,
        )
        assert conservative <= optimistic