"""Utilities for computing temporal train/val/test splits.

These functions are asset-agnostic: they operate on bar counts and
swing periods without knowledge of strategy-specific metrics.
Import them from ``strategy_optimizer`` or any module that needs to
partition time-series data for model selection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

STAT_FLOOR_VAL_TRADES = 8
STAT_FLOOR_TEST_TRADES = 10
TECH_FLOOR_TRAIN_BARS = 200
DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR = 5

VAL_PCT_FLOOR = 0.15
TEST_PCT_FLOOR = 0.15

MAX_VAL_TEST_SHARE = 0.45


def compute_dynamic_split(
    n_bars: int,
    swing_period: int,
    embargo_days: int,
    min_val_trades: int = STAT_FLOOR_VAL_TRADES,
    min_test_trades: int = STAT_FLOOR_TEST_TRADES,
    min_train_bars: int = TECH_FLOOR_TRAIN_BARS,
    bars_per_trade_safety_factor: int = DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
    max_val_test_share: float = MAX_VAL_TEST_SHARE,
) -> tuple[int, int, int] | None:
    """Compute train/val/test bar counts guaranteeing minimum trade counts,
    a temporal embargo gap between blocks, AND a train-protective split shape.

    Two embargo gaps are reserved (train→val and val→test) so that no
    label computed with a forward-looking window (see ``compute_target``,
    which looks ahead up to ``swing_days``) can leak across a split
    boundary. The caller is responsible for leaving those gaps unused
    when slicing the DataFrame — see ``compute_split_boundaries``.

    val and test each take the LARGER of their trade-count floor (in
    bars) and their percentage floor (``VAL_PCT_FLOOR``/``TEST_PCT_FLOOR``)
    — this keeps proportions sensible on large datasets while still
    guaranteeing enough trades on small ones. But if the trade-count
    floors alone would push val+test past ``max_val_test_share`` of the
    allocatable bars, the split is rejected rather than silently starving
    train: a statistically "solid" val/test built on a starved train set
    just produces a badly-fit model, which defeats the purpose.

    Args:
        n_bars: Total number of bars available for the asset.
        swing_period: Cooldown bars per trade. Controls trade density.
            Callers should pass the LEAST frequent (i.e. largest) swing
            period in their grid here, so the window is sized for the
            worst case rather than the best case.
        embargo_days: Bars to reserve as a gap on each side of val
            (i.e. 2 * embargo_days is removed from the allocatable
            total). Should be >= the swing/lookahead window used to
            compute targets, so labels near a boundary never peek
            across it.
        min_val_trades: Minimum trades required in the validation set.
        min_test_trades: Minimum trades required in the test set.
        min_train_bars: Hard floor for training bars regardless of
            trade count. Default 200.
        bars_per_trade_safety_factor: Conservative multiplier on
            swing_period used to estimate how many bars are needed per
            realized trade when allocating val/test bars. Higher values
            assume a lower signal rate (more bars needed per trade) and
            therefore reserve MORE bars for val/test to reach the
            requested min_val_trades/min_test_trades in practice.
            Recalibrate this per-asset/timeframe by comparing requested
            vs. actually realized trade counts in your reports — see
            the empirical calibration done for BTC_USDT/1d (~4-5%
            observed signal rate, factor~5-6 as of this writing).
        max_val_test_share: Ceiling on (n_val + n_test) / allocatable.
            Default 0.45, i.e. train keeps at least ~55% of usable bars.

    Returns:
        Tuple ``(n_train, n_val, n_test)`` or ``None`` if the asset
        does not have enough data to satisfy all minimums simultaneously
        — either because there aren't enough bars at all, or because
        satisfying the trade-count floors would violate the train-share
        protection. When ``None`` is returned the caller should skip the
        asset and log a warning.
    """
    bars_per_trade_estimate = swing_period * bars_per_trade_safety_factor

    val_bars_needed   = min_val_trades  * bars_per_trade_estimate
    test_bars_needed  = min_test_trades * bars_per_trade_estimate
    embargo_total      = 2 * embargo_days
    total_needed        = min_train_bars + val_bars_needed + test_bars_needed + embargo_total

    if n_bars < total_needed:
        logger.warning(
            "Insufficient data for dynamic split: "
            "n_bars=%d, needed=%d (train=%d, val=%d, test=%d, embargo=%d) "
            "with swing_period=%d",
            n_bars, total_needed,
            min_train_bars, val_bars_needed, test_bars_needed, embargo_total,
            swing_period,
        )
        return None

    # Bars actually available to distribute among train/val/test once
    # both embargo gaps are set aside.
    allocatable = n_bars - embargo_total

    # Allocate val and test from the end of the allocatable range.
    # Each set takes the maximum of its minimum requirement and its
    # fixed-percentage floor so that large datasets keep sensible proportions.
    n_test  = max(test_bars_needed,  int(allocatable * TEST_PCT_FLOOR))
    n_val   = max(val_bars_needed,   int(allocatable * VAL_PCT_FLOOR))

    if (n_val + n_test) > allocatable * max_val_test_share:
        logger.warning(
            "Trade-count floors would push val+test to %.0f%% of "
            "allocatable bars (cap=%.0f%%): val=%d test=%d allocatable=%d "
            "(swing_period=%d, min_val_trades=%d, min_test_trades=%d). "
            "This would starve train — skipping asset instead of shipping "
            "an unbalanced split. Lower the trade-count floors, gather "
            "more history, or use a finer timeframe.",
            (n_val + n_test) / allocatable * 100, max_val_test_share * 100,
            n_val, n_test, allocatable,
            swing_period, min_val_trades, min_test_trades,
        )
        return None

    n_train = allocatable - n_val - n_test

    if n_train < min_train_bars:
        logger.warning(
            "Dynamic split would leave only %d train bars (min=%d) "
            "after reserving embargo=%d. Skipping asset.",
            n_train, min_train_bars, embargo_total,
        )
        return None

    logger.debug(
        "Dynamic split — bars: total=%d train=%d (%.0f%%) val=%d (%.0f%%) "
        "test=%d (%.0f%%) embargo=%d each side (swing_period=%d)",
        n_bars, n_train, n_train / allocatable * 100,
        n_val, n_val / allocatable * 100,
        n_test, n_test / allocatable * 100,
        embargo_days, swing_period,
    )
    return n_train, n_val, n_test


def compute_split_boundaries(
    n_train: int,
    n_val: int,
    n_test: int,
    embargo_days: int,
) -> tuple[slice, slice, slice]:
    """Turn split sizes into non-overlapping slices with embargo gaps.

    Args:
        n_train: Training bar count, from ``compute_dynamic_split``.
        n_val: Validation bar count, from ``compute_dynamic_split``.
        n_test: Test bar count, from ``compute_dynamic_split``.
        embargo_days: Same value passed to ``compute_dynamic_split`` —
            gap reserved on each side of the validation block.

    Returns:
        Tuple of three ``slice`` objects ``(train_slice, val_slice,
        test_slice)`` ready to use with ``df.iloc[...]``. The bars
        inside the embargo gaps are intentionally excluded from all
        three slices.
    """
    val_start = n_train + embargo_days
    val_end = val_start + n_val
    test_start = val_end + embargo_days
    test_end = test_start + n_test

    return (
        slice(0, n_train),
        slice(val_start, val_end),
        slice(test_start, test_end),
    )


def compute_min_val_trades(
    n_val_bars: int,
    swing_period: int,
    bars_per_trade_safety_factor: int = DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
    absolute_floor: int = 12,
) -> int:
    """Compute the minimum acceptable trade count for a validation set.

    Uses the SAME signal-rate assumption as ``compute_dynamic_split``
    (``bars_per_trade_safety_factor``), so the kill-switch threshold and
    the bar-allocation sizing stay consistent with each other instead of
    each embedding their own guess at the market's signal rate. The
    result is 50% of the theoretical expected trade count — a strategy
    must achieve at least half its theoretical maximum to be considered
    statistically meaningful.

    Args:
        n_val_bars: Number of bars in the validation set.
        swing_period: Cooldown bars per trade. Pass the SAME
            swing_period used to size the split (i.e. the worst-case
            one), not the individual combo's own swing_period — this
            keeps the kill-switch threshold uniform across every combo
            evaluated on this val window, matching how the window
            itself was sized.
        bars_per_trade_safety_factor: Same meaning as in
            ``compute_dynamic_split`` — keep these in sync.
        absolute_floor: Hard floor regardless of bar count, to prevent
            degenerate cases on very short validation windows.

    Returns:
        Minimum trade count for the kill-switch filter in
        ``_calculate_production_score``.
    """
    bars_per_trade_estimate = swing_period * bars_per_trade_safety_factor
    expected_trades = n_val_bars / bars_per_trade_estimate
    dynamic_min = int(expected_trades * 0.50)
    result = max(absolute_floor, dynamic_min)

    logger.debug(
        "Min val trades — n_val_bars=%d swing_period=%d factor=%d "
        "expected=%.1f dynamic_min=%d result=%d",
        n_val_bars, swing_period, bars_per_trade_safety_factor,
        expected_trades, dynamic_min, result,
    )
    return result