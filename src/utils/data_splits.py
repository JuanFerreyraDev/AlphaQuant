"""Utilities for computing temporal train/val/test splits.

These functions are asset-agnostic: they operate on bar counts and
swing periods without knowledge of strategy-specific metrics.
Import them from ``strategy_optimizer`` or any module that needs to
partition time-series data for model selection.

The calibration constants (``DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR``,
``STAT_FLOOR_VAL_TRADES``, etc.) were empirically measured on
``BTC_USDT`` / ``1d``.  When operating on a different timeframe, use
``get_calibrated_constants()`` to retrieve per-timeframe values or
fall back to the most conservative known entry with a warning.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

STAT_FLOOR_VAL_TRADES = 8
STAT_FLOOR_TEST_TRADES = 10
TECH_FLOOR_TRAIN_BARS = 200
DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR = 5

VAL_PCT_FLOOR = 0.15
TEST_PCT_FLOOR = 0.15

MAX_VAL_TEST_SHARE = 0.45

# Per-timeframe calibration.
_TIMEFRAME_CALIBRATIONS: dict[str, dict[str, Any]] = {
    "1d": {
        "bars_per_trade_safety_factor": DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
        "stat_floor_val_trades": STAT_FLOOR_VAL_TRADES,
        "stat_floor_test_trades": STAT_FLOOR_TEST_TRADES,
        "max_val_test_share": MAX_VAL_TEST_SHARE,
    },

    "4h": {
        "bars_per_trade_safety_factor": 2,
        "stat_floor_val_trades": STAT_FLOOR_VAL_TRADES,
        "stat_floor_test_trades": STAT_FLOOR_TEST_TRADES,
        "max_val_test_share": MAX_VAL_TEST_SHARE,
    },

    "1h": {
        "bars_per_trade_safety_factor": 2,
        "stat_floor_val_trades": STAT_FLOOR_VAL_TRADES,
        "stat_floor_test_trades": STAT_FLOOR_TEST_TRADES,
        "max_val_test_share": MAX_VAL_TEST_SHARE,
    },
}


def get_calibrated_constants(timeframe: str = "1d") -> dict[str, Any]:
    """Return the split calibration constants for a given timeframe.

    If the requested timeframe has an explicit calibration entry, those
    values are returned directly.  Otherwise the function falls back to
    the most conservative known entry (currently ``1d``) and logs a
    warning so the operator is aware the constants have not been
    validated for this granularity.

    Args:
        timeframe: Candle interval string (e.g. ``'1d'``, ``'4h'``).

    Returns:
        Dictionary with keys: ``bars_per_trade_safety_factor``,
        ``stat_floor_val_trades``, ``stat_floor_test_trades``,
        ``max_val_test_share``.
    """
    if timeframe in _TIMEFRAME_CALIBRATIONS:
        return dict(_TIMEFRAME_CALIBRATIONS[timeframe])

    logger.warning(
        "No calibrated data-split constants for timeframe '%s'. "
        "Falling back to '1d' defaults (factor=%d, val_floor=%d, "
        "test_floor=%d, max_share=%.2f). These values were measured "
        "on daily BTC_USDT data and may not be appropriate for '%s'. "
        "Recalibrate by comparing requested vs. realized trade counts "
        "in your optimization reports.",
        timeframe,
        DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
        STAT_FLOOR_VAL_TRADES,
        STAT_FLOOR_TEST_TRADES,
        MAX_VAL_TEST_SHARE,
        timeframe,
    )
    return dict(_TIMEFRAME_CALIBRATIONS["1d"])


def _compute_block_floor(
    min_trades: int,
    bars_per_trade_estimate: float,
    allocatable: int,
    pct_floor: float,
) -> int:
    """Compute one block's size based on trade-count and percentage floors.

    Pure calculation with no validation or side effects. Used by both
    compute_dynamic_split and compute_train_val_split to ensure consistent
    block sizing across 2-part and 3-part split strategies.

    Args:
        min_trades: Minimum trade count for this block
            (e.g., STAT_FLOOR_VAL_TRADES).
        bars_per_trade_estimate: swing_period * bars_per_trade_safety_factor.
        allocatable: n_bars - embargo_total (shared pool after embargo).
        pct_floor: VAL_PCT_FLOOR or TEST_PCT_FLOOR (typically 0.15).

    Returns:
        Block size: max(min_trades * bars_per_trade_estimate,
                       int(allocatable * pct_floor))
    """
    bars_needed = min_trades * bars_per_trade_estimate
    return max(bars_needed, int(allocatable * pct_floor))


def _validate_allocation(
    blocks: dict[str, int],
    allocatable: int,
    max_share: float,
) -> bool:
    """Validate that combined block sizes do not exceed a share ceiling.

    Logs a single warning message showing all block sizes together if
    validation fails. This preserves the original behavior of reporting
    the complete picture in one log message, not separately per block.

    Args:
        blocks: Dictionary of {block_name: block_size} to validate.
        allocatable: n_bars - embargo_total (total available for all blocks).
        max_share: Ceiling on sum(blocks.values()) / allocatable
            (typically MAX_VAL_TEST_SHARE = 0.45).

    Returns:
        True if validation passes, False if blocks exceed max_share.
        When False is returned, a warning has been logged.
    """
    total = sum(blocks.values())
    if total > allocatable * max_share:
        # Format block info for logging (order: val, test if present)
        block_items = ", ".join(f"{k}={v}" for k, v in sorted(blocks.items()))
        logger.warning(
            "Trade-count floors would push blocks to %.0f%% of allocatable "
            "bars (cap=%.0f%%): %s allocatable=%d. "
            "This would starve train — skipping instead of shipping an "
            "unbalanced split.",
            total / allocatable * 100, max_share * 100,
            block_items, allocatable,
        )
        return False
    return True


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

    # Compute block sizes using floor logic (consistent with train_val_split).
    n_val = _compute_block_floor(min_val_trades, bars_per_trade_estimate, allocatable, VAL_PCT_FLOOR)
    n_test = _compute_block_floor(min_test_trades, bars_per_trade_estimate, allocatable, TEST_PCT_FLOOR)

    # Validate combined allocation against the share ceiling in a single check.
    if not _validate_allocation(
        {"val": n_val, "test": n_test},
        allocatable,
        max_val_test_share,
    ):
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


def compute_train_val_split(
    n_bars: int,
    swing_period: int,
    embargo_days: int,
    min_val_trades: int = STAT_FLOOR_VAL_TRADES,
    min_train_bars: int = TECH_FLOOR_TRAIN_BARS,
    bars_per_trade_safety_factor: int = DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
    max_val_share: float = MAX_VAL_TEST_SHARE,
) -> tuple[int, int] | None:
    """Compute train/val bar counts for a 2-part split (no test set).

    Guarantees minimum validation trade count, a temporal embargo gap
    between train and val, AND a train-protective split shape. Used when
    the test role is fulfilled by external OOS data (e.g. walk-forward
    validation windows), avoiding the unnecessary train-size penalty of
    reserving a third test block inside the prior-data partition.

    A single embargo gap is reserved (train→val) so that no label
    computed with a forward-looking window can leak across the split
    boundary. The caller is responsible for leaving that gap unused
    when slicing the DataFrame — use ``compute_split_boundaries`` with
    ``n_test=0`` and discard the (empty) returned test slice.

    val takes the LARGER of its trade-count floor (in bars) and its
    percentage floor (VAL_PCT_FLOOR) — this keeps proportions sensible
    on large datasets while still guaranteeing enough trades on small
    ones. But if the trade-count floor alone would push val past
    ``max_val_share`` of the allocatable bars, the split is rejected
    rather than silently starving train.

    Args:
        n_bars: Total number of bars available.
        swing_period: Cooldown bars per trade. Controls trade density.
        embargo_days: Bars to reserve as a gap between train and val.
            Should be >= the swing/lookahead window used to compute
            targets, so labels near a boundary never peek across it.
        min_val_trades: Minimum trades required in the validation set.
        min_train_bars: Hard floor for training bars regardless of
            trade count. Default 200.
        bars_per_trade_safety_factor: Conservative multiplier on
            swing_period used to estimate how many bars are needed per
            realized trade when allocating val bars. Higher values
            assume a lower signal rate and therefore reserve MORE bars
            for val to reach the requested min_val_trades in practice.
        max_val_share: Ceiling on n_val / allocatable.
            Default 0.45, i.e. train keeps at least ~55% of usable bars.

    Returns:
        Tuple ``(n_train, n_val)`` or ``None`` if there is not enough
        data to satisfy all minimums simultaneously.
    """
    bars_per_trade_estimate = swing_period * bars_per_trade_safety_factor

    val_bars_needed = min_val_trades * bars_per_trade_estimate
    embargo_total = embargo_days
    total_needed = min_train_bars + val_bars_needed + embargo_total

    if n_bars < total_needed:
        logger.warning(
            "Insufficient data for train/val split: "
            "n_bars=%d, needed=%d (train=%d, val=%d, embargo=%d) "
            "with swing_period=%d",
            n_bars, total_needed,
            min_train_bars, val_bars_needed, embargo_total,
            swing_period,
        )
        return None

    allocatable = n_bars - embargo_total

    # Compute block size using floor logic (consistent with dynamic_split).
    n_val = _compute_block_floor(min_val_trades, bars_per_trade_estimate, allocatable, VAL_PCT_FLOOR)

    # Validate allocation against the share ceiling.
    if not _validate_allocation(
        {"val": n_val},
        allocatable,
        max_val_share,
    ):
        return None

    n_train = allocatable - n_val

    if n_train < min_train_bars:
        logger.warning(
            "Train/val split would leave only %d train bars (min=%d) "
            "after reserving embargo=%d. Skipping.",
            n_train, min_train_bars, embargo_total,
        )
        return None

    logger.debug(
        "Train/val split — bars: total=%d train=%d (%.0f%%) val=%d (%.0f%%) "
        "embargo=%d (swing_period=%d)",
        n_bars, n_train, n_train / allocatable * 100,
        n_val, n_val / allocatable * 100,
        embargo_days, swing_period,
    )
    return n_train, n_val


def compute_min_val_trades(
    n_val_bars: int,
    swing_period: int,
    bars_per_trade_safety_factor: int = DEFAULT_BARS_PER_TRADE_SAFETY_FACTOR,
    absolute_floor: int = STAT_FLOOR_VAL_TRADES,
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