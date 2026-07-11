"""Utilities for computing temporal train/val/test splits.

These functions are asset-agnostic: they operate on bar counts and
swing periods without knowledge of strategy-specific metrics.
Import them from ``strategy_optimizer`` or any module that needs to
partition time-series data for model selection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

STAT_FLOOR_VAL_TRADES = 20
STAT_FLOOR_TEST_TRADES = 30
TECH_FLOOR_TRAIN_BARS = 200

def compute_dynamic_split(
    n_bars: int,
    swing_period: int,
    min_val_trades: int = STAT_FLOOR_VAL_TRADES,
    min_test_trades: int = STAT_FLOOR_TEST_TRADES,
    min_train_bars: int = TECH_FLOOR_TRAIN_BARS,
) -> tuple[int, int, int] | None:
    """Compute train/val/test bar counts guaranteeing minimum trade counts.

    Works backwards from the minimum number of statistically meaningful
    trades required in each set. The train set receives whatever remains
    after val and test are allocated, subject to ``min_train_bars``.

    The bars-per-trade estimator uses ``swing_period * 3`` as a
    conservative denominator (assumes ~33% signal rate at the chosen
    threshold). This avoids over-optimistic projections that would
    under-allocate bars to val and test.

    Args:
        n_bars: Total number of bars available for the asset.
        swing_period: Cooldown bars per trade. Controls trade density.
        min_val_trades: Minimum trades required in the validation set.
            Default 20 — statistical floor for model selection.
        min_test_trades: Minimum trades required in the test set.
            Default 30 — statistical floor for OOS reporting.
        min_train_bars: Hard floor for training bars regardless of
            trade count. Default 200.

    Returns:
        Tuple ``(n_train, n_val, n_test)`` or ``None`` if the asset
        does not have enough data to satisfy all minimums simultaneously.
        When ``None`` is returned the caller should skip the asset and
        log a warning.
    """
    bars_per_trade_estimate = swing_period * 3

    val_bars_needed   = min_val_trades  * bars_per_trade_estimate
    test_bars_needed  = min_test_trades * bars_per_trade_estimate
    total_needed      = min_train_bars + val_bars_needed + test_bars_needed

    if n_bars < total_needed:
        logger.warning(
            "Insufficient data for dynamic split: "
            "n_bars=%d, needed=%d (train=%d, val=%d, test=%d) "
            "with swing_period=%d",
            n_bars, total_needed,
            min_train_bars, val_bars_needed, test_bars_needed,
            swing_period,
        )
        return None

    # Allocate val and test from the end of the series.
    # Each set takes the maximum of its minimum requirement and its
    # fixed-percentage floor so that large datasets keep sensible proportions.
    n_test  = max(test_bars_needed,  int(n_bars * 0.20))
    n_val   = max(val_bars_needed,   int(n_bars * 0.10))
    n_train = n_bars - n_val - n_test

    if n_train < min_train_bars:
        logger.warning(
            "Dynamic split would leave only %d train bars (min=%d). "
            "Skipping asset.",
            n_train, min_train_bars,
        )
        return None

    logger.debug(
        "Dynamic split — bars: total=%d train=%d val=%d test=%d "
        "(swing_period=%d)",
        n_bars, n_train, n_val, n_test, swing_period,
    )
    return n_train, n_val, n_test


def compute_min_val_trades(n_val_bars: int, swing_period: int) -> int:
    """Compute the minimum acceptable trade count for a validation set.

    Uses the same conservative 33% signal-rate assumption as
    ``compute_dynamic_split``. The result is 50% of the expected trade
    count — a strategy must achieve at least half its theoretical
    maximum to be considered statistically meaningful.

    The absolute floor of 12 is kept regardless of bar count to
    prevent degenerate cases on very short validation windows.

    Args:
        n_val_bars: Number of bars in the validation set.
        swing_period: Cooldown bars per trade.

    Returns:
        Minimum trade count for the kill-switch filter in
        ``_calculate_production_score``.
    """
    max_possible    = n_val_bars / swing_period
    expected_trades = max_possible * 0.33          # conservative signal rate
    dynamic_min     = int(expected_trades * 0.50)  # require 50 % of expected
    result          = max(12, dynamic_min)

    logger.debug(
        "Min val trades — n_val_bars=%d swing_period=%d "
        "max_possible=%.1f expected=%.1f dynamic_min=%d result=%d",
        n_val_bars, swing_period,
        max_possible, expected_trades, dynamic_min, result,
    )
    return result
