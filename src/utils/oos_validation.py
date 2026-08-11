"""Out-of-sample walk-forward validation with paired block bootstrap.

Provides a rigorous, leak-free validation engine that evaluates trading models
across expanding time windows and measures statistical significance against
a naive baseline using paired block bootstrapping.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.utils.data_splits import (
    compute_split_boundaries,
    compute_train_val_split,
    get_calibrated_constants,
)

# PRE-REGISTERED SUCCESS CRITERIA (Do not alter per-run to avoid p-hacking)
MIN_BOOTSTRAP_P5: float = 0.0
"""Minimum 5th percentile of the bootstrapped delta profit factor. Must be > 0.0."""

MIN_POOLED_TRADES: int = 300
"""Statistical floor for total OOS trades across all windows to ensure power."""


@dataclass
class WindowResult:
    """Stores performance metrics for a single walk-forward OOS window."""

    start: pd.Timestamp
    end: pd.Timestamp
    cum_ret: float
    vol: float
    threshold: float
    model_pf: float
    naive_pf: float
    delta: float
    model_trade_count: int
    naive_trade_count: int
    skipped_reason: str | None


@dataclass
class WalkForwardResult:
    """Stores aggregated results across all walk-forward windows."""

    windows: list[WindowResult]
    pooled_delta_bootstrap: tuple[float, float]
    pooled_trade_count: int
    passes_gate: bool


def _profit_factor(rets: np.ndarray) -> float:
    """Calculate Profit Factor (gross gains / gross losses).

    Args:
        rets: Array of per-trade returns.

    Returns:
        Profit Factor as a float. Returns NaN if rets is empty.
    """
    if len(rets) == 0:
        return float("nan")
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    return float(gains / max(losses, 1e-9))


def _simulate_trades_with_time(
    df: pd.DataFrame,
    proba_tp: np.ndarray,
    threshold: float,
    tp_multi: float,
    sl_multi: float,
    swing_period: int,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute sequential ternary simulation and return execution timestamps and returns.

    Args:
        df: DataFrame containing 'atr_14', 'close', 'target', and DatetimeIndex.
        proba_tp: Predicted probabilities for Take Profit (class 1).
        threshold: Decision threshold to execute a trade.
        tp_multi: Take Profit ATR multiplier.
        sl_multi: Stop Loss ATR multiplier.
        swing_period: Maximum trade duration in bars.
        fee_rate: Exchange fee rate per side (e.g. 0.001 for 0.1%).
        slippage: Estimated slippage per side (e.g. 0.0005).

    Returns:
        A tuple of (trade_timestamps, trade_returns).
    """
    cost_per_trade = (2.0 * fee_rate) + (2.0 * slippage)

    atr = df["atr_14"].values
    close = df["close"].values
    y_arr = df["target"].values.astype(np.float64)
    times = df.index.values

    n = len(proba_tp)
    trade_times: list[np.datetime64] = []
    rets: list[float] = []
    i = 0

    while i < n:
        if proba_tp[i] >= threshold:
            if y_arr[i] == 1.0:
                rets.append((atr[i] * tp_multi) / close[i] - cost_per_trade)
            elif y_arr[i] == -1.0:
                rets.append(-((atr[i] * sl_multi) / close[i]) - cost_per_trade)
            else:
                exit_idx = min(i + swing_period, n - 1)
                rets.append((close[exit_idx] - close[i]) / close[i] - cost_per_trade)

            trade_times.append(times[i])
            i += swing_period
        else:
            i += 1

    return np.array(trade_times), np.array(rets, dtype=np.float64)


def _find_optimal_threshold(
    df_val: pd.DataFrame,
    proba_val: np.ndarray,
    grid: tuple[float, float, float],
    tp_multi: float,
    sl_multi: float,
    swing_period: int,
    fee_rate: float,
    slippage: float,
    min_val_trades: int,
) -> float:
    """Search for the optimal execution threshold on validation data.

    Args:
        df_val: Validation DataFrame.
        proba_val: Predicted validation probabilities.
        grid: Tuple of (min_thresh, max_thresh, step).
        tp_multi: Take Profit ATR multiplier.
        sl_multi: Stop Loss ATR multiplier.
        swing_period: Maximum trade duration in bars.
        fee_rate: Exchange fee rate per side (e.g. 0.001 for 0.1%).
        slippage: Estimated slippage per side (e.g. 0.0005).
        min_val_trades: Minimum validation trades floor from calibrated constants.

    Returns:
        Optimal threshold float, or -1.0 if no threshold meets trade count floor.
    """
    best_thr = -1.0
    best_net = -1e18
    min_thresh, max_thresh, step = grid

    for thr in np.arange(min_thresh, max_thresh + step / 2.0, step):
        _, rets = _simulate_trades_with_time(
            df_val, proba_val, float(thr), tp_multi, sl_multi, swing_period,
            fee_rate=fee_rate, slippage=slippage,
        )
        if len(rets) < min_val_trades:
            continue
        net = float(rets.sum())
        if net > best_net:
            best_net = net
            best_thr = float(thr)

    return best_thr


def _bootstrap_paired_blocks(
    model_times: list[np.ndarray],
    model_rets: list[np.ndarray],
    naive_times: list[np.ndarray],
    naive_rets: list[np.ndarray],
    window_boundaries: list[tuple[pd.Timestamp, pd.Timestamp]],
    n_blocks: int,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    """Perform paired block bootstrap resampling across all OOS windows.

    Args:
        model_times: List of trade timestamp arrays for the model per window.
        model_rets: List of trade return arrays for the model per window.
        naive_times: List of trade timestamp arrays for naive baseline per window.
        naive_rets: List of trade return arrays for naive baseline per window.
        window_boundaries: List of (start_timestamp, end_timestamp) tuples per window.
        n_blocks: Number of contiguous time blocks to divide each window into.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (5th_percentile_delta, 95th_percentile_delta).
    """
    rng = np.random.default_rng(random_state)
    all_blocks_model: list[np.ndarray] = []
    all_blocks_naive: list[np.ndarray] = []

    for w_idx, (w_start, w_end) in enumerate(window_boundaries):
        block_edges = pd.date_range(start=w_start, end=w_end, periods=n_blocks + 1).values
        m_t, m_r = model_times[w_idx], model_rets[w_idx]
        n_t, n_r = naive_times[w_idx], naive_rets[w_idx]

        for i in range(n_blocks):
            m_mask = (m_t >= block_edges[i]) & (m_t < block_edges[i + 1])
            n_mask = (n_t >= block_edges[i]) & (n_t < block_edges[i + 1])

            all_blocks_model.append(m_r[m_mask])
            all_blocks_naive.append(n_r[n_mask])

    n_total_blocks = len(all_blocks_model)
    if n_total_blocks == 0:
        return float("nan"), float("nan")

    deltas = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.choice(n_total_blocks, size=n_total_blocks, replace=True)

        selected_model_blocks = [all_blocks_model[i] for i in idx if len(all_blocks_model[i]) > 0]
        selected_naive_blocks = [all_blocks_naive[i] for i in idx if len(all_blocks_naive[i]) > 0]

        b_model = np.concatenate(selected_model_blocks) if selected_model_blocks else np.array([])
        b_naive = np.concatenate(selected_naive_blocks) if selected_naive_blocks else np.array([])

        deltas[b] = _profit_factor(b_model) - _profit_factor(b_naive)

    return float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))


def run_walk_forward(
    df_raw: pd.DataFrame,
    symbol: str,
    timeframe: str,
    train_predict_fn: Callable[
        ...,
        tuple[Any, np.ndarray, Callable[[Any, pd.DataFrame], np.ndarray]],
    ],
    tp_multi: float,
    sl_multi: float,
    swing_period: int,
    features: list[str],
    window_months: int,
    step_months: int,
    embargo_days: int | None = None,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
    threshold_grid: tuple[float, float, float] = (0.50, 0.85, 0.01),
    n_bootstrap: int = 1000,
    n_blocks: int = 8,
    random_state: int = 42,
) -> WalkForwardResult:
    """Execute expanding-window walk-forward validation with paired block bootstrap.

    Args:
        df_raw: Raw DataFrame with technicals and targets already computed.
        symbol: Trading symbol identifier (e.g. 'BTC_USDT').
        timeframe: Timeframe identifier (e.g. '4h').
        train_predict_fn: Injected model training/prediction factory function.
            Must accept (X_train, y_train, X_val, y_val, **kwargs) and return
            (model_object, val_predictions, predict_func).
        tp_multi: Take Profit ATR multiplier.
        sl_multi: Stop Loss ATR multiplier.
        swing_period: Trade duration window in bars.
        features: List of feature column names to pass to the model.
        window_months: Duration of each OOS evaluation window in months (REQUIRED).
        step_months: Step size for window advancement in months (REQUIRED).
        embargo_days: Embargo buffer between train and OOS. Defaults to swing_period if None.
        fee_rate: Exchange fee rate per side (e.g. 0.001 for 0.1%). Round-trip
            cost is 2*fee_rate + 2*slippage, matching the production convention.
        slippage: Estimated slippage per side (e.g. 0.0005). Round-trip cost is
            2*fee_rate + 2*slippage, matching the production convention.
        threshold_grid: Tuple of (min, max, step) for optimal threshold search.
            For multiclass_3 models use (0.25, 0.70, 0.01) since softmax of 3
            classes rarely exceeds ~0.39 for the TP class.
        n_bootstrap: Number of bootstrap resamples.
        n_blocks: Number of time blocks per OOS window for paired bootstrap.
        random_state: Random seed for statistical reproducibility.

    Returns:
        WalkForwardResult containing window metrics and global bootstrap pass/fail gate.
    """
    if embargo_days is None:
        embargo_days = swing_period

    cal = get_calibrated_constants(timeframe)
    stat_floor_val_trades = cal["stat_floor_val_trades"]
    start_time = df_raw.index.min() + pd.DateOffset(months=step_months)
    end_time = df_raw.index.max()

    windows_results: list[WindowResult] = []
    model_t_all: list[np.ndarray] = []
    model_r_all: list[np.ndarray] = []
    naive_t_all: list[np.ndarray] = []
    naive_r_all: list[np.ndarray] = []
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    current_start = start_time

    while current_start < end_time:
        current_end = current_start + pd.DateOffset(months=window_months)
        if current_end > end_time:
            break

        prior_cutoff = current_start - pd.Timedelta(days=embargo_days)
        prior_data = df_raw.loc[df_raw.index < prior_cutoff]
        oos_data = df_raw.loc[(df_raw.index >= current_start) & (df_raw.index < current_end)]

        split = compute_train_val_split(
            n_bars=len(prior_data),
            swing_period=swing_period,
            embargo_days=embargo_days,
            bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
            min_val_trades=stat_floor_val_trades,
            max_val_share=cal["max_val_test_share"],
        )

        if split is None or len(oos_data) < swing_period * 2:
            windows_results.append(
                WindowResult(
                    start=current_start,
                    end=current_end,
                    cum_ret=0.0,
                    vol=0.0,
                    threshold=0.0,
                    model_pf=0.0,
                    naive_pf=0.0,
                    delta=0.0,
                    model_trade_count=0,
                    naive_trade_count=0,
                    skipped_reason="insufficient_prior_data",
                )
            )
            current_start += pd.DateOffset(months=step_months)
            continue

        n_train, n_val = split
        train_slice, val_slice, _ = compute_split_boundaries(
            n_train, n_val, 0, embargo_days=embargo_days,
        )
        df_prior_train = prior_data.iloc[train_slice]
        df_prior_val = prior_data.iloc[val_slice]

        model, proba_val, predict_fn = train_predict_fn(
            df_prior_train[features],
            df_prior_train["target"],
            df_prior_val[features],
            df_prior_val["target"],
            random_state=random_state,
        )

        thr = _find_optimal_threshold(
            df_prior_val,
            proba_val,
            threshold_grid,
            tp_multi,
            sl_multi,
            swing_period,
            fee_rate=fee_rate,
            slippage=slippage,
            min_val_trades=stat_floor_val_trades,
        )

        if thr < 0:
            windows_results.append(
                WindowResult(
                    start=current_start,
                    end=current_end,
                    cum_ret=0.0,
                    vol=0.0,
                    threshold=0.0,
                    model_pf=0.0,
                    naive_pf=0.0,
                    delta=0.0,
                    model_trade_count=0,
                    naive_trade_count=0,
                    skipped_reason="threshold_failed",
                )
            )
            current_start += pd.DateOffset(months=step_months)
            continue

        proba_oos = predict_fn(model, oos_data[features])
        proba_naive = np.ones(len(oos_data), dtype=np.float64)

        m_times, m_rets = _simulate_trades_with_time(
            oos_data, proba_oos, thr, tp_multi, sl_multi, swing_period,
            fee_rate=fee_rate, slippage=slippage,
        )
        n_times, n_rets = _simulate_trades_with_time(
            oos_data, proba_naive, 0.0, tp_multi, sl_multi, swing_period,
            fee_rate=fee_rate, slippage=slippage,
        )

        model_pf = _profit_factor(m_rets)
        naive_pf = _profit_factor(n_rets)

        windows_results.append(
            WindowResult(
                start=current_start,
                end=current_end,
                cum_ret=float(m_rets.sum()) if len(m_rets) > 0 else 0.0,
                vol=float(m_rets.std()) if len(m_rets) > 1 else 0.0,
                threshold=thr,
                model_pf=model_pf,
                naive_pf=naive_pf,
                delta=model_pf - naive_pf,
                model_trade_count=len(m_rets),
                naive_trade_count=len(n_rets),
                skipped_reason=None,
            )
        )

        model_t_all.append(m_times)
        model_r_all.append(m_rets)
        naive_t_all.append(n_times)
        naive_r_all.append(n_rets)
        boundaries.append((current_start, current_end))

        current_start += pd.DateOffset(months=step_months)

    pooled_trade_count = sum(len(r) for r in model_r_all)

    if len(model_t_all) > 0:
        p5, p95 = _bootstrap_paired_blocks(
            model_t_all,
            model_r_all,
            naive_t_all,
            naive_r_all,
            boundaries,
            n_blocks,
            n_bootstrap,
            random_state,
        )
    else:
        p5, p95 = float("nan"), float("nan")

    passes = (pooled_trade_count >= MIN_POOLED_TRADES) and (p5 > MIN_BOOTSTRAP_P5)

    return WalkForwardResult(
        windows=windows_results,
        pooled_delta_bootstrap=(p5, p95),
        pooled_trade_count=pooled_trade_count,
        passes_gate=passes,
    )