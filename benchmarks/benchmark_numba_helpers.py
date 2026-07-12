#!/usr/bin/env python3
"""Temporary parity + benchmark script for Numba-refactored helpers.

Run: .venv/bin/python benchmark_numba_helpers.py
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from src.utils.helpers import (
    _compute_threshold_loop,
    _resolve_targets_numba,
    _simulate_fitness_sequential,
    compute_target,
    find_optimal_threshold,
    fitness_score,
)


# ---------------------------------------------------------------------------
# Reference (pre-Numba) implementations for parity checks
# ---------------------------------------------------------------------------

def _compute_target_lower_tf_pandas(
    df: pd.DataFrame,
    swing_days: int,
    atr_tp_multi: float,
    atr_sl_multi: float,
    lower_tf_df: pd.DataFrame,
) -> np.ndarray:
    tp_price = df["close"] + (df["atr_14"] * atr_tp_multi)
    sl_price = df["close"] - (df["atr_14"] * atr_sl_multi)
    targets: list[int] = []
    for i, (date, _row) in enumerate(df.iterrows()):
        window_start = date + pd.Timedelta(days=1)
        window_end = date + pd.Timedelta(days=swing_days)
        ltf_window = lower_tf_df.loc[
            (lower_tf_df.index >= window_start) & (lower_tf_df.index <= window_end)
        ]
        label = 0
        for _, ltf_row in ltf_window.iterrows():
            if ltf_row["low"] <= sl_price.iloc[i]:
                label = 0
                break
            if ltf_row["high"] >= tp_price.iloc[i]:
                label = 1
                break
        targets.append(label)
    return np.array(targets, dtype=np.int32)


def _find_optimal_threshold_pandas(
    proba: np.ndarray,
    y_arr: np.ndarray,
    atr: np.ndarray,
    close: np.ndarray,
    tp_val: float,
    sl_val: float,
    cost_per_trade: float,
    swing_period: int,
) -> tuple[float, float]:
    n = len(proba)
    min_trades = max(5, int(n / (swing_period + 15)))
    best_threshold = 0.50
    best_profit = -np.inf
    for thresh in np.arange(0.50, 0.85, 0.01):
        profit = 0.0
        trade_count = 0
        i = 0
        while i < n:
            if proba[i] >= thresh:
                if y_arr[i] == 1:
                    profit += (atr[i] * tp_val) / close[i] - cost_per_trade
                else:
                    profit -= (atr[i] * sl_val) / close[i] + cost_per_trade
                trade_count += 1
                i += swing_period
            else:
                i += 1
        if trade_count < min_trades:
            continue
        if profit > best_profit:
            best_profit = profit
            best_threshold = float(thresh)
    if best_profit == -np.inf:
        return -1.0, 0.0
    return best_threshold, best_profit


def _fitness_score_pandas(
    proba: np.ndarray,
    y_arr: np.ndarray,  
    atr: np.ndarray,
    close: np.ndarray,
    threshold: float,
    tp_val: float,
    sl_val: float,
    cost_per_trade: float,
    swing_period: int,
) -> tuple[float, dict]:
    n = len(proba)
    trade_returns: list[float] = []
    i = 0
    while i < n:
        if proba[i] >= threshold:
            if y_arr[i] == 1:
                ret = (atr[i] * tp_val) / close[i] - cost_per_trade
            else:
                ret = -((atr[i] * sl_val) / close[i]) - cost_per_trade
            trade_returns.append(ret)
            i += swing_period
        else:
            i += 1
    trade_count = len(trade_returns)
    rets_arr = np.array(trade_returns)
    gross_profit = float(rets_arr[rets_arr > 0].sum()) if trade_count else 0.0
    gross_loss = float(abs(rets_arr[rets_arr < 0].sum())) if trade_count else 0.0
    pf = gross_profit / max(gross_loss, 1e-9)
    if trade_count:
        account_equity = np.cumprod(1.0 + rets_arr)
        running_max = np.maximum.accumulate(account_equity)
        mdd = float(((running_max - account_equity) / running_max).max())
    else:
        mdd = 0.0
    metrics = {
        "profit_factor": round(pf, 4),
        "max_drawdown": round(mdd, 4),
        "trade_count": trade_count,
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
    }
    if pf <= 1.0:
        return -999.0, metrics
    capped_pf = min(pf, 10.0)
    safe_dd = max(mdd, 0.01)
    max_possible_trades = max(1, n / swing_period)
    target_min_trades = max(12, int(max_possible_trades * 0.15))
    frequency_penalty = min(1.0, trade_count / target_min_trades)
    score = ((capped_pf - 1.0) / safe_dd) * math.log(max(trade_count, 1)) * frequency_penalty
    return float(score), metrics


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_synthetic_daily(n_days: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n_days))
    close = np.maximum(close, 10.0)
    atr = rng.uniform(1.0, 5.0, n_days)
    high = close + rng.uniform(0.1, 2.0, n_days)
    low = close - rng.uniform(0.1, 2.0, n_days)
    return pd.DataFrame(
        {"close": close, "high": high, "low": low, "atr_14": atr},
        index=dates,
    )


def make_synthetic_lower_tf(daily_df: pd.DataFrame, bars_per_day: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for day, row in daily_df.iterrows():
        for h in range(bars_per_day):
            ts = day + pd.Timedelta(hours=h + 1)
            noise_h = rng.uniform(-0.5, 1.5)
            noise_l = rng.uniform(-1.5, 0.5)
            rows.append({"high": row["high"] + noise_h, "low": row["low"] + noise_l})
    idx = pd.DatetimeIndex(
        [daily_df.index[i] + pd.Timedelta(hours=h + 1)
         for i in range(len(daily_df))
         for h in range(bars_per_day)]
    )
    return pd.DataFrame(rows, index=idx)


def make_trading_arrays(n: int = 500, seed: int = 99) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0.0, 1.0, n)
    y_arr = (rng.random(n) > 0.4).astype(np.int64)
    atr = rng.uniform(1.0, 5.0, n)
    close = rng.uniform(80.0, 200.0, n)
    return proba, y_arr, atr, close


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------

def check_compute_target_parity() -> None:
    print("\n=== Paridad: compute_target (lower_tf) ===")
    df = make_synthetic_daily(200)
    ltf = make_synthetic_lower_tf(df, bars_per_day=24)
    swing = 5
    tp_m, sl_m = 1.5, 1.0

    ref = _compute_target_lower_tf_pandas(df, swing, tp_m, sl_m, ltf)
    out = compute_target(
        df.copy(), swing_days=swing, atr_tp_multi=tp_m, atr_sl_multi=sl_m, lower_tf_df=ltf
    )["target"].to_numpy(dtype=np.int32)

    assert np.array_equal(ref, out), f"Mismatch count: {(ref != out).sum()}"
    print("  OK — 100% match on 200 daily rows × 24 hourly bars")


def check_find_optimal_threshold_parity() -> None:
    print("\n=== Paridad: find_optimal_threshold ===")
    proba, y_arr, atr, close = make_trading_arrays(500)
    tp_val, sl_val = 2.0, 1.0
    cost = 0.003
    swing = 5

    ref = _find_optimal_threshold_pandas(proba, y_arr, atr, close, tp_val, sl_val, cost, swing)
    n = len(proba)
    min_trades = max(5, int(n / (swing + 15)))
    numba = _compute_threshold_loop(
        proba, y_arr, atr, close, tp_val, sl_val, cost, swing, n, min_trades
    )
    if numba[1] == -np.inf:
        numba = (-1.0, 0.0)

    assert math.isclose(ref[0], numba[0], abs_tol=1e-9)
    assert math.isclose(ref[1], numba[1], abs_tol=1e-12)
    print(f"  OK — threshold={ref[0]:.2f}, profit={ref[1]:.6f}")


def check_fitness_score_parity() -> None:
    print("\n=== Paridad: fitness_score ===")
    proba, y_arr, atr, close = make_trading_arrays(500)
    tp_val, sl_val = 2.0, 1.0
    cost = 0.003
    swing = 5
    threshold = 0.55

    score_ref, metrics_ref = _fitness_score_pandas(
        proba, y_arr, atr, close, threshold, tp_val, sl_val, cost, swing
    )
    tc, gp, gl, mdd = _simulate_fitness_sequential(
        proba, y_arr, atr, close, threshold, tp_val, sl_val, cost, swing, len(proba)
    )
    pf = gp / max(gl, 1e-9)
    metrics_numba = {
        "profit_factor": round(pf, 4),
        "max_drawdown": round(mdd, 4),
        "trade_count": tc,
        "gross_profit": round(gp, 6),
        "gross_loss": round(gl, 6),
    }
    if pf <= 1.0:
        score_numba = -999.0
    else:
        capped_pf = min(pf, 10.0)
        safe_dd = max(mdd, 0.01)
        max_possible = max(1, len(proba) / swing)
        target_min = max(12, int(max_possible * 0.15))
        freq_pen = min(1.0, tc / target_min)
        score_numba = ((capped_pf - 1.0) / safe_dd) * math.log(max(tc, 1)) * freq_pen

    assert metrics_ref["trade_count"] == metrics_numba["trade_count"]
    assert math.isclose(metrics_ref["gross_profit"], metrics_numba["gross_profit"], abs_tol=1e-9)
    assert math.isclose(metrics_ref["gross_loss"], metrics_numba["gross_loss"], abs_tol=1e-9)
    np.testing.assert_allclose(
        metrics_ref["max_drawdown"], metrics_numba["max_drawdown"], atol=1e-6
    )
    np.testing.assert_allclose(score_ref, score_numba, atol=1e-6)
    print(f"  OK — score={score_ref:.6f}, MDD ref={metrics_ref['max_drawdown']}, numba={metrics_numba['max_drawdown']}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _timeit(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run_benchmarks() -> None:
    print("\n" + "=" * 60)
    print("BENCHMARK PERFORMANCE")
    print("=" * 60)

    # --- compute_target ---
    n_days = 2000
    df = make_synthetic_daily(n_days)
    ltf = make_synthetic_lower_tf(df, bars_per_day=24)
    swing, tp_m, sl_m = 5, 1.5, 1.0

    def numba_target():
        compute_target(
            df.copy(), swing_days=swing, atr_tp_multi=tp_m,
            atr_sl_multi=sl_m, lower_tf_df=ltf,
        )

    def pandas_target():
        _compute_target_lower_tf_pandas(df, swing, tp_m, sl_m, ltf)

    # Warm-up Numba JIT
    numba_target()
    t_numba = _timeit(numba_target, repeats=5)
    t_pandas = _timeit(pandas_target, repeats=3)
    print(f"\ncompute_target (lower_tf, {n_days} days × 24 hours):")
    print(f"  Pandas iterrows : {t_pandas:.4f}s")
    print(f"  Numba JIT       : {t_numba:.4f}s")
    print(f"  Acceleration    : {t_pandas / t_numba:.1f}x faster")

    # --- find_optimal_threshold ---
    proba, y_arr, atr, close = make_trading_arrays(2000)
    tp_val, sl_val, cost, swing = 2.0, 1.0, 0.003, 5
    n = len(proba)
    min_trades = max(5, int(n / (swing + 15)))

    def numba_thresh():
        _compute_threshold_loop(proba, y_arr, atr, close, tp_val, sl_val, cost, swing, n, min_trades)

    def pandas_thresh():
        _find_optimal_threshold_pandas(proba, y_arr, atr, close, tp_val, sl_val, cost, swing)

    numba_thresh()
    t_numba = _timeit(numba_thresh, repeats=5)
    t_pandas = _timeit(pandas_thresh, repeats=3)
    print(f"\nfind_optimal_threshold ({n} bars):")
    print(f"  Pandas loop     : {t_pandas:.4f}s")
    print(f"  Numba JIT       : {t_numba:.4f}s")
    print(f"  Acceleration    : {t_pandas / t_numba:.1f}x faster")

    # --- fitness_score ---
    threshold = 0.55

    def numba_fitness():
        _simulate_fitness_sequential(
            proba, y_arr, atr, close, threshold, tp_val, sl_val, cost, swing, n
        )

    def pandas_fitness():
        _fitness_score_pandas(proba, y_arr, atr, close, threshold, tp_val, sl_val, cost, swing)

    numba_fitness()
    t_numba = _timeit(numba_fitness, repeats=5)
    t_pandas = _timeit(pandas_fitness, repeats=3)
    print(f"\nfitness_score / _simulate_fitness ({n} bars):")
    print(f"  Pandas loop     : {t_pandas:.4f}s")
    print(f"  Numba JIT       : {t_numba:.4f}s")
    print(f"  Acceleration    : {t_pandas / t_numba:.1f}x faster")


def main() -> None:
    print("AlphaQuant — Numba Audit (feat/quant-engine-overhaul)")
    check_compute_target_parity()
    check_find_optimal_threshold_parity()
    check_fitness_score_parity()
    run_benchmarks()
    print("\n✓ Mathematical parity verified at 100% (MDD tolerance: 1e-6)")


if __name__ == "__main__":
    main()
