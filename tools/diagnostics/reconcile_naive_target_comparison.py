"""Reconcile naive PF discrepancy (old 0.6595 vs reported 0.9920).

Compares:
  A) Target VIEJO (AND-pessimistic binary) + sim VIEJA (else -> full SL)
  B) Target VIEJO + sim NUEVA (y==0 -> timeout market-close)  # the bug path
  C) Target NUEVO (bar-by-bar ternary) + sim NUEVA

Also prints exact val/test date boundaries and naive vs model trade counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brain.data_fetcher import get_fear_and_greed
from src.brain.features import add_sentiment, compute_all_technicals
from src.utils.data_splits import (
    compute_dynamic_split,
    compute_split_boundaries,
    get_calibrated_constants,
)
from src.utils.helpers import (
    _simulate_fitness_sequential,
    cleanup_columns,
    find_optimal_threshold,
    load_csv_data,
)

SYMBOL = "BTC_USDT"
SWING = 10
TP = 1.5
SL = 1.0
COST = 0.0
FEATURES = ["atr_14", "bb_width", "bb_pos"]


def compute_target_old(df: pd.DataFrame, swing_days: int, atr_tp: float, atr_sl: float) -> pd.Series:
    """Original AND-pessimistic binary target (rolling max/min over window)."""
    tp_price = df["close"] + df["atr_14"] * atr_tp
    sl_price = df["close"] - df["atr_14"] * atr_sl
    max_high_future = df["high"].rolling(window=swing_days).max().shift(-swing_days)
    min_low_future = df["low"].rolling(window=swing_days).min().shift(-swing_days)
    return ((max_high_future >= tp_price) & (min_low_future > sl_price)).astype(int)


def compute_target_new(df: pd.DataFrame, swing_days: int, atr_tp: float, atr_sl: float) -> pd.Series:
    """Current bar-by-bar same-TF ternary target (TP=1, SL=-1, timeout=0)."""
    from src.utils.helpers import _resolve_targets_same_tf

    tp_prices = (df["close"] + df["atr_14"] * atr_tp).values
    sl_prices = (df["close"] - df["atr_14"] * atr_sl).values
    targets = _resolve_targets_same_tf(
        tp_prices=tp_prices,
        sl_prices=sl_prices,
        highs=df["high"].values,
        lows=df["low"].values,
        swing_days=int(swing_days),
        n=len(df),
    )
    return pd.Series(targets, index=df.index, name="target")


def simulate_old_binary(
    prices: pd.DataFrame, y: pd.Series, *, threshold: float, proba: np.ndarray | None = None
) -> dict:
    """Pre-ternary simulation: y==1 -> TP, else -> full SL (no timeout path)."""
    n = len(prices)
    if proba is None:
        proba = np.ones(n)
    atr = prices["atr_14"].values
    close = prices["close"].values
    y_arr = y.values.astype(np.float64)

    trade_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    account_equity = 1.0
    running_max = 1.0
    max_drawdown = 0.0
    i = 0
    while i < n:
        if proba[i] >= threshold:
            if y_arr[i] == 1:
                ret = (atr[i] * TP) / close[i] - COST
                gross_profit += ret
            else:
                ret = -((atr[i] * SL) / close[i]) - COST
                gross_loss += abs(ret)
            trade_count += 1
            account_equity *= 1.0 + ret
            if account_equity > running_max:
                running_max = account_equity
            drawdown = (running_max - account_equity) / running_max
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            i += SWING
        else:
            i += 1
    pf = gross_profit / max(gross_loss, 1e-9)
    return {
        "trade_count": trade_count,
        "profit_factor": pf,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown": max_drawdown,
    }


def simulate_new(
    prices: pd.DataFrame, y: pd.Series, *, threshold: float, proba: np.ndarray | None = None
) -> dict:
    """Current simulation with timeout market-close for y==0."""
    n = len(prices)
    if proba is None:
        proba = np.ones(n)
    tc, gp, gl, mdd = _simulate_fitness_sequential(
        proba=proba,
        y_arr=y.values.astype(np.float64),
        atr=prices["atr_14"].values,
        close=prices["close"].values,
        threshold=threshold,
        tp_val=TP,
        sl_val=SL,
        cost_per_trade=COST,
        swing_period=SWING,
        n=n,
    )
    return {
        "trade_count": tc,
        "profit_factor": gp / max(gl, 1e-9),
        "gross_profit": gp,
        "gross_loss": gl,
        "max_drawdown": mdd,
    }


def split_frame(df_model: pd.DataFrame, timeframe: str):
    cal = get_calibrated_constants(timeframe)
    split_cal = compute_dynamic_split(
        n_bars=len(df_model),
        swing_period=SWING,
        embargo_days=SWING,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        min_val_trades=cal["stat_floor_val_trades"],
        min_test_trades=cal["stat_floor_test_trades"],
        max_val_test_share=cal["max_val_test_share"],
    )
    split_uncal = compute_dynamic_split(
        n_bars=len(df_model),
        swing_period=SWING,
        embargo_days=SWING,
        # defaults = 1d calibration (factor=5)
    )
    assert split_cal is not None and split_uncal is not None
    n_train, n_val, n_test = split_cal
    train_sl, val_sl, test_sl = compute_split_boundaries(
        n_train, n_val, n_test, embargo_days=SWING
    )
    return {
        "split_cal": split_cal,
        "split_uncal": split_uncal,
        "train_sl": train_sl,
        "val_sl": val_sl,
        "test_sl": test_sl,
        "df_train": df_model.iloc[train_sl],
        "df_val": df_model.iloc[val_sl],
        "df_test": df_model.iloc[test_sl],
    }


def train_xgb(X_train, y_train_bin, X_val, y_val_bin):
    imbalance = sum(y_train_bin == 0) / max(sum(y_train_bin == 1), 1)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=imbalance,
        early_stopping_rounds=10,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_bin, eval_set=[(X_val, y_val_bin)], verbose=False)
    return model, imbalance


def pick_threshold_new(model, X_val, y_val_for_thresh, prices_val):
    """Threshold search via current helpers (ternary-aware sim)."""
    best_threshold, _ = find_optimal_threshold(
        model,
        X_val,
        y_val_for_thresh,
        TP,
        SL,
        prices_val,
        fee_rate=0.0,
        slippage=0.0,
        swing_period=SWING,
    )
    return best_threshold


def pick_threshold_old(model, X_val, y_val, prices_val):
    """Threshold search with OLD sim + OLD min_trades formula."""
    proba_val = model.predict_proba(X_val)[:, 1]
    best_thr, best_profit = -1.0, -1e18
    n = len(proba_val)
    min_trades = max(5, int(n / (SWING + 15)))
    for t in np.arange(0.50, 0.86, 0.01):
        r = simulate_old_binary(prices_val, y_val, threshold=float(t), proba=proba_val)
        net = r["gross_profit"] - r["gross_loss"]
        if r["trade_count"] >= min_trades and net > best_profit:
            best_profit = net
            best_thr = float(t)
    return best_thr


def run_timeframe(timeframe: str) -> None:
    print("=" * 78)
    print(f"TIMEFRAME {timeframe}  swing={SWING} tp={TP} sl={SL} cost={COST}")
    print("=" * 78)

    df_raw = load_csv_data(SYMBOL, timeframe)
    compute_all_technicals(df_raw)
    df_fg = get_fear_and_greed()
    df_raw, _ = add_sentiment(df_raw, df_fg)

    y_old_full = compute_target_old(df_raw, SWING, TP, SL)
    y_new_full = compute_target_new(df_raw, SWING, TP, SL)
    prices_full = df_raw[["close", "atr_14", "ema_50"]].copy()

    # Match production: attach target BEFORE cleanup so dropna removes
    # trailing NaN labels (old rolling target) and feature warm-up NaNs.
    # Use OLD labels for the shared dropna mask so both regimes share
    # the exact same bar set / split (fair comparison).
    df_feat = df_raw.copy()
    df_feat["target"] = y_old_full
    cleanup_columns(df_feat)
    common_idx = df_feat.index
    y_old = y_old_full.loc[common_idx].astype(int)
    y_new = y_new_full.loc[common_idx].astype(int)
    prices = prices_full.loc[common_idx]

    parts = split_frame(df_feat, timeframe)
    print("\n--- Split sizes ---")
    print(f"calibrated:   train/val/test = {parts['split_cal']}")
    print(f"uncalibrated: train/val/test = {parts['split_uncal']}")
    print(f"sizes identical? {parts['split_cal'] == parts['split_uncal']}")

    val_idx = parts["df_val"].index
    test_idx = parts["df_test"].index
    print("\n--- Exact date boundaries (calibrated split) ---")
    print(f"val  start={val_idx[0]}  end={val_idx[-1]}  n={len(val_idx)}")
    print(f"test start={test_idx[0]}  end={test_idx[-1]}  n={len(test_idx)}")

    # Also confirm uncalibrated boundaries if sizes match.
    if parts["split_cal"] == parts["split_uncal"]:
        print("calibrated vs uncalibrated boundaries: IDENTICAL (same n_* => same slices)")

    def window(name: str):
        idx = parts[f"df_{name}"].index
        return prices.loc[idx], y_old.loc[idx], y_new.loc[idx], parts[f"df_{name}"]

    # ------------------------------------------------------------------
    # Naive baselines under three regimes
    # ------------------------------------------------------------------
    print("\n--- Naive PF under three regimes ---")
    rows = []
    for split_name in ("val", "test"):
        p, yo, yn, _ = window(split_name)
        a = simulate_old_binary(p, yo, threshold=0.0)  # target viejo + sim vieja
        b = simulate_new(p, yo, threshold=0.0)  # target viejo + sim nueva (BUG)
        c = simulate_new(p, yn, threshold=0.0)  # target nuevo + sim nueva
        rows.append((split_name, a, b, c))
        print(
            f"{split_name}: "
            f"OLD+OLD_sim pf={a['profit_factor']:.4f} trades={a['trade_count']} | "
            f"OLD+NEW_sim pf={b['profit_factor']:.4f} trades={b['trade_count']} | "
            f"NEW+NEW_sim pf={c['profit_factor']:.4f} trades={c['trade_count']}"
        )

    print("\nExpected reference (target VIEJO + sim VIEJA): naive_val≈0.6595 naive_test≈0.5737")

    # ------------------------------------------------------------------
    # Full model comparison: old target+old sim vs new target+new sim
    # ------------------------------------------------------------------
    print("\n--- Full comparison (model vs naive), trade counts separated ---")
    print(
        "NOTE: 'NEW+binarized_y' collapses SL(-1) into timeout(0) for model/threshold.\n"
        "      That was the pre-fix diagnose_naive_baseline.py path (model_val_pf=1.2791)."
    )
    header = (
        f"{'regime':<28} {'split':<5} {'naive_pf':>9} {'model_pf':>9} "
        f"{'delta':>9} {'naive_tc':>8} {'model_tc':>8} {'imb':>7} {'thr':>6}"
    )
    print(header)
    print("-" * len(header))

    # regime, y_labels, sim_fn, y_for_model_thresh, thresh_picker
    regimes = [
        (
            "OLD_target+OLD_sim",
            y_old,
            simulate_old_binary,
            y_old,  # binary already
            "old",
        ),
        (
            "NEW_target+NEW_sim",
            y_new,
            simulate_new,
            y_new,  # ternary (correct)
            "new",
        ),
        (
            "NEW+binarized_y (diagnose)",
            y_new,
            simulate_new,
            (y_new == 1).astype(int),  # BUG: collapses -1 and 0
            "new",
        ),
    ]

    for regime, y_all, sim_fn, y_for_sim_model, thresh_mode in regimes:
        train_df = parts["df_train"]
        val_df = parts["df_val"]
        test_df = parts["df_test"]
        y_tr = y_all.loc[train_df.index]
        y_va = y_all.loc[val_df.index]
        y_te = y_all.loc[test_df.index]
        y_tr_bin = (y_tr == 1).astype(int)
        y_va_bin = (y_va == 1).astype(int)

        prices_val = prices.loc[val_df.index]
        prices_test = prices.loc[test_df.index]
        X_train = train_df[FEATURES]
        X_val = val_df[FEATURES]
        X_test = test_df[FEATURES]

        model, imb = train_xgb(X_train, y_tr_bin, X_val, y_va_bin)
        y_va_thresh = y_for_sim_model.loc[val_df.index] if hasattr(y_for_sim_model, "loc") else y_for_sim_model.loc[val_df.index]
        # y_for_sim_model is a Series aligned to common_idx
        if thresh_mode == "new":
            thr = pick_threshold_new(model, X_val, y_for_sim_model.loc[val_df.index], prices_val)
        else:
            thr = pick_threshold_old(model, X_val, y_va, prices_val)

        for split_name, X, y_naive, y_model, p in (
            ("val", X_val, y_va, y_for_sim_model.loc[val_df.index], prices_val),
            ("test", X_test, y_te, y_for_sim_model.loc[test_df.index], prices_test),
        ):
            naive = sim_fn(p, y_naive, threshold=0.0)
            if thr < 0:
                model_r = {"trade_count": 0, "profit_factor": float("nan")}
                delta = float("nan")
            else:
                proba = model.predict_proba(X)[:, 1]
                model_r = sim_fn(p, y_model, threshold=thr, proba=proba)
                delta = model_r["profit_factor"] - naive["profit_factor"]
            print(
                f"{regime:<28} {split_name:<5} "
                f"{naive['profit_factor']:9.4f} {model_r['profit_factor']:9.4f} "
                f"{delta:+9.4f} {naive['trade_count']:8d} {model_r['trade_count']:8d} "
                f"{imb:7.2f} {thr:6.2f}"
            )


def main() -> None:
    print("=== Reconcile naive PF: target/sim OLD vs NEW ===\n")
    print("NOTE: diagnose_naive_baseline.py is untracked (no git history).")
    print("Diff beyond calibration: the *helpers* payoff path changed underneath")
    print("(_simulate_fitness_sequential now treats y==0 as timeout market-close).\n")
    for tf in ("4h", "1h"):
        run_timeframe(tf)


if __name__ == "__main__":
    main()
