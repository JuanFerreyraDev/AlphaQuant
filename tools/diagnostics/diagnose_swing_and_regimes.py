"""Standalone diagnostic: swing sweep (Part A) + cross-regime consistency (Part B).

Part A: for swing in {2,3,4,5,7,10}, train the Volatility Hunter model and
compare it against the naive "enter long whenever possible" baseline on the
production val/test split (fee=0/slippage=0, tp=1.5xATR, sl=1.0xATR fixed).

Part B: with the best swing from Part A, repeat model-vs-naive on 2-3
additional ~1-year historical windows outside the production split
(bear, range, bull regimes) to check whether the model's edge over naive
is consistent across regimes or specific to the current val/test partition.

Run:  python3 tools/diagnose_swing_and_regimes.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xgboost as xgb

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
    compute_target,
    find_optimal_threshold,
    load_csv_data,
)

SYMBOL = "BTC_USDT"
TIMEFRAME = "4h"
TP = 1.5
SL = 1.0
COST = 0.0  # diagnostic only: fee=0/slippage=0
FEATURES = ["atr_14", "bb_width", "bb_pos"]  # Volatility Hunter

SWINGS = [2, 3, 4, 5, 7, 10]


def simulate(prices: pd.DataFrame, y: pd.Series, proba: np.ndarray,
             threshold: float, swing: int) -> dict:
    """Run the sequential one-position-at-a-time simulation."""
    n = len(prices)
    trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
        proba=proba,
        y_arr=y.values.astype(np.float64),
        atr=prices["atr_14"].values,
        close=prices["close"].values,
        threshold=threshold,
        tp_val=TP,
        sl_val=SL,
        cost_per_trade=COST,
        swing_period=swing,
        n=n,
    )
    pf = gross_profit / max(gross_loss, 1e-9)
    return {
        "trade_count": trade_count,
        "net_profit": gross_profit - gross_loss,
        "profit_factor": pf,
        "max_drawdown": mdd,
    }


def train_model(df_train: pd.DataFrame, df_val: pd.DataFrame,
                prices_val: pd.DataFrame, swing: int):
    """Train XGBoost on the given features and find the optimal threshold."""
    X_train = df_train[FEATURES]
    y_train = (df_train["target"] == 1).astype(int)
    X_val = df_val[FEATURES]
    y_val = (df_val["target"] == 1).astype(int)

    imbalance = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        scale_pos_weight=imbalance, early_stopping_rounds=10,
        eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_threshold, _ = find_optimal_threshold(
        model, X_val, y_val, TP, SL, prices_val,
        fee_rate=0.0, slippage=0.0, swing_period=swing,
    )
    return model, best_threshold


def prepare_data(swing: int):
    """Load CSV, compute features+target for a given swing, split."""
    df = load_csv_data(SYMBOL, TIMEFRAME)
    compute_all_technicals(df)
    df_fg = get_fear_and_greed()
    df, _ = add_sentiment(df, df_fg)
    compute_target(df, swing_days=swing, atr_tp_multi=TP, atr_sl_multi=SL,
                   timeframe_hours=4.0)

    prices = df[["close", "atr_14", "ema_50"]].copy()
    df_model = df.copy()
    cleanup_columns(df_model)

    cal = get_calibrated_constants(TIMEFRAME)
    split = compute_dynamic_split(
        n_bars=len(df_model), swing_period=max(SWINGS), embargo_days=swing,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        min_val_trades=cal["stat_floor_val_trades"],
        min_test_trades=cal["stat_floor_test_trades"],
        max_val_test_share=cal["max_val_test_share"],
    )
    n_train, n_val, n_test = split
    train_sl, val_sl, test_sl = compute_split_boundaries(
        n_train, n_val, n_test, embargo_days=swing,
    )
    return (df_model, prices, train_sl, val_sl, test_sl)


def regime_stats(close: pd.Series) -> tuple[float, float]:
    """Cumulative return % and per-bar return std % for a window."""
    rets = close.pct_change().dropna()
    cum_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
    return cum_ret, rets.std() * 100


def main() -> None:
    print(f"=== Part A: swing sweep ({SWINGS}) on production val/test split ===")
    print(f"Config: tp={TP}xATR sl={SL}xATR cost={COST} features={FEATURES}\n")

    header = (f"{'swing':>5} | {'val_pf':>7} {'naive':>7} {'delta':>7} "
              f"{'val_tc':>6} | {'test_pf':>7} {'naive':>7} {'delta':>7} {'test_tc':>6}")
    print(header)
    print("-" * len(header))

    results_a = {}
    for swing in SWINGS:
        df_model, prices, train_sl, val_sl, test_sl = prepare_data(swing)
        df_train = df_model.iloc[train_sl]
        df_val = df_model.iloc[val_sl]
        df_test = df_model.iloc[test_sl]
        prices_val = prices.loc[df_val.index]
        prices_test = prices.loc[df_test.index]

        # Naive baselines
        naive_val = simulate(prices_val, df_val["target"],
                             np.ones(len(df_val)), 0.0, swing)
        naive_test = simulate(prices_test, df_test["target"],
                              np.ones(len(df_test)), 0.0, swing)

        # Model
        model, thresh = train_model(df_train, df_val, prices_val, swing)
        if thresh == -1.0:
            print(f"{swing:>5} | find_optimal_threshold returned -1.0 "
                  f"(no threshold met min trade count) — skipping")
            continue
        proba_val = model.predict_proba(df_val[FEATURES])[:, 1]
        proba_test = model.predict_proba(df_test[FEATURES])[:, 1]
        model_val = simulate(prices_val, df_val["target"], proba_val, thresh, swing)
        model_test = simulate(prices_test, df_test["target"], proba_test, thresh, swing)

        d_val = model_val["profit_factor"] - naive_val["profit_factor"]
        d_test = model_test["profit_factor"] - naive_test["profit_factor"]
        results_a[swing] = {
            "model_val": model_val, "naive_val": naive_val,
            "model_test": model_test, "naive_test": naive_test,
            "delta_val": d_val, "delta_test": d_test,
        }
        print(f"{swing:>5} | {model_val['profit_factor']:7.4f} "
              f"{naive_val['profit_factor']:7.4f} {d_val:+7.4f} "
              f"{model_val['trade_count']:6d} | "
              f"{model_test['profit_factor']:7.4f} "
              f"{naive_test['profit_factor']:7.4f} {d_test:+7.4f} "
              f"{model_test['trade_count']:6d}")

    # Pick the swing with the best average delta (model - naive) across
    # val and test, requiring a minimum trade count to be usable.
    MIN_TRADES = 20
    usable = {s: r for s, r in results_a.items()
              if r["model_val"]["trade_count"] >= MIN_TRADES
              and r["model_test"]["trade_count"] >= MIN_TRADES}
    if usable:
        best_swing = max(usable, key=lambda s: (usable[s]["delta_val"]
                                                + usable[s]["delta_test"]) / 2)
    else:
        best_swing = 10
    print(f"\nBest swing by avg delta (min {MIN_TRADES} trades): {best_swing}")

    # ------------------------------------------------------------------
    # Part B: cross-regime consistency with the chosen swing
    # ------------------------------------------------------------------
    print(f"\n=== Part B: cross-regime consistency (swing={best_swing}) ===")

    # Define ~1-year windows by date on the full dataset.
    windows = {
        "bear_2022": ("2022-01-01", "2022-12-31"),
        "range_2023": ("2023-01-01", "2023-12-31"),
        "bull_2024": ("2024-01-01", "2024-12-31"),
    }

    df_model, prices, _, _, _ = prepare_data(best_swing)

    header_b = (f"{'window':>12} | {'cum_ret':>8} {'vol':>6} | "
                f"{'model_pf':>8} {'naive_pf':>8} {'delta':>7} | "
                f"{'m_tc':>4} {'n_tc':>4}")
    print(header_b)
    print("-" * len(header_b))

    for name, (start, end) in windows.items():
        mask = (df_model.index >= start) & (df_model.index <= end)
        df_w = df_model.loc[mask]
        prices_w = prices.loc[df_w.index]
        if len(df_w) < 200:
            print(f"{name:>12} | insufficient data ({len(df_w)} bars)")
            continue

        cum_ret, vol = regime_stats(prices_w["close"])

        # Train on everything BEFORE this window to avoid lookahead.
        df_prior = df_model.loc[df_model.index < start]
        if len(df_prior) < 500:
            print(f"{name:>12} | insufficient prior data to train "
                  f"({len(df_prior)} bars)")
            continue
        # Use the tail of prior data as a pseudo-val for threshold search.
        n_prior = len(df_prior)
        n_val_prior = max(200, int(n_prior * 0.15))
        df_prior_train = df_prior.iloc[:-n_val_prior]
        df_prior_val = df_prior.iloc[-n_val_prior:]
        prices_prior_val = prices.loc[df_prior_val.index]

        model, thresh = train_model(df_prior_train, df_prior_val,
                                    prices_prior_val, best_swing)
        if thresh == -1.0:
            print(f"{name:>12} | threshold search failed on prior data")
            continue

        proba_w = model.predict_proba(df_w[FEATURES])[:, 1]
        model_w = simulate(prices_w, df_w["target"], proba_w, thresh, best_swing)
        naive_w = simulate(prices_w, df_w["target"],
                           np.ones(len(df_w)), 0.0, best_swing)
        delta = model_w["profit_factor"] - naive_w["profit_factor"]

        print(f"{name:>12} | {cum_ret:+7.2f}% {vol:5.2f}% | "
              f"{model_w['profit_factor']:8.4f} {naive_w['profit_factor']:8.4f} "
              f"{delta:+7.4f} | {model_w['trade_count']:4d} "
              f"{naive_w['trade_count']:4d}")


if __name__ == "__main__":
    main()
