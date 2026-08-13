"""Standalone diagnostic: cross-regime consistency, done rigorously.

Corrects the methodological issues in the previous Part B:

  - Each window is evaluated with a model trained ONLY on data prior to
    that window's start (no lookahead within the experiment).
  - Each window is explicitly marked as IN-SAMPLE or OUT-OF-SAMPLE
    relative to the PRODUCTION training range (2020-01 -> 2024-08), so
    we don't draw "does the edge generalize?" conclusions from windows
    the production model already saw.
  - For genuinely out-of-sample windows, a bootstrap (1000 resamples
    with replacement over per-trade returns) reports the percentile 5-95
    interval of the model-vs-naive delta, to tell whether the observed
    delta is distinguishable from zero given the sample size.

Run:  python3 tools/diagnostics/diagnose_regimes_rigorous.py --symbol BTC_USDT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xgboost as xgb

from src.brain.data_fetcher import get_fear_and_greed
from src.brain.features import add_sentiment, compute_all_technicals
from src.utils.helpers import (
    cleanup_columns,
    compute_target,
    find_optimal_threshold,
    load_csv_data,
)

TIMEFRAME = "4h"
SWING = 10
TP = 1.5
SL = 1.0
COST = 0.0
FEATURES = ["atr_14", "bb_width", "bb_pos"]

# Production training range (from the current split): train ends 2024-08-12.
PROD_TRAIN_END = pd.Timestamp("2024-08-12")

# Windows to evaluate. Each is ~1 year. We include both in-sample
# (relative to production train) and out-of-sample windows, clearly marked.
WINDOWS = {
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "range_2023": ("2023-01-01", "2023-12-31"),
    "bull_2024H1": ("2024-01-01", "2024-06-30"),
    # Genuinely out-of-sample windows (after production train end):
    "oos_val_2024H2": ("2024-08-14", "2025-08-10"),   # == production val
    "oos_test_2025": ("2025-08-12", "2026-08-07"),    # == production test
}

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def per_trade_returns(prices: pd.DataFrame, y: pd.Series, proba: np.ndarray,
                      threshold: float, swing: int) -> np.ndarray:
    """Replicate _simulate_fitness_sequential but return per-trade returns."""
    atr = prices["atr_14"].values
    close = prices["close"].values
    y_arr = y.values
    n = len(proba)
    rets = []
    i = 0
    while i < n:
        if proba[i] >= threshold:
            if y_arr[i] == 1:
                rets.append((atr[i] * TP) / close[i] - COST)
            else:
                rets.append(-((atr[i] * SL) / close[i]) - COST)
            i += swing
        else:
            i += 1
    return np.array(rets)


def profit_factor(rets: np.ndarray) -> float:
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    return gains / max(losses, 1e-9)


def bootstrap_delta(model_rets: np.ndarray, naive_rets: np.ndarray,
                    n: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """Bootstrap the delta in profit factor between model and naive.

    Resamples each set of per-trade returns independently with
    replacement and recomputes PF delta. Returns (p5, median, p95).
    """
    deltas = []
    for _ in range(n):
        m = RNG.choice(model_rets, size=len(model_rets), replace=True)
        nv = RNG.choice(naive_rets, size=len(naive_rets), replace=True)
        deltas.append(profit_factor(m) - profit_factor(nv))
    deltas = np.array(deltas)
    return (float(np.percentile(deltas, 5)),
            float(np.percentile(deltas, 50)),
            float(np.percentile(deltas, 95)))


def train_model(df_train, df_val, prices_val, swing):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-regime consistency (rigorous)")
    parser.add_argument("--symbol", type=str, default="BTC_USDT", help="Trading pair (default: BTC_USDT)")
    args = parser.parse_args()
    symbol = args.symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"

    print(f"=== Cross-regime consistency (rigorous) — {symbol}/{TIMEFRAME} "
          f"swing={SWING} tp={TP} sl={SL} cost={COST} ===")
    print(f"Production train range ends: {PROD_TRAIN_END.date()}\n")

    df = load_csv_data(symbol, TIMEFRAME)
    compute_all_technicals(df)
    df_fg = get_fear_and_greed()
    df, _ = add_sentiment(df, df_fg)
    compute_target(df, swing_days=SWING, atr_tp_multi=TP, atr_sl_multi=SL,
                   timeframe_hours=4.0)
    prices = df[["close", "atr_14"]].copy()
    df_model = df.copy()
    cleanup_columns(df_model)

    header = (f"{'window':>16} | {'type':>12} | {'cum_ret':>8} {'vol':>6} | "
              f"{'m_pf':>7} {'n_pf':>7} {'delta':>7} | "
              f"{'m_tc':>4} {'n_tc':>4} | {'boot_p5':>8} {'boot_p95':>8}")
    print(header)
    print("-" * len(header))

    for name, (start, end) in WINDOWS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (df_model.index >= start_ts) & (df_model.index <= end_ts)
        df_w = df_model.loc[mask]
        prices_w = prices.loc[df_w.index]
        if len(df_w) < 200:
            print(f"{name:>16} | insufficient data ({len(df_w)} bars)")
            continue

        # In-sample vs out-of-sample relative to PRODUCTION train range.
        sample_type = "IN-SAMPLE" if start_ts <= PROD_TRAIN_END else "OOS"

        close_w = prices_w["close"]
        rets_w = close_w.pct_change().dropna()
        cum_ret = (close_w.iloc[-1] / close_w.iloc[0] - 1) * 100
        vol = rets_w.std() * 100

        # Train a model using ONLY data prior to this window.
        df_prior = df_model.loc[df_model.index < start_ts]
        if len(df_prior) < 500:
            print(f"{name:>16} | {sample_type:>12} | insufficient prior data "
                  f"({len(df_prior)} bars)")
            continue
        n_prior = len(df_prior)
        n_val_prior = max(200, int(n_prior * 0.15))
        df_prior_train = df_prior.iloc[:-n_val_prior]
        df_prior_val = df_prior.iloc[-n_val_prior:]
        prices_prior_val = prices.loc[df_prior_val.index]

        model, thresh = train_model(df_prior_train, df_prior_val,
                                    prices_prior_val, SWING)
        if thresh == -1.0:
            print(f"{name:>16} | {sample_type:>12} | threshold search failed")
            continue

        proba_w = model.predict_proba(df_w[FEATURES])[:, 1]
        model_rets = per_trade_returns(prices_w, df_w["target"], proba_w,
                                       thresh, SWING)
        naive_rets = per_trade_returns(prices_w, df_w["target"],
                                       np.ones(len(df_w)), 0.0, SWING)

        m_pf = profit_factor(model_rets)
        n_pf = profit_factor(naive_rets)
        delta = m_pf - n_pf

        if sample_type == "OOS" and len(model_rets) >= 20 and len(naive_rets) >= 20:
            p5, _, p95 = bootstrap_delta(model_rets, naive_rets)
            boot_str = f"{p5:+8.4f} {p95:+8.4f}"
        else:
            boot_str = f"{'—':>8} {'—':>8}"

        print(f"{name:>16} | {sample_type:>12} | {cum_ret:+7.2f}% {vol:5.2f}% | "
              f"{m_pf:7.4f} {n_pf:7.4f} {delta:+7.4f} | "
              f"{len(model_rets):4d} {len(naive_rets):4d} | {boot_str}")

    print("\nNote: IN-SAMPLE windows overlap the production training range "
          "(2020-01 -> 2024-08) and are NOT valid evidence for/against "
          "edge generalization. Only OOS rows with bootstrap intervals are "
          "comparable for that question.")


if __name__ == "__main__":
    main()
