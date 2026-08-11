"""Standalone diagnostic: swing sweep on BTC_USDT/1h (Part A equivalent).

Sweeps swing in {20, 30, 40, 50, 60} bars (~20h-60h holding) on the
production val/test split for 1h, comparing the Volatility Hunter model
against the naive "enter long whenever possible" baseline. fee=0/slippage=0,
tp=1.5xATR, sl=1.0xATR fixed (confirmed as a good starting point in 4h).

Also reports the ATR-as-%-of-price distribution on 1h to sanity-check
whether the tp=1.5/sl=1.0 ratio still makes economic sense at this
timeframe (vs the round-trip cost of 0.3%).

Run:  python3 tools/diagnose_timeframe_swing_sweep.py
"""

import argparse
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.timeframe_utils import parse_timeframe_hours

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
TP = 1.5
SL = 1.0
COST = 0.0
FEATURES = ["atr_14", "bb_width", "bb_pos"]

SWINGS = [20, 30, 40, 50, 60]


def simulate(prices, y, proba, threshold, swing):
    n = len(prices)
    trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
        proba=proba, y_arr=y.values.astype(np.float64),
        atr=prices["atr_14"].values, close=prices["close"].values,
        threshold=threshold, tp_val=TP, sl_val=SL,
        cost_per_trade=COST, swing_period=swing, n=n,
    )
    pf = gross_profit / max(gross_loss, 1e-9)
    return {"trade_count": trade_count, "net_profit": gross_profit - gross_loss,
            "profit_factor": pf, "max_drawdown": mdd}


def train_model(df_train, df_val, prices_val, swing):
    X_train, y_train = df_train[FEATURES], (df_train["target"] == 1).astype(int)
    X_val, y_val = df_val[FEATURES], (df_val["target"] == 1).astype(int)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="4h")
    args = parser.parse_args()
    timeframe = args.timeframe
    print(f"=== swing sweep {SWINGS} — {SYMBOL}/{timeframe} ===")
    print(f"Config: tp={TP}xATR sl={SL}xATR cost={COST} features={FEATURES}\n")

    # ATR-as-%-of-price sanity check for the TP/SL ratio at timeframe.
    df_raw = load_csv_data(SYMBOL, timeframe)
    compute_all_technicals(df_raw)
    atr_pct = (df_raw["atr_14"] / df_raw["close"]).dropna()
    median_atr_pct = atr_pct.median()
    cost_rt = 2 * 0.001 + 2 * 0.0005  # real round-trip cost
    print(f"ATR% of price ({timeframe}): median={median_atr_pct*100:.3f}%  "
          f"p25={atr_pct.quantile(0.25)*100:.3f}%  "
          f"p75={atr_pct.quantile(0.75)*100:.3f}%")
    tp_dist = median_atr_pct * TP
    print(f"TP distance at {TP}xATR (median): {tp_dist*100:.3f}%  |  "
          f"real round-trip cost: {cost_rt*100:.2f}%  |  "
          f"cost as % of TP: {cost_rt/tp_dist*100:.1f}%")
    print(f"(If cost is a large % of TP distance, the TP/SL ratio may need "
          f"rethinking at {timeframe} even though it worked at 4h.)\n")

    header = (f"{'swing':>5} {'~hours':>6} | {'val_pf':>7} {'naive':>7} "
              f"{'delta':>7} {'val_tc':>6} | {'test_pf':>7} {'naive':>7} "
              f"{'delta':>7} {'test_tc':>6}")
    print(header)
    print("-" * len(header))

    df_fg = get_fear_and_greed()
    cal = get_calibrated_constants(timeframe)

    for swing in SWINGS:
        df = df_raw.copy()
        add_sentiment(df, df_fg)
        compute_target(df, swing_days=swing, atr_tp_multi=TP, atr_sl_multi=SL,
                       timeframe_hours=parse_timeframe_hours(timeframe))
        prices = df[["close", "atr_14"]].copy()
        df_model = df.copy()
        cleanup_columns(df_model)

        split = compute_dynamic_split(
            n_bars=len(df_model), swing_period=max(SWINGS), embargo_days=swing,
            bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
            min_val_trades=cal["stat_floor_val_trades"],
            min_test_trades=cal["stat_floor_test_trades"],
            max_val_test_share=cal["max_val_test_share"],
        )
        if split is None:
            print(f"{swing:>5} {swing:>5}h | compute_dynamic_split returned None")
            continue
        n_train, n_val, n_test = split
        train_sl, val_sl, test_sl = compute_split_boundaries(
            n_train, n_val, n_test, embargo_days=swing,
        )
        df_train, df_val, df_test = (df_model.iloc[train_sl],
                                     df_model.iloc[val_sl],
                                     df_model.iloc[test_sl])
        prices_val = prices.loc[df_val.index]
        prices_test = prices.loc[df_test.index]

        naive_val = simulate(prices_val, df_val["target"],
                             np.ones(len(df_val)), 0.0, swing)
        naive_test = simulate(prices_test, df_test["target"],
                              np.ones(len(df_test)), 0.0, swing)

        model, thresh = train_model(df_train, df_val, prices_val, swing)
        if thresh == -1.0:
            print(f"{swing:>5} {swing:>5}h | threshold search failed")
            continue
        proba_val = model.predict_proba(df_val[FEATURES])[:, 1]
        proba_test = model.predict_proba(df_test[FEATURES])[:, 1]
        model_val = simulate(prices_val, df_val["target"], proba_val, thresh, swing)
        model_test = simulate(prices_test, df_test["target"], proba_test, thresh, swing)

        d_val = model_val["profit_factor"] - naive_val["profit_factor"]
        d_test = model_test["profit_factor"] - naive_test["profit_factor"]
        print(f"{swing:>5} {swing:>5}h | {model_val['profit_factor']:7.4f} "
              f"{naive_val['profit_factor']:7.4f} {d_val:+7.4f} "
              f"{model_val['trade_count']:6d} | "
              f"{model_test['profit_factor']:7.4f} "
              f"{naive_test['profit_factor']:7.4f} {d_test:+7.4f} "
              f"{model_test['trade_count']:6d}")


if __name__ == "__main__":
    main()
