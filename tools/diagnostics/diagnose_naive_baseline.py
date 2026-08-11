"""Standalone diagnostic: naive long-only baseline vs trained model.

Answers the question: is the model's apparent "edge" in val just the
market's directional drift (beta), or does the classifier add timing?

Simulates "enter long on every eligible bar" (no prediction at all,
threshold accepts 100% of bars) with the same one-position-at-a-time
constraint, TP/SL (1.5xATR / 1.0xATR) and swing=10 window as the winning
config, on val and test separately. Also trains the actual winning-config
model to compare its signals against the naive baseline and against
ema_50 trend.

Run:  python3 tools/diagnostics/diagnose_naive_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    compute_target,
    load_csv_data,
)

SYMBOL = "BTC_USDT"
TIMEFRAME = "4h"
SWING = 10
TP = 1.5
SL = 1.0
# fee=0/slippage=0 to match the diagnostic conditions under which the
# winning config showed val_pf=1.0543 / test_pf=0.7437.
COST_PER_TRADE = 0.0


def simulate_naive(prices: pd.DataFrame, y: pd.Series) -> dict:
    """Simulate entering long on every eligible bar (no prediction).

    Uses the same sequential one-position-at-a-time loop as
    fitness_score, but with proba=1 everywhere so every bar is eligible
    (subject to the swing_period cooldown).
    """
    n = len(prices)
    proba = np.ones(n)  # accept every bar
    trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
        proba=proba,
        y_arr=y.values.astype(np.float64),
        atr=prices["atr_14"].values,
        close=prices["close"].values,
        threshold=0.0,  # proba=1 >= 0.0 always
        tp_val=TP,
        sl_val=SL,
        cost_per_trade=COST_PER_TRADE,
        swing_period=SWING,
        n=n,
    )
    pf = gross_profit / max(gross_loss, 1e-9)
    return {
        "trade_count": trade_count,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": gross_profit - gross_loss,
        "profit_factor": pf,
        "max_drawdown": mdd,
    }


def simulate_model(model, X: pd.DataFrame, prices: pd.DataFrame,
                   y: pd.Series, threshold: float) -> dict:
    """Simulate the trained model's signals on a window."""
    proba = model.predict_proba(X)[:, 1]
    n = len(proba)
    trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
        proba=proba,
        y_arr=y.values.astype(np.float64),
        atr=prices["atr_14"].values,
        close=prices["close"].values,
        threshold=threshold,
        tp_val=TP,
        sl_val=SL,
        cost_per_trade=COST_PER_TRADE,
        swing_period=SWING,
        n=n,
    )
    pf = gross_profit / max(gross_loss, 1e-9)
    fired = proba >= threshold
    return {
        "trade_count": trade_count,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": gross_profit - gross_loss,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "fired_mask": fired,
    }


def main() -> None:
    print(f"=== Naive long-only baseline vs model: {SYMBOL} / {TIMEFRAME} ===")
    print(f"Config: swing={SWING}, tp={TP}xATR, sl={SL}xATR, cost={COST_PER_TRADE}\n")

    df = load_csv_data(SYMBOL, TIMEFRAME)
    compute_all_technicals(df)
    df_fg = get_fear_and_greed()
    df, has_sentiment = add_sentiment(df, df_fg)
    compute_target(df, swing_days=SWING, atr_tp_multi=TP, atr_sl_multi=SL,
                   timeframe_hours=4.0)

    # Keep close/atr for simulation; keep ema_50 for the trend analysis.
    prices = df[["close", "atr_14", "ema_50"]].copy()

    df_model = df.copy()
    cleanup_columns(df_model)  # drops OHLCV etc., drops NaN rows

    cal = get_calibrated_constants(TIMEFRAME)
    split = compute_dynamic_split(
        n_bars=len(df_model), swing_period=SWING, embargo_days=SWING,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        min_val_trades=cal["stat_floor_val_trades"],
        min_test_trades=cal["stat_floor_test_trades"],
        max_val_test_share=cal["max_val_test_share"],
    )
    n_train, n_val, n_test = split
    train_sl, val_sl, test_sl = compute_split_boundaries(
        n_train, n_val, n_test, embargo_days=SWING,
    )

    df_train = df_model.iloc[train_sl]
    df_val = df_model.iloc[val_sl]
    df_test = df_model.iloc[test_sl]

    # Align prices to the (post-dropna) model frames.
    prices_train = prices.loc[df_train.index]
    prices_val = prices.loc[df_val.index]
    prices_test = prices.loc[df_test.index]

    # ------------------------------------------------------------------
    # (2) Naive baseline on val and test
    # ------------------------------------------------------------------
    print("--- (2) Naive baseline (enter long whenever possible) ---")
    naive_val = simulate_naive(prices_val, df_val["target"])
    naive_test = simulate_naive(prices_test, df_test["target"])
    naive_train = simulate_naive(prices_train, df_train["target"])
    for name, r in [("train", naive_train), ("val", naive_val), ("test", naive_test)]:
        print(f"{name:5s}: trades={r['trade_count']:4d}  "
              f"profit_factor={r['profit_factor']:.4f}  "
              f"net_profit={r['net_profit']:+.4f}  "
              f"max_dd={r['max_drawdown']:.4f}")
    print()

    # ------------------------------------------------------------------
    # (3) Trained model (winning config: Volatility Hunter features)
    # ------------------------------------------------------------------
    print("--- (3) Trained model (Volatility Hunter features) ---")
    import xgboost as xgb

    features = ["atr_14", "bb_width", "bb_pos"]
    X_train = df_train[features]
    X_val = df_val[features]
    X_test = df_test[features]
    # XGBoost needs a binary label, but payoff simulation MUST keep the
    # ternary target (-1/0/1). Binarizing before _simulate_fitness_sequential
    # collapses SL (-1) into timeout (0) and inflates model PF.
    y_train_xgb = (df_train["target"] == 1).astype(int)
    y_val_xgb = (df_val["target"] == 1).astype(int)
    y_val = df_val["target"]
    y_test = df_test["target"]

    imbalance = sum(y_train_xgb == 0) / sum(y_train_xgb == 1)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        scale_pos_weight=imbalance, early_stopping_rounds=10,
        eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train_xgb, eval_set=[(X_val, y_val_xgb)], verbose=False)

    # Use the same threshold search as the pipeline (find_optimal_threshold).
    from src.utils.helpers import find_optimal_threshold
    best_threshold, _ = find_optimal_threshold(
        model, X_val, y_val, TP, SL, prices_val,
        fee_rate=0.0, slippage=0.0, swing_period=SWING,
    )
    print(f"best_threshold from find_optimal_threshold: {best_threshold}")

    model_val = simulate_model(model, X_val, prices_val, y_val, best_threshold)
    model_test = simulate_model(model, X_test, prices_test, y_test, best_threshold)
    for name, r in [("val", model_val), ("test", model_test)]:
        print(f"{name:5s}: model_trade_count={r['trade_count']:4d}  "
              f"profit_factor={r['profit_factor']:.4f}  "
              f"net_profit={r['net_profit']:+.4f}  "
              f"max_dd={r['max_drawdown']:.4f}")
    print()

    # ------------------------------------------------------------------
    # (3b) Direct comparison
    # ------------------------------------------------------------------
    print("--- (3b) Model vs naive ---")
    for name, m, n_ in [("val", model_val, naive_val), ("test", model_test, naive_test)]:
        delta = m["profit_factor"] - n_["profit_factor"]
        print(f"{name:5s}: model_pf={m['profit_factor']:.4f}  "
              f"naive_pf={n_['profit_factor']:.4f}  "
              f"delta={delta:+.4f}  "
              f"naive_trade_count={n_['trade_count']}  "
              f"model_trade_count={m['trade_count']}")
    print()

    # ------------------------------------------------------------------
    # (4) Fraction of model signals fired while price > ema_50
    # ------------------------------------------------------------------
    print("--- (4) Model signals vs ema_50 trend ---")
    for name, r, p in [("val", model_val, prices_val), ("test", model_test, prices_test)]:
        fired = r["fired_mask"]
        if fired.sum() == 0:
            print(f"{name}: no signals fired")
            continue
        above_ema = (p["close"] > p["ema_50"]).values
        frac = (fired & above_ema).sum() / fired.sum() * 100
        # Also: what fraction of ALL bars are above ema_50 (base rate)?
        base_rate = above_ema.mean() * 100
        print(f"{name:5s}: {frac:.1f}% of model signals fired while "
              f"price > ema_50 (base rate: {base_rate:.1f}% of all bars)")


if __name__ == "__main__":
    main()
