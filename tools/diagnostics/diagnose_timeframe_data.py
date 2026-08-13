"""Standalone Level-1 diagnostic for any symbol / timeframe.

Loads the timeframe CSV, runs compute_all_technicals + add_sentiment +
compute_target ONCE (swing=10, tp=1.5, sl=1.0), and reports:

  a) Per-feature % NaN and % exactly-0 (catches silent pandas_ta
     fallback-to-0 features that inject pure noise).
  b) Sentiment merge sanity: index dtype before/after merge_asof,
     duplicate/misaligned rows, and the expected "staircase" pattern
     (same FNG value repeated within a calendar day, changing once/day).
  c) Target class balance (% target==1 vs target==0).
  d) Val vs test regime comparison (cumulative return, return std).
  e) Per-feature point-biserial correlation with target on the train
     split (cheap signal proxy, no model training).

Run:  python3 tools/diagnostics/diagnose_timeframe_data.py --symbol BTC_USDT
     python3 tools/diagnostics/diagnose_timeframe_data.py --symbol SOL_USDT --timeframe 1h
"""

import sys
from pathlib import Path

import argparse
import numpy as np
import pandas as pd

# Make src/ importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.brain.data_fetcher import get_fear_and_greed
from src.brain.features import add_sentiment, compute_all_technicals
from src.utils.data_splits import (
    compute_dynamic_split,
    compute_split_boundaries,
    get_calibrated_constants,
)
from src.utils.helpers import compute_target, load_csv_data
from src.utils.timeframe_utils import parse_timeframe_hours

TP = 1.0
SL = 2.0

# Feature columns produced by compute_all_technicals + add_sentiment
# (excluding OHLCV raw columns and the target itself).
FEATURE_COLS = [
    "rsi_14", "macd", "macd_hist", "stoch_k",
    "ema_50", "dist_ema_50", "adx_14",
    "atr_14", "bb_width", "bb_pos",
    "obv", "vol_sma_20", "rel_volume",
    "fng_value", "fng_sma_14", "fng_vol_14",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC_USDT",
                        help="Trading pair (default: BTC_USDT)")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--swing", type=int, default=10)
    args = parser.parse_args()
    SYMBOL = args.symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    timeframe = args.timeframe
    swing = args.swing

    print(f"=== Level-1 diagnostic: {SYMBOL} / {timeframe} ===")
    print()

    df = load_csv_data(SYMBOL, timeframe)
    print(f"Loaded {len(df)} bars from CSV "
          f"({df.index.min()} -> {df.index.max()})")
    print(f"Index dtype after load_csv_data: {df.index.dtype}\n")

    compute_all_technicals(df)

    df_fg = get_fear_and_greed()
    print(f"Fear & Greed rows: {len(df_fg)}, index dtype: {df_fg.index.dtype}")

    df, has_sentiment = add_sentiment(df, df_fg)
    print(f"has_sentiment={has_sentiment}, "
          f"index dtype after add_sentiment: {df.index.dtype}\n")

    compute_target(df, swing_days=swing, atr_tp_multi=TP, atr_sl_multi=SL,
                   timeframe_hours=parse_timeframe_hours(timeframe))

    # ------------------------------------------------------------------
    # (a) Per-feature % NaN and % exactly-0
    # ------------------------------------------------------------------
    print("--- (a) Feature health: % NaN / % exactly-0 ---")
    rows = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            rows.append((col, "MISSING", "MISSING"))
            continue
        s = df[col]
        pct_nan = s.isna().mean() * 100
        pct_zero = (s == 0).mean() * 100
        rows.append((col, f"{pct_nan:6.2f}%", f"{pct_zero:6.2f}%"))
    print(pd.DataFrame(rows, columns=["feature", "% NaN", "% == 0"]).to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # (b) Sentiment merge sanity
    # ------------------------------------------------------------------
    print("--- (b) Sentiment merge sanity ---")
    print(f"df index dtype: {df.index.dtype} (expected datetime64[ns])")
    print(f"df index monotonic increasing: {df.index.is_monotonic_increasing}")
    print(f"df index duplicates: {df.index.duplicated().sum()}")
    if has_sentiment:
        fng = df["fng_value"]
        print(f"fng_value NaN count: {fng.isna().sum()} / {len(df)}")
        # Staircase check: within each calendar day, fng_value should be
        # constant (backward-filled from the daily FNG series).
        by_day = df.groupby(df.index.normalize())["fng_value"].nunique()
        days_with_multiple = (by_day > 1).sum()
        print(f"Calendar days with >1 distinct fng_value: {days_with_multiple} "
              f"(expected 0 if staircase pattern holds)")
        # Show a small sample of the staircase
        sample = df[["fng_value"]].iloc[100:112]
        print("Sample (12 consecutive 4h bars):")
        print(sample.to_string())
    print()

    # ------------------------------------------------------------------
    # (c) Target class balance
    # ------------------------------------------------------------------
    print("--- (c) Target class balance ---")
    tgt = df["target"].dropna()
    pct_1 = (tgt == 1).mean() * 100
    pct_0 = (tgt == 0).mean() * 100
    print(f"target==1: {pct_1:.2f}%  ({int((tgt == 1).sum())} rows)")
    print(f"target==0: {pct_0:.2f}%  ({int((tgt == 0).sum())} rows)")
    print(f"target NaN (live/unlabeled tail): {df['target'].isna().sum()} rows")
    print()

    # ------------------------------------------------------------------
    # (d) Val vs test regime comparison
    # ------------------------------------------------------------------
    print("--- (d) Val vs test regime ---")
    df_clean = df.dropna(subset=["target"]).copy()
    cal = get_calibrated_constants(timeframe)
    split = compute_dynamic_split(
        n_bars=len(df_clean), swing_period=swing, embargo_days=swing,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        min_val_trades=cal["stat_floor_val_trades"],
        min_test_trades=cal["stat_floor_test_trades"],
        max_val_test_share=cal["max_val_test_share"],
    )
    if split is None:
        print("compute_dynamic_split returned None — cannot compare regimes.")
    else:
        n_train, n_val, n_test = split
        train_sl, val_sl, test_sl = compute_split_boundaries(
            n_train, n_val, n_test, embargo_days=swing,
        )
        close = df_clean["close"]
        for name, sl in [("train", train_sl), ("val", val_sl), ("test", test_sl)]:
            window = close.iloc[sl]
            rets = window.pct_change().dropna()
            cum_ret = (window.iloc[-1] / window.iloc[0] - 1) * 100
            print(f"{name:5s}: bars={len(window):5d}  "
                  f"cum_return={cum_ret:+8.2f}%  "
                  f"ret_std={rets.std() * 100:.3f}%  "
                  f"({window.index.min().date()} -> {window.index.max().date()})")
    print()

    # ------------------------------------------------------------------
    # (e) Per-feature point-biserial correlation with target (train only)
    # ------------------------------------------------------------------
    print("--- (e) Point-biserial corr(feature, target) on train split ---")
    if split is not None:
        df_train = df_clean.iloc[train_sl]
        corrs = []
        for col in FEATURE_COLS:
            if col not in df_train.columns:
                continue
            s = df_train[col]
            if s.isna().all() or s.nunique() <= 1:
                corrs.append((col, np.nan))
                continue
            # Point-biserial == Pearson corr between continuous var and 0/1 var.
            valid = df_train[[col, "target"]].dropna()
            if len(valid) < 10:
                corrs.append((col, np.nan))
                continue
            corrs.append((col, valid[col].corr(valid["target"])))
        corr_df = (
            pd.DataFrame(corrs, columns=["feature", "corr_with_target"])
            .dropna()
            .assign(abs_corr=lambda d: d["corr_with_target"].abs())
            .sort_values("abs_corr", ascending=False)
        )
        print("Top 5 by |corr|:")
        print(corr_df.head(5)[["feature", "corr_with_target"]].to_string(index=False))
        print("\nBottom 5 by |corr|:")
        print(corr_df.tail(5)[["feature", "corr_with_target"]].to_string(index=False))


if __name__ == "__main__":
    main()
