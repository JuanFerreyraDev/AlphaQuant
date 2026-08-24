"""aq.py — Unified CLI for AlphaQuant multi-asset experiment pipeline.

Subcommands
-----------
baseline                       Screen a symbol with the 14-feature control baseline.
ab-test                        Run control vs treatment walk-forward for a feature profile.
diagnose-data                  Level-1 data health diagnostic for a symbol/timeframe pair.
diagnose-naive-baseline        Naive long-only baseline vs trained model comparison.
diagnose-swing-and-regimes     Swing sweep + cross-regime consistency (Part A+B).
diagnose-regimes-rigorous      Cross-regime analysis with bootstrap confidence intervals.
diagnose-timeframe-swing-sweep Swing sweep on alternative timeframe.

Examples::

    # Screen a new asset
    python -m tools.aq baseline SOL_USDT --timeframes 4h 1h

    # A/B test of a feature enrichment
    python -m tools.aq ab-test SOL_USDT --profile trend_htf --timeframes 4h

    # Data health check
    python -m tools.aq diagnose-data SOL_USDT --timeframe 4h

    # Diagnose baseline
    python -m tools.aq diagnose-naive-baseline BTC_USDT

    # Swing sweep + regimes
    python -m tools.aq diagnose-swing-and-regimes ETH_USDT

    # Rigorous cross-regime bootstrap
    python -m tools.aq diagnose-regimes-rigorous BTC_USDT

    # Swing sweep on 1h
    python -m tools.aq diagnose-timeframe-swing-sweep SOL_USDT --timeframe 1h

    # Fetch data first, then baseline
    python -m tools.aq baseline BNB_USDT --fetch --timeframes 4h
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable when run as ``python -m tools.aq``
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.experiment_defaults import DEFAULT_TIMEFRAMES, ExperimentConfig
from src.config.paths import _sanitize_symbol
from src.pipeline.feature_profiles import list_profile_names


# ---------------------------------------------------------------------------
# Subcommand: baseline
# ---------------------------------------------------------------------------
def _cmd_baseline(args: argparse.Namespace) -> None:
    """Screen a symbol with the 14-feature control baseline."""
    from src.pipeline.walkforward_runner import run_baseline

    symbol = _sanitize_symbol(args.symbol)
    tfs = args.timeframes or list(DEFAULT_TIMEFRAMES)
    cfg = _build_config(args)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.fetch:
        _fetch_data(symbol, tfs)

    run_baseline(
        symbol=symbol,
        timeframes=tfs,
        config=cfg,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Subcommand: ab-test
# ---------------------------------------------------------------------------
def _cmd_ab_test(args: argparse.Namespace) -> None:
    """Run control vs treatment walk-forward for a feature profile."""
    from src.pipeline.walkforward_runner import run_ab_test

    symbol = _sanitize_symbol(args.symbol)
    tfs = args.timeframes or list(DEFAULT_TIMEFRAMES)
    cfg = _build_config(args)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.fetch:
        _fetch_data(symbol, tfs)

    run_ab_test(
        symbol=symbol,
        profile=args.profile,
        timeframes=tfs,
        config=cfg,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Subcommand: diagnose-data
# ---------------------------------------------------------------------------
def _cmd_diagnose_data(args: argparse.Namespace) -> None:
    """Level-1 data health diagnostic for a symbol/timeframe pair."""
    import numpy as np
    import pandas as pd

    from src.brain.data_fetcher import get_fear_and_greed
    from src.brain.features import add_sentiment, compute_all_technicals
    from src.utils.data_splits import (
        compute_dynamic_split,
        compute_split_boundaries,
        get_calibrated_constants,
    )
    from src.utils.helpers import compute_target, load_csv_data
    from src.utils.timeframe_utils import parse_timeframe_hours

    symbol = _sanitize_symbol(args.symbol)
    timeframe = args.timeframe
    swing = args.swing

    print(f"=== Level-1 diagnostic: {symbol} / {timeframe} ===\n")

    df = load_csv_data(symbol, timeframe)
    print(f"Loaded {len(df)} bars from CSV "
          f"({df.index.min()} -> {df.index.max()})")
    print(f"Index dtype after load_csv_data: {df.index.dtype}\n")

    compute_all_technicals(df)

    df_fg = get_fear_and_greed()
    print(f"Fear & Greed rows: {len(df_fg)}, index dtype: {df_fg.index.dtype}")

    df, has_sentiment = add_sentiment(df, df_fg)
    print(f"has_sentiment={has_sentiment}, "
          f"index dtype after add_sentiment: {df.index.dtype}\n")

    compute_target(df, swing_days=swing, atr_tp_multi=1.5, atr_sl_multi=1.0,
                   timeframe_hours=parse_timeframe_hours(timeframe))

    # Feature columns produced by compute_all_technicals + add_sentiment
    FEATURE_COLS = [
        "rsi_14", "macd", "macd_hist", "stoch_k",
        "ema_50", "dist_ema_50", "adx_14",
        "atr_14", "bb_width", "bb_pos",
        "obv", "vol_sma_20", "rel_volume",
        "fng_value", "fng_sma_14", "fng_vol_14",
    ]

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
        by_day = df.groupby(df.index.normalize())["fng_value"].nunique()
        days_with_multiple = (by_day > 1).sum()
        print(f"Calendar days with >1 distinct fng_value: {days_with_multiple} "
              f"(expected 0 if staircase pattern holds)")
        sample = df[["fng_value"]].iloc[100:112]
        print("Sample (12 consecutive bars):")
        print(sample.to_string())
    print()

    # ------------------------------------------------------------------
    # (c) Target class balance
    # ------------------------------------------------------------------
    print("--- (c) Target class balance ---")
    tgt = df["target"].dropna()
    for label in sorted(tgt.unique()):
        count = int((tgt == label).sum())
        pct = (tgt == label).mean() * 100
        print(f"target=={label}: {pct:.2f}%  ({count} rows)")
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


# ---------------------------------------------------------------------------
# Subcommand: diagnose-naive-baseline
# ---------------------------------------------------------------------------
def _cmd_diagnose_naive_baseline(args: argparse.Namespace) -> None:
    """Standalone diagnostic: naive long-only baseline vs trained model."""
    import numpy as np
    import pandas as pd
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

    symbol = _sanitize_symbol(args.symbol)
    TIMEFRAME = "4h"
    SWING = 10
    TP = 1.5
    SL = 1.0
    COST_PER_TRADE = 0.0
    FEATURES = ["atr_14", "bb_width", "bb_pos"]

    print(f"=== Naive long-only baseline vs model: {symbol} / {TIMEFRAME} ===")
    print(f"Config: swing={SWING}, tp={TP}xATR, sl={SL}xATR, cost={COST_PER_TRADE}\n")

    df = load_csv_data(symbol, TIMEFRAME)
    compute_all_technicals(df)
    df_fg = get_fear_and_greed()
    df, _ = add_sentiment(df, df_fg)
    compute_target(df, swing_days=SWING, atr_tp_multi=TP, atr_sl_multi=SL, timeframe_hours=4.0)

    prices = df[["close", "atr_14", "ema_50"]].copy()
    df_model = df.copy()
    cleanup_columns(df_model)

    cal = get_calibrated_constants(TIMEFRAME)
    split = compute_dynamic_split(
        n_bars=len(df_model), swing_period=SWING, embargo_days=SWING,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        min_val_trades=cal["stat_floor_val_trades"],
        min_test_trades=cal["stat_floor_test_trades"],
        max_val_test_share=cal["max_val_test_share"],
    )
    n_train, n_val, n_test = split
    train_sl, val_sl, test_sl = compute_split_boundaries(n_train, n_val, n_test, embargo_days=SWING)

    df_train = df_model.iloc[train_sl]
    df_val = df_model.iloc[val_sl]
    df_test = df_model.iloc[test_sl]

    prices_train = prices.loc[df_train.index]
    prices_val = prices.loc[df_val.index]
    prices_test = prices.loc[df_test.index]

    def simulate_naive(prices_df, y):
        n = len(prices_df)
        proba = np.ones(n)
        trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
            proba=proba, y_arr=y.values.astype(np.float64),
            atr=prices_df["atr_14"].values, close=prices_df["close"].values,
            threshold=0.0, tp_val=TP, sl_val=SL, cost_per_trade=COST_PER_TRADE,
            swing_period=SWING, n=n,
        )
        pf = gross_profit / max(gross_loss, 1e-9)
        return {
            "trade_count": trade_count, "gross_profit": gross_profit,
            "gross_loss": gross_loss, "net_profit": gross_profit - gross_loss,
            "profit_factor": pf, "max_drawdown": mdd,
        }

    def simulate_model(model, X, prices_df, y, threshold):
        proba = model.predict_proba(X)[:, 1]
        n = len(proba)
        trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
            proba=proba, y_arr=y.values.astype(np.float64),
            atr=prices_df["atr_14"].values, close=prices_df["close"].values,
            threshold=threshold, tp_val=TP, sl_val=SL, cost_per_trade=COST_PER_TRADE,
            swing_period=SWING, n=n,
        )
        pf = gross_profit / max(gross_loss, 1e-9)
        fired = proba >= threshold
        return {
            "trade_count": trade_count, "gross_profit": gross_profit,
            "gross_loss": gross_loss, "net_profit": gross_profit - gross_loss,
            "profit_factor": pf, "max_drawdown": mdd, "fired_mask": fired,
        }

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

    print("--- (3) Trained model (Volatility Hunter features) ---")
    X_train = df_train[FEATURES]
    X_val = df_val[FEATURES]
    X_test = df_test[FEATURES]
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

    print("--- (3b) Model vs naive ---")
    for name, m, n_ in [("val", model_val, naive_val), ("test", model_test, naive_test)]:
        delta = m["profit_factor"] - n_["profit_factor"]
        print(f"{name:5s}: model_pf={m['profit_factor']:.4f}  "
              f"naive_pf={n_['profit_factor']:.4f}  "
              f"delta={delta:+.4f}  "
              f"naive_trade_count={n_['trade_count']}  "
              f"model_trade_count={m['trade_count']}")
    print()

    print("--- (4) Model signals vs ema_50 trend ---")
    for name, r, p in [("val", model_val, prices_val), ("test", model_test, prices_test)]:
        fired = r["fired_mask"]
        if fired.sum() == 0:
            print(f"{name}: no signals fired")
            continue
        above_ema = (p["close"] > p["ema_50"]).values
        frac = (fired & above_ema).sum() / fired.sum() * 100
        base_rate = above_ema.mean() * 100
        print(f"{name:5s}: {frac:.1f}% of model signals fired while "
              f"price > ema_50 (base rate: {base_rate:.1f}% of all bars)")


# ---------------------------------------------------------------------------
# Subcommand: diagnose-swing-and-regimes
# ---------------------------------------------------------------------------
def _cmd_diagnose_swing_and_regimes(args: argparse.Namespace) -> None:
    """Standalone diagnostic: swing sweep + cross-regime consistency."""
    import numpy as np
    import pandas as pd
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

    symbol = _sanitize_symbol(args.symbol)
    TIMEFRAME = "4h"
    TP = 1.5
    SL = 1.0
    COST = 0.0
    FEATURES = ["atr_14", "bb_width", "bb_pos"]
    SWINGS = [2, 3, 4, 5, 7, 10]

    print(f"=== Part A: swing sweep ({SWINGS}) on production val/test split for {symbol} ===")
    print(f"Config: tp={TP}xATR sl={SL}xATR cost={COST} features={FEATURES}\n")

    def simulate(prices_df, y, proba, threshold, swing):
        n = len(prices_df)
        trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
            proba=proba, y_arr=y.values.astype(np.float64),
            atr=prices_df["atr_14"].values, close=prices_df["close"].values,
            threshold=threshold, tp_val=TP, sl_val=SL, cost_per_trade=COST,
            swing_period=swing, n=n,
        )
        pf = gross_profit / max(gross_loss, 1e-9)
        return {
            "trade_count": trade_count, "net_profit": gross_profit - gross_loss,
            "profit_factor": pf, "max_drawdown": mdd,
        }

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

    def prepare_data(symbol_name, swing):
        df = load_csv_data(symbol_name, TIMEFRAME)
        compute_all_technicals(df)
        df_fg = get_fear_and_greed()
        df, _ = add_sentiment(df, df_fg)
        compute_target(df, swing_days=swing, atr_tp_multi=TP, atr_sl_multi=SL, timeframe_hours=4.0)
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
        train_sl, val_sl, test_sl = compute_split_boundaries(n_train, n_val, n_test, embargo_days=swing)
        return (df_model, prices, train_sl, val_sl, test_sl)

    def regime_stats(close):
        rets = close.pct_change().dropna()
        cum_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
        return cum_ret, rets.std() * 100

    header = (f"{'swing':>5} | {'val_pf':>7} {'naive':>7} {'delta':>7} "
              f"{'val_tc':>6} | {'test_pf':>7} {'naive':>7} {'delta':>7} {'test_tc':>6}")
    print(header)
    print("-" * len(header))

    results_a = {}
    for swing in SWINGS:
        df_model, prices, train_sl, val_sl, test_sl = prepare_data(symbol, swing)
        df_train = df_model.iloc[train_sl]
        df_val = df_model.iloc[val_sl]
        df_test = df_model.iloc[test_sl]
        prices_val = prices.loc[df_val.index]
        prices_test = prices.loc[df_test.index]

        naive_val = simulate(prices_val, df_val["target"], np.ones(len(df_val)), 0.0, swing)
        naive_test = simulate(prices_test, df_test["target"], np.ones(len(df_test)), 0.0, swing)

        model, thresh = train_model(df_train, df_val, prices_val, swing)
        if thresh == -1.0:
            print(f"{swing:>5} | find_optimal_threshold returned -1.0 — skipping")
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

    print(f"\n=== Part B: cross-regime consistency (swing={best_swing}) ===")
    windows = {
        "bear_2022": ("2022-01-01", "2022-12-31"),
        "range_2023": ("2023-01-01", "2023-12-31"),
        "bull_2024": ("2024-01-01", "2024-12-31"),
    }

    df_model, prices, _, _, _ = prepare_data(symbol, best_swing)

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

        df_prior = df_model.loc[df_model.index < start]
        if len(df_prior) < 500:
            print(f"{name:>12} | insufficient prior data to train ({len(df_prior)} bars)")
            continue
        n_prior = len(df_prior)
        n_val_prior = max(200, int(n_prior * 0.15))
        df_prior_train = df_prior.iloc[:-n_val_prior]
        df_prior_val = df_prior.iloc[-n_val_prior:]
        prices_prior_val = prices.loc[df_prior_val.index]

        model, thresh = train_model(df_prior_train, df_prior_val, prices_prior_val, best_swing)
        if thresh == -1.0:
            print(f"{name:>12} | threshold search failed on prior data")
            continue

        proba_w = model.predict_proba(df_w[FEATURES])[:, 1]
        model_w = simulate(prices_w, df_w["target"], proba_w, thresh, best_swing)
        naive_w = simulate(prices_w, df_w["target"], np.ones(len(df_w)), 0.0, best_swing)
        delta = model_w["profit_factor"] - naive_w["profit_factor"]

        print(f"{name:>12} | {cum_ret:+7.2f}% {vol:5.2f}% | "
              f"{model_w['profit_factor']:8.4f} {naive_w['profit_factor']:8.4f} "
              f"{delta:+7.4f} | {model_w['trade_count']:4d} {naive_w['trade_count']:4d}")


# ---------------------------------------------------------------------------
# Subcommand: diagnose-regimes-rigorous
# ---------------------------------------------------------------------------
def _cmd_diagnose_regimes_rigorous(args: argparse.Namespace) -> None:
    """Standalone diagnostic: cross-regime consistency, done rigorously."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from src.brain.data_fetcher import get_fear_and_greed
    from src.brain.features import add_sentiment, compute_all_technicals
    from src.utils.helpers import (
        cleanup_columns,
        compute_target,
        find_optimal_threshold,
        load_csv_data,
    )

    symbol = _sanitize_symbol(args.symbol)
    TIMEFRAME = "4h"
    SWING = 10
    TP = 1.5
    SL = 1.0
    COST = 0.0
    FEATURES = ["atr_14", "bb_width", "bb_pos"]
    PROD_TRAIN_END = pd.Timestamp("2024-08-12")
    WINDOWS = {
        "bear_2022": ("2022-01-01", "2022-12-31"),
        "range_2023": ("2023-01-01", "2023-12-31"),
        "bull_2024H1": ("2024-01-01", "2024-06-30"),
        "oos_val_2024H2": ("2024-08-14", "2025-08-10"),
        "oos_test_2025": ("2025-08-12", "2026-08-07"),
    }
    N_BOOTSTRAP = 1000
    RNG = np.random.default_rng(42)

    print(f"=== Cross-regime consistency (rigorous) — {symbol}/{TIMEFRAME} "
          f"swing={SWING} tp={TP} sl={SL} cost={COST} ===")
    print(f"Production train range ends: {PROD_TRAIN_END.date()}\n")

    df = load_csv_data(symbol, TIMEFRAME)
    compute_all_technicals(df)
    df_fg = get_fear_and_greed()
    df, _ = add_sentiment(df, df_fg)
    compute_target(df, swing_days=SWING, atr_tp_multi=TP, atr_sl_multi=SL, timeframe_hours=4.0)
    prices = df[["close", "atr_14"]].copy()
    df_model = df.copy()
    cleanup_columns(df_model)

    def per_trade_returns(prices_df, y, proba, threshold, swing):
        atr = prices_df["atr_14"].values
        close = prices_df["close"].values
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

    def profit_factor(rets):
        gains = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        return gains / max(losses, 1e-9)

    def bootstrap_delta(model_rets, naive_rets, n=N_BOOTSTRAP):
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

        sample_type = "IN-SAMPLE" if start_ts <= PROD_TRAIN_END else "OOS"
        close_w = prices_w["close"]
        rets_w = close_w.pct_change().dropna()
        cum_ret = (close_w.iloc[-1] / close_w.iloc[0] - 1) * 100
        vol = rets_w.std() * 100

        df_prior = df_model.loc[df_model.index < start_ts]
        if len(df_prior) < 500:
            print(f"{name:>16} | {sample_type:>12} | insufficient prior data ({len(df_prior)} bars)")
            continue
        n_prior = len(df_prior)
        n_val_prior = max(200, int(n_prior * 0.15))
        df_prior_train = df_prior.iloc[:-n_val_prior]
        df_prior_val = df_prior.iloc[-n_val_prior:]
        prices_prior_val = prices.loc[df_prior_val.index]

        model, thresh = train_model(df_prior_train, df_prior_val, prices_prior_val, SWING)
        if thresh == -1.0:
            print(f"{name:>16} | {sample_type:>12} | threshold search failed")
            continue

        proba_w = model.predict_proba(df_w[FEATURES])[:, 1]
        model_rets = per_trade_returns(prices_w, df_w["target"], proba_w, thresh, SWING)
        naive_rets = per_trade_returns(prices_w, df_w["target"], np.ones(len(df_w)), 0.0, SWING)

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
          "edge generalization. Only OOS rows with bootstrap intervals are comparable.")


# ---------------------------------------------------------------------------
# Subcommand: diagnose-timeframe-swing-sweep
# ---------------------------------------------------------------------------
def _cmd_diagnose_timeframe_swing_sweep(args: argparse.Namespace) -> None:
    """Standalone diagnostic: swing sweep on alternative timeframe."""
    import numpy as np
    import pandas as pd
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
    from src.utils.timeframe_utils import parse_timeframe_hours

    symbol = _sanitize_symbol(args.symbol)
    timeframe = args.timeframe or "1h"
    TP = 1.5
    SL = 1.0
    COST = 0.0
    FEATURES = ["atr_14", "bb_width", "bb_pos"]
    SWINGS = [20, 30, 40, 50, 60]

    print(f"=== swing sweep {SWINGS} — {symbol}/{timeframe} ===")
    print(f"Config: tp={TP}xATR sl={SL}xATR cost={COST} features={FEATURES}\n")

    df_raw = load_csv_data(symbol, timeframe)
    compute_all_technicals(df_raw)
    atr_pct = (df_raw["atr_14"] / df_raw["close"]).dropna()
    median_atr_pct = atr_pct.median()
    cost_rt = 2 * 0.001 + 2 * 0.0005
    print(f"ATR% of price ({timeframe}): median={median_atr_pct*100:.3f}%  "
          f"p25={atr_pct.quantile(0.25)*100:.3f}%  "
          f"p75={atr_pct.quantile(0.75)*100:.3f}%")
    tp_dist = median_atr_pct * TP
    print(f"TP distance at {TP}xATR (median): {tp_dist*100:.3f}%  |  "
          f"real round-trip cost: {cost_rt*100:.2f}%  |  "
          f"cost as % of TP: {cost_rt/tp_dist*100:.1f}%")
    print(f"(If cost is a large % of TP distance, the TP/SL ratio may need "
          f"rethinking at {timeframe} even though it worked at 4h.)\n")

    def simulate(prices_df, y, proba, threshold, swing):
        n = len(prices_df)
        trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
            proba=proba, y_arr=y.values.astype(np.float64),
            atr=prices_df["atr_14"].values, close=prices_df["close"].values,
            threshold=threshold, tp_val=TP, sl_val=SL, cost_per_trade=COST,
            swing_period=swing, n=n,
        )
        pf = gross_profit / max(gross_loss, 1e-9)
        return {"trade_count": trade_count, "net_profit": gross_profit - gross_loss,
                "profit_factor": pf, "max_drawdown": mdd}

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
        train_sl, val_sl, test_sl = compute_split_boundaries(n_train, n_val, n_test, embargo_days=swing)
        df_train, df_val, df_test = (df_model.iloc[train_sl], df_model.iloc[val_sl], df_model.iloc[test_sl])
        prices_val = prices.loc[df_val.index]
        prices_test = prices.loc[df_test.index]

        naive_val = simulate(prices_val, df_val["target"], np.ones(len(df_val)), 0.0, swing)
        naive_test = simulate(prices_test, df_test["target"], np.ones(len(df_test)), 0.0, swing)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from CLI overrides, falling back to defaults."""
    overrides = {}
    for field_name in (
        "swing_period", "tp_multi", "sl_multi",
        "window_months", "step_months",
        "fee_rate", "slippage",
        "n_bootstrap", "n_blocks", "random_state",
    ):
        val = getattr(args, field_name, None)
        if val is not None:
            overrides[field_name] = val
    return ExperimentConfig(**overrides)


def _fetch_data(symbol: str, timeframes: list[str]) -> None:
    """Invoke the data fetcher for each requested timeframe."""
    for tf in timeframes:
        print(f"\n⟳ Fetching {symbol}/{tf}...")
        cmd = [
            sys.executable, "-m", "src.brain.data_fetcher",
            symbol, "--timeframe", tf,
        ]
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
        if result.returncode != 0:
            print(f"[WARN] data_fetcher exited with code {result.returncode} "
                  f"for {symbol}/{tf}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aq",
        description="AlphaQuant multi-asset experiment pipeline CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m tools.aq baseline SOL_USDT --timeframes 4h 1h\n"
            "  python -m tools.aq ab-test SOL_USDT --profile trend_htf --timeframes 4h\n"
            "  python -m tools.aq diagnose-data SOL_USDT --timeframe 4h\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Shared args factory --
    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("symbol", type=str, help="Trading pair (e.g. SOL_USDT)")
        sp.add_argument("--timeframes", nargs="+", default=None,
                        help=f"Candle intervals (default: {' '.join(DEFAULT_TIMEFRAMES)})")
        sp.add_argument("--fetch", action="store_true",
                        help="Run data_fetcher before the experiment")
        sp.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: reports/{symbol}/)")

    def _add_config_overrides(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--swing-period", type=int, default=None)
        sp.add_argument("--tp-multi", type=float, default=None)
        sp.add_argument("--sl-multi", type=float, default=None)
        sp.add_argument("--window-months", type=int, default=None)
        sp.add_argument("--step-months", type=int, default=None)
        sp.add_argument("--fee-rate", type=float, default=None)
        sp.add_argument("--slippage", type=float, default=None)
        sp.add_argument("--n-bootstrap", type=int, default=None)
        sp.add_argument("--n-blocks", type=int, default=None)
        sp.add_argument("--random-state", type=int, default=None)

    # -- baseline --
    sp_baseline = subparsers.add_parser(
        "baseline",
        help="Screen a symbol with the 14-feature control baseline.",
        description=(
            "Run walk-forward OOS validation on the control feature set "
            "across all registered formulations. "
            "Gate: ΔPF p5 > 0.0 (model vs naive_long all-in long)."
        ),
    )
    _add_common(sp_baseline)
    _add_config_overrides(sp_baseline)
    sp_baseline.set_defaults(func=_cmd_baseline)

    # -- ab-test --
    sp_ab = subparsers.add_parser(
        "ab-test",
        help="Run control vs treatment walk-forward for a feature profile.",
        description=(
            "Run paired walk-forward validations (control vs treatment) for "
            "a given feature enrichment profile. Gate: ΔPF p5 > 0."
        ),
    )
    _add_common(sp_ab)
    _add_config_overrides(sp_ab)
    sp_ab.add_argument(
        "--profile", type=str, required=True,
        choices=list_profile_names(),
        help="Feature enrichment profile for the treatment arm.",
    )
    sp_ab.set_defaults(func=_cmd_ab_test)

    # -- diagnose-data --
    sp_diag = subparsers.add_parser(
        "diagnose-data",
        help="Level-1 data health diagnostic.",
        description=(
            "Loads the CSV for a symbol/timeframe, runs technicals + "
            "sentiment + target, and reports feature health, sentiment "
            "merge sanity, class balance, regime comparison, and "
            "point-biserial correlations."
        ),
    )
    sp_diag.add_argument("symbol", type=str, help="Trading pair (e.g. SOL_USDT)")
    sp_diag.add_argument("--timeframe", type=str, default="4h",
                         help="Candle interval (default: 4h)")
    sp_diag.add_argument("--swing", type=int, default=10,
                         help="Swing period in bars (default: 10)")
    sp_diag.set_defaults(func=_cmd_diagnose_data)

    # -- diagnose-naive-baseline --
    sp_naive = subparsers.add_parser(
        "diagnose-naive-baseline",
        help="Naive long-only baseline vs trained model comparison.",
        description=(
            "Simulate naive long-only baseline (100% entry on every eligible bar) "
            "vs Volatility Hunter model on val/test split. Compares PF, trades, "
            "and signal vs ema_50 trend correlation."
        ),
    )
    # This handler only reads `symbol`; do not advertise unused flags.
    sp_naive.add_argument("symbol", type=str, help="Trading pair (e.g. SOL_USDT)")
    sp_naive.set_defaults(func=_cmd_diagnose_naive_baseline)

    # -- diagnose-swing-and-regimes --
    sp_swing = subparsers.add_parser(
        "diagnose-swing-and-regimes",
        help="Swing sweep + cross-regime consistency analysis.",
        description=(
            "Part A: Sweep swing={2,3,4,5,7,10} on 4h, compare model vs naive. "
            "Part B: Use best swing to evaluate consistency across 3 historical regimes "
            "(bear 2022, range 2023, bull 2024). Train-only-past-data to avoid lookahead."
        ),
    )
    # This handler only reads `symbol`; do not advertise unused flags.
    sp_swing.add_argument("symbol", type=str, help="Trading pair (e.g. SOL_USDT)")
    sp_swing.set_defaults(func=_cmd_diagnose_swing_and_regimes)

    # -- diagnose-regimes-rigorous --
    sp_rig = subparsers.add_parser(
        "diagnose-regimes-rigorous",
        help="Cross-regime analysis with rigorous bootstrap confidence intervals.",
        description=(
            "Evaluates models on 5 windows (3 in-sample, 2 out-of-sample). "
            "For OOS windows, bootstrap 1000x per-trade returns to report "
            "p5/p95 intervals of model-vs-naive delta. Fully avoids lookahead."
        ),
    )
    # This handler only reads `symbol`; do not advertise unused flags.
    sp_rig.add_argument("symbol", type=str, help="Trading pair (e.g. SOL_USDT)")
    sp_rig.set_defaults(func=_cmd_diagnose_regimes_rigorous)

    # -- diagnose-timeframe-swing-sweep --
    sp_tf = subparsers.add_parser(
        "diagnose-timeframe-swing-sweep",
        help="Swing sweep on alternative timeframe (default 1h).",
        description=(
            "Sweep swing={20,30,40,50,60} bars on specified timeframe. "
            "Reports ATR-as-%-price sanity check and model vs naive comparison "
            "to validate TP/SL ratio at different timeframe scales."
        ),
    )
    # This handler reads `symbol` and `--timeframe` only; do not advertise other flags.
    sp_tf.add_argument("symbol", type=str, help="Trading pair (e.g. SOL_USDT)")
    sp_tf.add_argument("--timeframe", type=str, default="1h",
                       help="Candle interval (default: 1h)")
    sp_tf.set_defaults(func=_cmd_diagnose_timeframe_swing_sweep)

    # -- Parse and dispatch --
    args = parser.parse_args()

    # Normalize hyphenated CLI args to underscored attrs for ExperimentConfig
    for attr in ("swing_period", "tp_multi", "sl_multi", "window_months",
                 "step_months", "fee_rate", "n_bootstrap", "n_blocks",
                 "random_state"):
        # argparse converts --swing-period to swing_period automatically
        pass

    args.func(args)


if __name__ == "__main__":
    main()
