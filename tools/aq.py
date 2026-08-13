"""aq.py — Unified CLI for AlphaQuant multi-asset experiment pipeline.

Subcommands
-----------
baseline     Screen a symbol with the 14-feature control baseline.
ab-test      Run control vs treatment walk-forward for a feature profile.
diagnose-data  Level-1 data health diagnostic for a symbol/timeframe pair.

Examples::

    # Screen a new asset
    python -m tools.aq baseline SOL_USDT --timeframes 4h 1h

    # A/B test of a feature enrichment
    python -m tools.aq ab-test SOL_USDT --profile trend_htf --timeframes 4h

    # Data health check
    python -m tools.aq diagnose-data SOL_USDT --timeframe 4h

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
