"""Experiment 02: funding_rate_current feature — Control vs Treatment walk-forward comparison.

Runs 8 walk-forward validations:
  4 CONTROL (current feature set, NO funding_rate_current):
    C1. BTC_USDT / 4h  × binary_homerun
    C2. BTC_USDT / 4h  × multiclass_3
    C3. BTC_USDT / 1h  × binary_homerun
    C4. BTC_USDT / 1h  × multiclass_3
  4 TREATMENT (current feature set + ["funding_rate_current"]):
    T1. BTC_USDT / 4h  × binary_homerun
    T2. BTC_USDT / 4h  × multiclass_3
    T3. BTC_USDT / 1h  × binary_homerun
    T4. BTC_USDT / 1h  × multiclass_3

All runs use identical configuration:
  - swing_period = 10 bars
  - tp_multi     = 1.5 × ATR_14
  - sl_multi     = 1.0 × ATR_14
  - window_months = 6, step_months = 6
  - fee_rate = 0.0, slippage = 0.0  (diagnostic mode: signal existence, not net PnL)
  - threshold_grid = per-formulation standard (see helpers.py)
  - n_bootstrap = 1000, n_blocks = 8, random_state = 42

NOTE: trend_htf is excluded from the base feature set on purpose — it was
discarded in exp01 (8/8 configurations failed the gate with no consistent
improvement pattern).  The "current feature set" here is the 14-col
technicals+sentiment baseline only.

Run:  python3 tools/diagnostics/exp02_funding_rate_walkforward.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.brain.data_fetcher import get_fear_and_greed
from src.brain.features import (
    add_funding_rate,
    add_sentiment,
    compute_all_technicals,
)
from src.config.paths import load_funding_rate_csv
from src.utils.helpers import (
    BINARY_HOMERUN_THRESHOLD_GRID,
    MULTICLASS_3_THRESHOLD_GRID,
    SENTIMENT_COLS,
    cleanup_columns,
    compute_target,
    load_csv_data,
    train_predict_binary_homerun,
    train_predict_multiclass_3,
)
from src.utils.oos_validation import WalkForwardResult, run_walk_forward
from src.utils.timeframe_utils import parse_timeframe_hours

# ---------------------------------------------------------------------------
# Experiment configuration (one place, shared across all 8 runs)
# ---------------------------------------------------------------------------
SYMBOL = "BTC_USDT"
SWING_PERIOD = 10
TP_MULTI = 1.5
SL_MULTI = 1.0
WINDOW_MONTHS = 6
STEP_MONTHS = 6
FEE_RATE = 0.0
SLIPPAGE = 0.0
N_BOOTSTRAP = 1000
N_BLOCKS = 8
RANDOM_STATE = 42

FORMULATIONS = [
    (
        "binary_homerun",
        train_predict_binary_homerun,
        BINARY_HOMERUN_THRESHOLD_GRID,
    ),
    (
        "multiclass_3",
        train_predict_multiclass_3,
        MULTICLASS_3_THRESHOLD_GRID,
    ),
]

TIMEFRAMES = ["4h", "1h"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_full_df(symbol: str, timeframe: str) -> tuple[pd.DataFrame, list[str]]:
    """Load data → technicals → sentiment → funding_rate → targets → cleanup.

    Returns:
        (df_with_targets_and_all_features, current_feature_set_list)
        ``current_feature_set_list`` contains all features present EXCEPT
        ``funding_rate_current`` — the caller adds it for the TREATMENT arm.

    NOTE: trend_htf is NOT included here (discarded after exp01).
    """
    tf_hours = parse_timeframe_hours(timeframe)

    df = load_csv_data(symbol, timeframe)
    compute_all_technicals(df)

    df_fg = get_fear_and_greed()
    df, has_sentiment = add_sentiment(df, df_fg)

    df_funding = load_funding_rate_csv(symbol)
    df, has_funding = add_funding_rate(df, df_funding)
    assert has_funding, (
        f"funding_rate_current not added for {symbol}/{timeframe}. "
        f"Check data/raw_csv/{symbol}/funding_rate.csv exists."
    )

    compute_target(
        df,
        swing_days=SWING_PERIOD,
        atr_tp_multi=TP_MULTI,
        atr_sl_multi=SL_MULTI,
        timeframe_hours=tf_hours,
    )

    cols_to_drop = ("open", "high", "low", "volume", "ema_50", "vol_sma_20",
                    "max_high_future", "min_low_future")
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    current_features = [
        c for c in df.columns
        if c not in ("close", "target", "funding_rate_current")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    required_base = {"rsi_14", "atr_14", "bb_width", "bb_pos", "obv", "rel_volume"}
    missing_base = required_base - set(current_features)
    assert not missing_base, f"Missing base features: {missing_base} — available: {list(df.columns)}"

    sentiment_present = any(c in current_features for c in SENTIMENT_COLS)
    assert sentiment_present, "Sentiment columns missing"

    assert "funding_rate_current" in df.columns, "funding_rate_current not present after add_funding_rate"

    return df, current_features


def run_one(
    df: pd.DataFrame,
    features: list[str],
    symbol: str,
    timeframe: str,
    formulation_name: str,
    train_fn,
    threshold_grid: tuple[float, float, float],
) -> WalkForwardResult:
    """Execute a single run_walk_forward call and return the result."""
    return run_walk_forward(
        df_raw=df,
        symbol=symbol,
        timeframe=timeframe,
        train_predict_fn=train_fn,
        tp_multi=TP_MULTI,
        sl_multi=SL_MULTI,
        swing_period=SWING_PERIOD,
        features=features,
        window_months=WINDOW_MONTHS,
        step_months=STEP_MONTHS,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        threshold_grid=threshold_grid,
        n_bootstrap=N_BOOTSTRAP,
        n_blocks=N_BLOCKS,
        random_state=RANDOM_STATE,
    )


def result_to_dict(name: str, variant: str, tf: str, form: str,
                   r: WalkForwardResult) -> dict[str, Any]:
    """Flatten WalkForwardResult into a JSON-serialisable dict."""
    return {
        "name": name,
        "variant": variant,
        "timeframe": tf,
        "formulation": form,
        "pooled_delta_bootstrap_p5": r.pooled_delta_bootstrap[0],
        "pooled_delta_bootstrap_p95": r.pooled_delta_bootstrap[1],
        "pooled_trade_count": r.pooled_trade_count,
        "passes_gate": r.passes_gate,
        "windows": [asdict(w) for w in r.windows],
    }


def print_windows_table(name: str, r: WalkForwardResult) -> None:
    """Print per-window detail lines for a single run."""
    header = (f"  {'window':>3} | {'start':>10} → {'end':>10} | {'cum_ret%':>9} "
              f"{'vol%':>6} | {'m_pf':>7} {'n_pf':>7} {'delta':>8} | "
              f"{'m_tc':>5} {'n_tc':>5} | {'thr':>5} | skip")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, w in enumerate(r.windows):
        s = w.start.strftime("%Y-%m-%d")
        e = w.end.strftime("%Y-%m-%d")
        cr = w.cum_ret * 100
        v = w.vol * 100
        skip = w.skipped_reason or "—"
        print(
            f"  {i:>3} | {s:>10} → {e:>10} | {cr:+9.4f} {v:6.3f} | "
            f"{w.model_pf:7.4f} {w.naive_pf:7.4f} {w.delta:+8.4f} | "
            f"{w.model_trade_count:>5} {w.naive_trade_count:>5} | "
            f"{w.threshold:5.3f} | {skip}"
        )


def format_pooled(r: WalkForwardResult) -> str:
    p5, p95 = r.pooled_delta_bootstrap
    gate = "✅ PASS" if r.passes_gate else "❌ FAIL"
    return (
        f"p5={p5:+.4f}  p95={p95:+.4f}  trades={r.pooled_trade_count:>5d}  "
        f"gate={gate}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()

    print("=" * 80)
    print("EXPERIMENTO 02 — funding_rate_current (última liquidación de funding 8h")
    print(f"Symbol: {SYMBOL}  |  Swing={SWING_PERIOD}  TP={TP_MULTI}xATR  "
          f"SL={SL_MULTI}xATR  |  Window={WINDOW_MONTHS}m Step={STEP_MONTHS}m  "
          f"|  fee={FEE_RATE} slip={SLIPPAGE}")
    print("=" * 80)

    master_dfs: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for tf in TIMEFRAMES:
        print(f"\n⟳ Building master DataFrame for {SYMBOL}/{tf}...")
        master_dfs[tf] = build_full_df(SYMBOL, tf)
        df, features_current = master_dfs[tf]
        print(f"  ✓ {len(df)} rows × {len(features_current)} current features  "
              f"(funding_rate_current extra: {'funding_rate_current' in df.columns})")
        print(f"    Range: {df.index.min().date()} → {df.index.max().date()}")

    all_results: list[dict[str, Any]] = []

    run_counter = 0
    for tf in TIMEFRAMES:
        df, features_current = master_dfs[tf]
        features_with_funding = features_current + ["funding_rate_current"]

        for form_name, train_fn, grid in FORMULATIONS:
            for variant, features in [
                ("CONTROL (no funding_rate)", features_current),
                ("TREATMENT (+ funding_rate_current)", features_with_funding),
            ]:
                run_counter += 1
                tag = f"[{run_counter}/8] {tf} × {form_name}  —  {variant}"
                print(f"\n{'─' * 80}")
                print(f"{tag}")
                print(f"{'─' * 80}")
                print(f"  Features: {len(features)} cols "
                      f"(current base: {len(features_current)} + "
                      f"{len(features) - len(features_current)} funding)")

                t_start = time.time()
                result = run_one(
                    df=df,
                    features=features,
                    symbol=SYMBOL,
                    timeframe=tf,
                    formulation_name=form_name,
                    train_fn=train_fn,
                    threshold_grid=grid,
                )
                t_elapsed = time.time() - t_start

                print(f"  Pooled bootstrap delta PF [{WINDOW_MONTHS}m×{STEP_MONTHS}m]:")
                print(f"    {format_pooled(result)}")
                print(f"  Windows detail:")
                print_windows_table(tag, result)
                print(f"  ⏱ Elapsed: {t_elapsed:.1f}s")

                short_name = (
                    f"{'T' if 'TREATMENT' in variant else 'C'}"
                    f"{'-'.join([tf, form_name])}"
                )
                all_results.append(
                    result_to_dict(
                        short_name, variant, tf, form_name, result
                    )
                )

    # ------------------------------------------------------------------
    # Final comparison table (control vs treatment, same 4 configs
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 100)
    print("TABLA FINAL — Control vs Treatment  |  "
          f"swing={SWING_PERIOD} TP={TP_MULTI} SL={SL_MULTI} "
          f"window={WINDOW_MONTHS}m step={STEP_MONTHS}m fee=0 slip=0")
    print("=" * 100)
    header = (
        f"{'Run':<32} {'Variant':<32} | "
        f"{'ΔPF p5':>8} {'ΔPF p95':>8} {'trades':>7} {'gate':>6} | "
        f"{'#wins':>5} {'ΔPF>0':>6} {'skip':>4}"
    )
    print(header)
    print("-" * 100)

    for r in all_results:
        n_used = sum(1 for w in r["windows"] if w["skipped_reason"] is None)
        n_pos = sum(1 for w in r["windows"]
                    if w["skipped_reason"] is None and w["delta"] > 0)
        n_skip = sum(1 for w in r["windows"] if w["skipped_reason"] is not None)
        gate = "PASS" if r["passes_gate"] else "FAIL"
        label = f"{r['timeframe']} × {r['formulation']}"
        print(
            f"{label:<32} {r['variant']:<32} | "
            f"{r['pooled_delta_bootstrap_p5']:+8.4f} "
            f"{r['pooled_delta_bootstrap_p95']:+8.4f} "
            f"{r['pooled_trade_count']:>7d} {gate:>6} | "
            f"{n_used:>5d} {n_pos:>6d} {n_skip:>4d}"
        )

    print("-" * 100)

    print("\nPairwise Δ(Treatment − Control) per configuration:")
    print(f"  {'Config':<30} {'Δp5':>9} {'Δp95':>9} {'Δtrades':>9} {'gate flips?':>12}")
    print("  " + "-" * 72)
    by_key = {(r["timeframe"], r["formulation"], r["variant"].split(" ")[0]): r
              for r in all_results}
    for tf in TIMEFRAMES:
        for form_name, _, _ in FORMULATIONS:
            c = by_key[(tf, form_name, "CONTROL")]
            t = by_key[(tf, form_name, "TREATMENT")]
            dp5 = (t["pooled_delta_bootstrap_p5"]
                   - c["pooled_delta_bootstrap_p5"])
            dp95 = (t["pooled_delta_bootstrap_p95"]
                    - c["pooled_delta_bootstrap_p95"])
            dtc = t["pooled_trade_count"] - c["pooled_trade_count"]
            c_gate = c["passes_gate"]
            t_gate = t["passes_gate"]
            flip = ("C→T PASS" if not c_gate and t_gate else
                    "C→T FAIL" if c_gate and not t_gate else
                    "no flip")
            print(
                f"  {tf + ' × ' + form_name:<30} {dp5:+9.4f} {dp95:+9.4f} "
                f"{dtc:+9d} {flip:>12}"
            )

    out_path = Path(__file__).resolve().parent / "exp02_funding_rate_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n💾 Full per-window detail saved to: {out_path}")
    print(f"⏱ Total experiment wall-time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
