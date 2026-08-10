"""strategy_optimizer.py — Trading strategy optimizer.

Searches for the optimal combination of parameters (swing, ATR TP, ATR SL) and
the best feature strategy for each symbol.
"""

import argparse
import datetime
import itertools
import json
import logging
import math
import warnings
from typing import Any

import pandas as pd
import numpy as np
    

from src.brain.features import compute_all_technicals, add_sentiment
from src.config.settings_loader import (
    get_active_symbols_with_timeframe,
    get_project_root,
    get_symbol_timeframe,
)
from src.utils.data_splits import (
    compute_dynamic_split,
    compute_min_val_trades,
    compute_split_boundaries,
    get_calibrated_constants,
)
from src.utils.helpers import (
    build_strategies,
    cleanup_columns,
    compute_target,
    load_csv_data,
    train_and_evaluate,
    train_and_evaluate_val_only,
)
from src.utils.timeframe_utils import parse_timeframe_hours

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

ATR_TP_RANGE: list[float] = [1.0, 1.5, 2.0, 3.0]
ATR_SL_RANGE: list[float] = [1.0, 1.5, 2.0, 2.5]
SWING_RANGE: list[int] = [5, 7, 10]
MAX_DEPTH_RANGE: list[int] = [2, 3, 4]
N_ESTIMATORS_RANGE: list[int] = [100, 200]
LEARNING_RATE_RANGE: list[float] = [0.01, 0.05]

# --- QUICK MODE (smoke-testing / diagnostics only) ---
# A drastically reduced grid used ONLY when --quick is passed on the CLI
# (or quick=True is passed to optimize_strategy directly). Kept as
# SEPARATE constants so the real production grid above is never touched
# or accidentally weakened. NEVER use a --quick config.json in
# production — it is not meaningfully optimized.
QUICK_ATR_TP_RANGE: list[float] = [1.5, 2.0]
QUICK_ATR_SL_RANGE: list[float] = [1.0, 1.5]
QUICK_SWING_RANGE: list[int] = [10]
QUICK_MAX_DEPTH_RANGE: list[int] = [3]
QUICK_N_ESTIMATORS_RANGE: list[int] = [100]
QUICK_LEARNING_RATE_RANGE: list[float] = [0.05]

# RECENT_WINDOW: Number of recent **bars** (not calendar days) used to
# count signals in the most recent portion of test+live data.
RECENT_WINDOW: int = 60

def _calculate_production_score(
    config: dict[str, Any],
    min_trades: int = 12,
) -> float:
    """Calculate a production score using ONLY validation metrics.

    The Test Set is never consulted here. Test-set divergence is
    evaluated separately in ``passes_oos_sanity_check`` after the
    winner is already selected.

    Args:
        config: Result dictionary containing validation metrics.
        min_trades: Minimum val trade count required to consider this
            config. Computed dynamically per asset via
            ``compute_min_val_trades``; falls back to 12 if not provided.

    Returns:
        Production score (higher is better). Returns a large negative
        value for configs that fail hard filters.
    """
    tc = config.get("val_trade_count", 0)
    pf = config.get("val_profit_factor", 0.0)
    dd = config.get("val_max_drawdown", 1.0)

    if tc < min_trades or pf <= 1.05:
        return -9999.0 + tc

    safe_dd = max(dd, 0.01)
    return ((pf - 1.0) / safe_dd) * math.log(tc)


def _passes_oos_sanity_check(
    winner: dict[str, Any],
    degradation_threshold: float = 0.50,
) -> bool:
    """Validate the selected winner against the Test Set.

    This function is called ONCE, AFTER the winner has been selected
    exclusively on validation metrics. It is a binary gate: approve or
    reject. It never re-ranks candidates, which would introduce
    Selection Bias.

    Args:
        winner: The top-ranked config dict from `_calculate_production_score`.
        degradation_threshold: Minimum acceptable ratio of
            test_fitness_score / val_fitness_score. Default 0.50 means
            the Test performance must be at least 50% of Validation
            performance. Tune this value based on your regime.

    Returns:
        True if the winner passes out-of-sample validation, False otherwise.
    """
    val_fs = winner.get("val_fitness_score", 0.0)
    test_fs = winner.get("test_fitness_score", 0.0)

    if val_fs <= 0:
        return False

    degradation = test_fs / val_fs
    return degradation >= degradation_threshold


def optimize_strategy(
    symbol: str,
    df_fg: pd.DataFrame | None = None,
    audit: bool = False,
    timeframe: str | None = None,
    quick: bool = False,
    quick_swing: int = QUICK_SWING_RANGE[0],
) -> pd.DataFrame:
    """Run optimization for a symbol and save the best config to JSON.

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'``).
        df_fg: Fear & Greed DataFrame (injected by the caller).
        audit: When True, evaluates test set and X_live for EVERY grid
            combination, producing a full optimization_report.csv for
            manual inspection — this is slow (the original behavior).
            When False (default), the inner loop only trains and scores
            on validation; test/X_live are evaluated once, only for the
            winning config, after selection. Use audit=True when you
            specifically want to inspect how close runner-up configs
            were, or to sanity-check `_calculate_production_score`.
        timeframe: Candle interval for this symbol (e.g. ``'4h'``).
            If ``None``, resolved from the per-symbol configuration in
            ``settings.yaml`` / ``bot_state.json``.
        quick: When True, uses the drastically reduced ``QUICK_*`` grids
            instead of the production grids — for fast smoke-testing or
            diagnosing a failure without re-running the full grid. NEVER
            use for a real production config; the resulting config.json
            is not meaningfully optimized. The split is still sized for
            the production worst-case swing (``max(SWING_RANGE)``), so a
            ``quick`` run exercises the same split-sizing path a real run
            would.
        quick_swing: Swing period (bars) used for the single combo in the
            ``--quick`` grid. Only meaningful when ``quick=True``.
            Defaults to ``QUICK_SWING_RANGE[0]`` (10, matching the
            production grid's worst case).
    """
    if timeframe is None:
        timeframe = get_symbol_timeframe(symbol)

    tf_hours = parse_timeframe_hours(timeframe)

    logger.info(
        "Starting optimization for %s (timeframe=%s, audit=%s, quick=%s)...",
        symbol, timeframe, audit, quick,
    )

    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"

    base_dir = get_project_root()
    output_dir = base_dir / "data" / "models" / safe_symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_raw = load_csv_data(safe_symbol, timeframe)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    # Retrieve timeframe-aware calibration constants
    cal = get_calibrated_constants(timeframe)

    compute_all_technicals(df_raw)
    fng_data = df_fg if df_fg is not None else pd.DataFrame()
    df_raw, has_sentiment = add_sentiment(df_raw, fng_data)

    logger.info("Searching for optimal parameters (Swing, ATR TP, ATR SL, Strategy)...")

    # max_swing_in_grid is ALWAYS the production worst-case swing
    # (max(SWING_RANGE)), regardless of --quick/--quick-swing: it sizes
    # compute_dynamic_split's split so a --quick run exercises the exact
    # same split-sizing path a real run would, not an easier case.
    max_swing_in_grid = max(SWING_RANGE)
    n_val = 0

    if quick:
        swing_values = [quick_swing]
        atr_tp_values, atr_sl_values = QUICK_ATR_TP_RANGE, QUICK_ATR_SL_RANGE
        max_depth_values = QUICK_MAX_DEPTH_RANGE
        n_estimators_values = QUICK_N_ESTIMATORS_RANGE
        learning_rate_values = QUICK_LEARNING_RATE_RANGE
        logger.info(
            "QUICK MODE: reduced grid for smoke-testing/diagnostics only "
            "(swing=%s tp=%s sl=%s) — resulting config.json is NOT "
            "production-quality.", swing_values, atr_tp_values, atr_sl_values,
        )
    else:
        swing_values = SWING_RANGE
        atr_tp_values, atr_sl_values = ATR_TP_RANGE, ATR_SL_RANGE
        max_depth_values = MAX_DEPTH_RANGE
        n_estimators_values = N_ESTIMATORS_RANGE
        learning_rate_values = LEARNING_RATE_RANGE

    all_results: list[dict[str, Any]] = []
    combos = list(itertools.product(swing_values, atr_tp_values, atr_sl_values))
    hp_combos = list(
        itertools.product(max_depth_values, n_estimators_values, learning_rate_values)
    )

    # Diagnostic counters (see Case 1 / Case 2 failure-path reporting below).
    skipped_swings: set[int] = set()
    combo_attempts = 0
    feature_skips = 0

    for sw, tp_m, sl_m in combos:
        df = df_raw.copy()

        compute_target(
            df, swing_days=sw, atr_tp_multi=tp_m, atr_sl_multi=sl_m,
            timeframe_hours=tf_hours,
        )

        prices_pre = df[["close", "atr_14"]].copy()

        df_live = df[df["target"].isna()].copy()

        cleanup_columns(df)
        cleanup_columns(df_live, drop_nan=False)

        split = compute_dynamic_split(
            n_bars=len(df),
            swing_period=max_swing_in_grid,
            embargo_days=sw,
            bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
            min_val_trades=cal["stat_floor_val_trades"],
            min_test_trades=cal["stat_floor_test_trades"],
            max_val_test_share=cal["max_val_test_share"],
        )

        if split is None:
            if sw not in skipped_swings:
                skipped_swings.add(sw)
                logger.warning(
                    "Asset %s: swing_period=%d has insufficient data for a "
                    "valid split (split sized for worst-case swing_period=%d). "
                    "Skipping all combos for this swing_period only — other "
                    "swing values in the grid are unaffected. See the "
                    "compute_dynamic_split warning above for n_bars/needed.",
                    symbol, sw, max_swing_in_grid,
                )
            continue

        n_train, n_val, n_test = split
        train_slice, val_slice, test_slice = compute_split_boundaries(
            n_train, n_val, n_test, embargo_days=sw,
        )

        df_train = df.iloc[train_slice]
        df_val = df.iloc[val_slice]
        df_test = df.iloc[test_slice]

        logger.info(
            "Dynamic split for %s — train=%d val=%d test=%d bars, embargo=%d each side",
            symbol, len(df_train), len(df_val), len(df_test), sw,
        )

        y_train = df_train["target"]
        y_val = df_val["target"]
        y_test = df_test["target"]

        prices_val = prices_pre.loc[df_val.index]
        prices_test = prices_pre.loc[df_test.index]

        strategies = build_strategies(df, has_sentiment)

        for name, features in strategies.items():
            valid_features = [f for f in features if f in df.columns]
            if not valid_features:
                feature_skips += 1
                continue

            X_train = df_train[valid_features]
            X_val = df_val[valid_features]
            X_test = df_test[valid_features]

            for md, ne, lr in hp_combos:
                hyperparams = {
                    "max_depth": md,
                    "n_estimators": ne,
                    "learning_rate": lr,
                }
                combo_attempts += 1

                if audit:
                    model, metrics, preds_test, _, opt_thresh = train_and_evaluate(
                        X_train, X_val, X_test, y_train, y_val, y_test,
                        tp_val=tp_m, sl_val=sl_m,
                        prices_val=prices_val, prices_test=prices_test,
                        hyperparams=hyperparams, swing_period=sw,
                    )

                    if opt_thresh == -1.0:
                        continue

                    X_live = df_live[valid_features]
                    if not X_live.empty:
                        y_probs_live = model.predict_proba(X_live)[:, 1]
                        preds_live = (y_probs_live >= opt_thresh).astype(int)
                        recent_preds = np.concatenate([preds_test, preds_live])
                    else:
                        recent_preds = preds_test

                    recent_signals = (
                        int(recent_preds[-RECENT_WINDOW:].sum())
                        if len(recent_preds) >= RECENT_WINDOW
                        else int(recent_preds.sum())
                    )

                    result = {
                        "symbol": symbol, "strategy_name": name,
                        "features": valid_features,
                        "optimal_threshold": float(opt_thresh),
                        "atr_tp_multi": float(tp_m), "atr_sl_multi": float(sl_m),
                        "swing_period": int(sw),
                        "max_depth": md, "n_estimators": ne, "learning_rate": lr,
                        "net_profit": metrics["net_profit_pct"],
                        "val_fitness_score": metrics["val_fitness_score"],
                        "test_fitness_score": metrics["test_fitness_score"],
                        "test_profit_factor": metrics["test_profit_factor"],
                        "test_max_drawdown": metrics["test_max_drawdown"],
                        "test_trade_count": metrics["test_trade_count"],
                        "val_profit_factor": metrics["val_profit_factor"],
                        "val_max_drawdown": metrics["val_max_drawdown"],
                        "val_trade_count": metrics["val_trade_count"],
                        "recent_signals": recent_signals,
                    }
                else:
                    _, val_metrics, opt_thresh = train_and_evaluate_val_only(
                        X_train, X_val, y_train, y_val,
                        tp_val=tp_m, sl_val=sl_m, prices_val=prices_val,
                        hyperparams=hyperparams, swing_period=sw,
                    )

                    if opt_thresh == -1.0:
                        continue

                    result = {
                        "symbol": symbol, "strategy_name": name,
                        "features": valid_features,
                        "optimal_threshold": float(opt_thresh),
                        "atr_tp_multi": float(tp_m), "atr_sl_multi": float(sl_m),
                        "swing_period": int(sw),
                        "max_depth": md, "n_estimators": ne, "learning_rate": lr,
                        **val_metrics,
                    }

                all_results.append(result)

    if not all_results:
        usable_combos = len(combos) - len(skipped_swings) * len(atr_tp_values) * len(atr_sl_values)
        if combo_attempts == 0:
            reason = (
                f"no strategy had any required feature present in the data across "
                f"{usable_combos} usable (swing,tp,sl) combos ({feature_skips} "
                f"strategy-instances skipped for missing features)."
            )
        else:
            reason = (
                f"all {combo_attempts} (strategy,hyperparameter) attempts across "
                f"{usable_combos} usable (swing,tp,sl) combos were rejected by "
                f"find_optimal_threshold for failing to reach the minimum trade "
                f"count (opt_thresh=-1.0)."
            )
        logger.error(
            "No satisfactory results for %s: %d/%d (swing,tp,sl) combos had "
            "sufficient data for a split; %s",
            symbol, usable_combos, len(combos), reason,
        )
        raise RuntimeError(f"No satisfactory results found for {symbol}.")

    valid_results = [
        res for res in all_results
        if res["val_fitness_score"] != -999.0
    ]

    if not valid_results:
        pf_values = [res.get("val_profit_factor", 0.0) for res in all_results]
        logger.error(
            "No valid configurations for %s: val_profit_factor across %d "
            "combos: min=%.2f mean=%.2f max=%.2f. All failed the "
            "val_fitness_score sentinel filter (pf<=1.05 or trade count "
            "below floor).",
            symbol, len(all_results), min(pf_values),
            sum(pf_values) / len(pf_values), max(pf_values),
        )
        failed_df = pd.DataFrame(all_results).sort_values(
            "val_profit_factor", ascending=False
        )
        failed_path = output_dir / "optimization_report_FAILED.csv"
        failed_df.to_csv(failed_path, index=False)
        logger.error("Failure diagnostic report saved to %s", failed_path)
        raise RuntimeError(
            f"No valid configurations found for {symbol}. "
            "All strategies produced unprofitable results on the validation set."
        )

    # n_val comes from the last swing's split in the loop above. This is
    # safe even after the skip-only-failing-swing fix: compute_dynamic_split
    # is always sized with the fixed max_swing_in_grid (not the per-combo
    # sw), so n_val varies by at most a few bars across SWING_RANGE
    # regardless of which swing's split happens to be the last one that
    # survives the loop — the kill-switch threshold below stays consistent.
    min_trades_val = compute_min_val_trades(
        n_val,
        max_swing_in_grid,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        absolute_floor=cal["stat_floor_val_trades"],
    )

    logger.info(
        "Dynamic kill-switch threshold for %s: min_val_trades=%d",
        symbol, min_trades_val,
    )

    sorted_results = sorted(
        valid_results,
        key=lambda c: _calculate_production_score(c, min_trades=min_trades_val),
        reverse=True,
    )
    best_config = sorted_results[0]

    if not audit:
        logger.info(
            "Fast mode: test/live metrics computed only for the winner "
            "(no CSV report will be saved — use --audit for a full report)."
        )
        sw_w = best_config["swing_period"]
        tp_w = best_config["atr_tp_multi"]
        sl_w = best_config["atr_sl_multi"]

        df_w = df_raw.copy()
        compute_target(
            df_w, swing_days=sw_w, atr_tp_multi=tp_w, atr_sl_multi=sl_w,
            timeframe_hours=tf_hours,
        )
        prices_pre_w = df_w[["close", "atr_14"]].copy()
        df_live_w = df_w[df_w["target"].isna()].copy()
        cleanup_columns(df_w)
        cleanup_columns(df_live_w, drop_nan=False)

        split_w = compute_dynamic_split(
            n_bars=len(df_w), swing_period=max_swing_in_grid, embargo_days=sw_w,
            bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
            min_val_trades=cal["stat_floor_val_trades"],
            min_test_trades=cal["stat_floor_test_trades"],
            max_val_test_share=cal["max_val_test_share"],
        )
        n_train_w, n_val_w, n_test_w = split_w
        train_slice_w, val_slice_w, test_slice_w = compute_split_boundaries(
            n_train_w, n_val_w, n_test_w, embargo_days=sw_w,
        )
        df_train_w = df_w.iloc[train_slice_w]
        df_val_w = df_w.iloc[val_slice_w]
        df_test_w = df_w.iloc[test_slice_w]

        feats = best_config["features"]
        X_train_w, X_val_w, X_test_w = (
            df_train_w[feats], df_val_w[feats], df_test_w[feats],
        )
        y_train_w, y_val_w, y_test_w = (
            df_train_w["target"], df_val_w["target"], df_test_w["target"],
        )
        prices_val_w = prices_pre_w.loc[df_val_w.index]
        prices_test_w = prices_pre_w.loc[df_test_w.index]

        hyperparams_w = {
            "max_depth": best_config["max_depth"],
            "n_estimators": best_config["n_estimators"],
            "learning_rate": best_config["learning_rate"],
        }

        model, metrics, preds_test, _, opt_thresh = train_and_evaluate(
            X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w,
            tp_val=tp_w, sl_val=sl_w,
            prices_val=prices_val_w, prices_test=prices_test_w,
            hyperparams=hyperparams_w, swing_period=sw_w,
            n_jobs=1,  # reproducible rerun of an already-selected config
        )

        X_live_w = df_live_w[feats]
        if not X_live_w.empty:
            y_probs_live = model.predict_proba(X_live_w)[:, 1]
            preds_live = (y_probs_live >= opt_thresh).astype(int)
            recent_preds = np.concatenate([preds_test, preds_live])
        else:
            recent_preds = preds_test
        recent_signals = (
            int(recent_preds[-RECENT_WINDOW:].sum())
            if len(recent_preds) >= RECENT_WINDOW
            else int(recent_preds.sum())
        )

        best_config = {
            **best_config,
            "net_profit": metrics["net_profit_pct"],
            "test_fitness_score": metrics["test_fitness_score"],
            "test_profit_factor": metrics["test_profit_factor"],
            "test_max_drawdown": metrics["test_max_drawdown"],
            "test_trade_count": metrics["test_trade_count"],
            "recent_signals": recent_signals,
        }
        sorted_results[0] = best_config

    # DEBUG
    if _passes_oos_sanity_check(best_config):
        logger.info("Winner passes out-of-sample sanity check.")
    else:
        logger.warning("Winner does not pass out-of-sample sanity check.")

    output_file = output_dir / "config.json"

    final_json: dict[str, Any] = {
        "symbol": best_config["symbol"],
        "timeframe": timeframe,
        "timeframe_hours": tf_hours,
        "strategy_name": best_config["strategy_name"],
        "features": best_config["features"],
        "optimal_threshold": round(best_config["optimal_threshold"], 3),
        "atr_tp_multi": best_config["atr_tp_multi"],
        "atr_sl_multi": best_config["atr_sl_multi"],
        "swing_period": best_config["swing_period"],
        "max_depth": best_config["max_depth"],
        "n_estimators": best_config["n_estimators"],
        "learning_rate": best_config["learning_rate"],
        "test_profit_factor": best_config["test_profit_factor"],
        "test_max_drawdown": best_config["test_max_drawdown"],
        "test_trade_count": best_config["test_trade_count"],
        "last_trained": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed_oos_sanity_check": bool(_passes_oos_sanity_check(best_config)),
    }
    try:
        with output_file.open("w", encoding="utf-8") as fh:
            json.dump(final_json, fh, indent=4)
    except IOError as e:
        logger.error("Failed to write config.json for %s: %s", symbol, e)
        raise

    df_report = pd.DataFrame(sorted_results)

    logger.info("Optimization completed for %s.", symbol)
    logger.info(
        "Winner: %s | Profit factor: %s",
        final_json["strategy_name"],
        best_config["test_profit_factor"],
    )

    return df_report

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy optimizer for Trading Bot.")
    parser.add_argument(
        "symbol",
        type=str,
        nargs="?",
        help="Symbol (e.g.: ETH_USDT). If not provided, uses settings.yaml.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe (e.g.: 4h, 1d). Uses per-symbol config if not provided.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Evaluate test set and X_live for every grid combo (slow, full CSV report). "
            "Default: only the winning config is evaluated on test/X_live.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a drastically reduced grid for fast smoke-testing (e.g. "
             "validating a new timeframe end-to-end, or diagnosing a failure "
             "without re-running the full grid). NEVER use for a real "
             "production config — the resulting config.json is not "
             "meaningfully optimized.",
    )
    parser.add_argument(
        "--quick-swing",
        type=int,
        default=QUICK_SWING_RANGE[0],
        help="Swing period (bars) for the --quick grid's single combo. "
             "Defaults to 10 (matches the production grid's worst-case "
             "swing used for split-sizing). Only meaningful with --quick.",
    )
    args = parser.parse_args()

    from src.brain.data_fetcher import get_fear_and_greed

    from src.utils.logging_config import setup_logging
    setup_logging()

    fng_data = get_fear_and_greed()
    base_dir = get_project_root()

    if args.symbol:
        safe_sym = args.symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
        sym_output_dir = base_dir / "data" / "models" / safe_sym
        sym_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            df_report = optimize_strategy(
                args.symbol, fng_data, audit=args.audit, timeframe=args.timeframe,
                quick=args.quick, quick_swing=args.quick_swing,
            )
        except Exception as e:
            logger.error("Error optimizing %s: %s", args.symbol, e)
            df_report = None
        if args.audit and isinstance(df_report, pd.DataFrame) and not df_report.empty:
            report_path = sym_output_dir / "optimization_report.csv"
            df_report.to_csv(report_path, index=False)
            logger.info("Full optimization report saved to %s", report_path)
        elif args.audit:
            logger.info(
                "No results to report — optimization was skipped or produced nothing."
            )
        else:
            logger.info(
                "Fast mode: no CSV report saved (only winner has test/live metrics). "
                "Re-run with --audit for a full report."
            )
    else:
        entries = get_active_symbols_with_timeframe()
        if not entries:
            logger.error("No symbol found in arguments or settings.yaml.")
        else:
            logger.info("Batch Mode: Optimizing %d assets...", len(entries))
            for entry in entries:
                try:
                    s = entry["symbol"]
                    tf = args.timeframe or entry["timeframe"]
                    safe_sym = s.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
                    sym_output_dir = base_dir / "data" / "models" / safe_sym
                    sym_output_dir.mkdir(parents=True, exist_ok=True)
                    df_report = optimize_strategy(
                        s, fng_data, audit=args.audit, timeframe=tf,
                        quick=args.quick, quick_swing=args.quick_swing,
                    )
                    if args.audit and isinstance(df_report, pd.DataFrame) and not df_report.empty:
                        report_path = sym_output_dir / "optimization_report.csv"
                        df_report.to_csv(report_path, index=False)
                        logger.info("Full optimization report saved to %s", report_path)
                    elif args.audit:
                        logger.info(
                            "No results to report for %s — optimization was skipped "
                            "or produced nothing.", s,
                        )
                    else:
                        logger.info("Fast mode: no CSV report saved for %s.", s)
                except (KeyError, ValueError, IOError) as exc:
                    logger.error("Error optimizing %s: %s", entry["symbol"], exc)
                    continue

if __name__ == "__main__":
    main()
