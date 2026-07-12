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
from src.config.settings_loader import get_active_symbols, get_project_root
from src.utils.data_splits import (
    compute_dynamic_split,
    compute_min_val_trades,
    compute_split_boundaries,
)
from src.utils.helpers import (
    build_strategies,
    cleanup_columns,
    compute_target,
    load_csv_data,
    train_and_evaluate,
    train_and_evaluate_val_only,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

ATR_TP_RANGE: list[float] = [1.0, 1.5, 2.0, 3.0]
ATR_SL_RANGE: list[float] = [1.0, 1.5, 2.0, 2.5]
SWING_RANGE: list[int] = [5, 7, 10]
MAX_DEPTH_RANGE: list[int] = [2, 3, 4]
N_ESTIMATORS_RANGE: list[int] = [100, 200]
LEARNING_RATE_RANGE: list[float] = [0.01, 0.05]
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
) -> list[dict[str, Any]]:
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
    """
    logger.info(
        "Starting optimization for %s (audit=%s)...", symbol, audit,
    )

    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    filename = f"{safe_symbol}_1d.csv"

    try:
        df_raw = load_csv_data(filename)
    except FileNotFoundError:
        raise RuntimeError(f"File data/raw_csv/{filename} not found")

    compute_all_technicals(df_raw)
    fng_data = df_fg if df_fg is not None else pd.DataFrame()
    df_raw, has_sentiment = add_sentiment(df_raw, fng_data)

    logger.info("Searching for optimal parameters (Swing, ATR TP, ATR SL, Strategy)...")

    min_swing_in_grid = max(SWING_RANGE)
    n_val = 0

    all_results: list[dict[str, Any]] = []
    combos = list(itertools.product(SWING_RANGE, ATR_TP_RANGE, ATR_SL_RANGE))

    for sw, tp_m, sl_m in combos:
        df = df_raw.copy()

        compute_target(df, swing_days=sw, atr_tp_multi=tp_m, atr_sl_multi=sl_m)

        prices_pre = df[["close", "atr_14"]].copy()

        df_live = df[df["target"].isna()].copy()

        cleanup_columns(df)
        cleanup_columns(df_live, drop_nan=False)

        split = compute_dynamic_split(
            n_bars=len(df),
            swing_period=min_swing_in_grid,
            embargo_days=sw,
        )

        if split is None:
            logger.warning(
                "Asset %s has insufficient data for a statistically valid split "
                "(swing_period=%d). Skipping optimization.",
                symbol, sw,
            )
            return []

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

        hp_combos = list(
            itertools.product(MAX_DEPTH_RANGE, N_ESTIMATORS_RANGE, LEARNING_RATE_RANGE)
        )

        for name, features in strategies.items():
            valid_features = [f for f in features if f in df.columns]
            if not valid_features:
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
        raise RuntimeError(f"No satisfactory results found for {symbol}.")

    valid_results = [
        res for res in all_results
        if res["val_fitness_score"] != -999.0
    ]

    if not valid_results:
        raise RuntimeError(
            f"No valid configurations found for {symbol}. "
            "All strategies produced unprofitable results on the validation set."
        )

    min_trades_val = compute_min_val_trades(n_val, min_swing_in_grid)

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
        compute_target(df_w, swing_days=sw_w, atr_tp_multi=tp_w, atr_sl_multi=sl_w)
        prices_pre_w = df_w[["close", "atr_14"]].copy()
        df_live_w = df_w[df_w["target"].isna()].copy()
        cleanup_columns(df_w)
        cleanup_columns(df_live_w, drop_nan=False)

        split_w = compute_dynamic_split(
            n_bars=len(df_w), swing_period=min_swing_in_grid, embargo_days=sw_w,
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

    base_dir = get_project_root()
    output_dir = base_dir / "data" / "models" / safe_symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "config.json"

    final_json: dict[str, Any] = {
        "symbol": best_config["symbol"],
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy optimizer for Trading Bot.")
    parser.add_argument(
        "--symbol",
        type=str,
        nargs="?",
        help="Symbol (e.g.: ETH_USDT). If not provided, uses settings.yaml.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Evaluate test set and X_live for every grid combo (slow, full CSV report). "
            "Default: only the winning config is evaluated on test/X_live.",
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
            df_report = optimize_strategy(args.symbol, fng_data, audit=args.audit)
        except Exception as e:
            logger.error("Error optimizing %s: %s", args.symbol, e)
        if args.audit:
            report_path = sym_output_dir / "optimization_report.csv"
            df_report.to_csv(report_path, index=False)
            logger.info("Full optimization report saved to %s", report_path)
        else:
            logger.info(
                "Fast mode: no CSV report saved (only winner has test/live metrics). "
                "Re-run with --audit for a full report."
            )
    else:
        symbols = get_active_symbols()
        if not symbols:
            logger.error("No symbol found in arguments or settings.yaml.")
        else:
            logger.info("Batch Mode: Optimizing %d assets...", len(symbols))
            for s in symbols:
                try:
                    safe_sym = s.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
                    sym_output_dir = base_dir / "data" / "models" / safe_sym
                    sym_output_dir.mkdir(parents=True, exist_ok=True)
                    df_report = optimize_strategy(s, fng_data, audit=args.audit)
                    if args.audit:
                        report_path = sym_output_dir / "optimization_report.csv"
                        df_report.to_csv(report_path, index=False)
                        logger.info("Full optimization report saved to %s", report_path)
                    else:
                        logger.info("Fast mode: no CSV report saved for %s.", s)
                except (KeyError, ValueError, IOError) as exc:
                    logger.error("Error optimizing %s: %s", s, exc)
                    continue
