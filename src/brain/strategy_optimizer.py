"""strategy_optimizer.py — Trading strategy optimizer.

Searches for the optimal combination of parameters (swing, ATR TP, ATR SL) and
the best feature strategy for each symbol.
"""
import argparse
import datetime
import itertools
import json
import logging
import warnings
from typing import Any

import pandas as pd


from src.brain.features import compute_all_technicals, add_sentiment
from src.config.settings_loader import get_active_symbols, get_project_root
from src.utils.helpers import (
    build_strategies,
    cleanup_columns,
    compute_target,
    load_csv_data,
    temporal_split_with_embargo,
    train_and_evaluate,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ============================================================================
#  OPTIMIZATION CONFIGURATION
# ============================================================================
ATR_TP_RANGE: list[float] = [1.5, 2.0, 2.5, 3.0]
ATR_SL_RANGE: list[float] = [1.0, 1.5, 2.0]
SWING_RANGE: list[int] = [5, 7, 10]
RECENT_WINDOW: int = 60


def optimize_strategy(symbol: str, df_fg: pd.DataFrame | None = None) -> None:
    """Run optimization for a symbol and save the best config to JSON.

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'``).
        df_fg: Fear & Greed DataFrame (injected by the caller).
    """
    logger.info("Starting optimization for %s...", symbol)

    safe_symbol = (
        symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    )
    filename = f"{safe_symbol}_1d.csv"

    # 1. Load data and compute features
    try:
        df_raw = load_csv_data(filename)
    except FileNotFoundError:
        raise RuntimeError(
            f"File data/raw_csv/{filename} not found"
        )

    compute_all_technicals(df_raw)
    fng_data = df_fg if df_fg is not None else pd.DataFrame()
    df_raw, has_sentiment = add_sentiment(df_raw, fng_data)

    logger.info("Searching for optimal parameters (Swing, ATR TP, ATR SL, Strategy)...")

    all_results: list[dict[str, Any]] = []
    combos = list(itertools.product(SWING_RANGE, ATR_TP_RANGE, ATR_SL_RANGE))

    for sw, tp_m, sl_m in combos:
        df = df_raw.copy()

        compute_target(df, swing_days=sw, atr_tp_multi=tp_m, atr_sl_multi=sl_m)
        cleanup_columns(df)

        df_train, df_val, df_test = temporal_split_with_embargo(df)
        y_train = df_train["target"]
        y_val = df_val["target"]
        y_test = df_test["target"]

        recent_mask = pd.Series(False, index=y_test.index)
        if len(recent_mask) > 0:
            recent_mask.iloc[-RECENT_WINDOW:] = True

        strategies = build_strategies(df, has_sentiment)

        for name, features in strategies.items():
            valid_features = [f for f in features if f in df.columns]
            if not valid_features:
                continue

            X_train = df_train[valid_features]
            X_val = df_val[valid_features]
            X_test = df_test[valid_features]

            _, metrics, preds_test, _, opt_thresh = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test,
                tp_val=tp_m, sl_val=sl_m,
            )

            recent_signals = int(((preds_test == 1) & recent_mask.values).sum())

            all_results.append({
                "symbol": symbol,
                "strategy_name": name,
                "features": valid_features,
                "optimal_threshold": float(opt_thresh),
                "atr_tp_multi": float(tp_m),
                "atr_sl_multi": float(sl_m),
                "swing_period": int(sw),
                "profit_neto": metrics["Profit_Neto"],
                "recent_signals": recent_signals,
            })

    if not all_results:
        raise RuntimeError(
            f"No satisfactory results found for {symbol}."
        )

    # 2. Select the best configuration
    best_config = sorted(
        all_results,
        key=lambda x: (x["profit_neto"], x["recent_signals"]),
        reverse=True,
    )[0]

    # 3. Export to JSON
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
        "last_trained": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(final_json, fh, indent=4)

    logger.info("Optimization completed for %s.", symbol)
    logger.info(
        "Winner: %s | Profit (Val): %s",
        final_json["strategy_name"],
        best_config["profit_neto"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy optimizer for Trading Bot."
    )
    parser.add_argument(
        "symbol",
        type=str,
        nargs="?",
        help="Symbol (e.g.: ETH_USDT). If not provided, uses settings.yaml.",
    )
    args = parser.parse_args()

    from src.brain.data_fetcher import get_fear_and_greed

    fng_data = get_fear_and_greed()

    if args.symbol:
        optimize_strategy(args.symbol, fng_data)
    else:
        symbols = get_active_symbols()
        if not symbols:
            logger.error(
                "No symbol found in arguments or settings.yaml."
            )
        else:
            logger.info("Batch Mode: Optimizing %d assets...", len(symbols))
            for s in symbols:
                try:
                    optimize_strategy(s, fng_data)
                except (KeyError, ValueError, IOError) as exc:
                    logger.error("Error optimizando %s: %s", s, exc)
                    continue
