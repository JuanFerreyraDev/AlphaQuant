"""train.py — Production model training factory.

Reads the optimal configuration from ``config.json`` and trains a final
XGBoost model with all available data.
"""

import argparse
import datetime
import json
import logging
import warnings
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from src.brain.features import compute_all_technicals, add_sentiment
from src.config.settings_loader import (
    get_active_symbols_with_timeframe,
    get_project_root,
    get_symbol_timeframe,
)
from src.utils.helpers import (
    cleanup_columns,
    compute_target,
    load_csv_data,
)
from src.utils.timeframe_utils import parse_timeframe_hours

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def train_factory(
    symbol: str, df_fg: pd.DataFrame | None = None, timeframe: str | None = None
) -> None:
    """Train a model based on the configuration saved in config.json.

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'`` or ``'BTC/USDT'``).
        df_fg: Fear & Greed DataFrame (injected by the caller).
        timeframe: Candle interval for this symbol (e.g. ``'4h'``).
            If ``None``, resolved from the per-symbol configuration.
    """
    if timeframe is None:
        timeframe = get_symbol_timeframe(symbol)
    tf_hours = parse_timeframe_hours(timeframe)

    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    base_dir = get_project_root()
    config_path = base_dir / "data" / "models" / safe_symbol / "config.json"

    if not config_path.exists():
        raise RuntimeError(f"Configuration not found at {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Error parsing config.json for {safe_symbol}: {exc}"
        ) from exc

    strategy_name: str = config["strategy_name"]
    features_list: list[str] = config["features"]
    optimal_threshold: float = config["optimal_threshold"]
    atr_tp_multi: float = config["atr_tp_multi"]
    atr_sl_multi: float = config["atr_sl_multi"]
    swing_period: int = config["swing_period"]

    logger.info(
        "Starting Training Factory for %s (timeframe=%s)...", symbol, timeframe
    )
    logger.info("Strategy: %s", strategy_name)

    try:
        df = load_csv_data(safe_symbol, timeframe)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    compute_all_technicals(df)
    df, _ = add_sentiment(df, df_fg if df_fg is not None else pd.DataFrame())

    compute_target(
        df,
        swing_days=swing_period,
        atr_tp_multi=atr_tp_multi,
        atr_sl_multi=atr_sl_multi,
        timeframe_hours=tf_hours,
    )
    cleanup_columns(df)

    X = df[features_list]
    y = df["target"]

    imbalance = sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1

    hp_n_estimators: int = config["n_estimators"]
    hp_max_depth: int = config["max_depth"]
    hp_learning_rate: float = config["learning_rate"]

    model = xgb.XGBClassifier(
        n_estimators=hp_n_estimators,
        max_depth=hp_max_depth,
        learning_rate=hp_learning_rate,
        scale_pos_weight=imbalance,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    logger.info(
        "Training XGBClassifier with %d features (depth=%d, trees=%d, lr=%.4f)...",
        len(features_list), hp_max_depth, hp_n_estimators, hp_learning_rate,
    )

    # Hold out the most recent rows for early stopping to prevent overfitting,
    # mirroring the regularization used during optimization.
    holdout_size = min(30, int(len(X) * 0.1))
    if holdout_size >= 10:
        X_fit, X_hold = X.iloc[:-holdout_size], X.iloc[-holdout_size:]
        y_fit, y_hold = y.iloc[:-holdout_size], y.iloc[-holdout_size:]
        model.set_params(early_stopping_rounds=10)
        model.fit(X_fit, y_fit, eval_set=[(X_hold, y_hold)], verbose=False)
    else:
        model.fit(X, y, verbose=False)

    production_bundle: dict[str, Any] = {
        "model": model,
        "features": features_list,
        "threshold": optimal_threshold,
        "atr_tp_multi": atr_tp_multi,
        "atr_sl_multi": atr_sl_multi,
        "strategy_name": strategy_name,
        "timeframe": timeframe,
        "timeframe_hours": tf_hours,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    models_dir = base_dir / "data" / "models" / safe_symbol
    models_dir.mkdir(parents=True, exist_ok=True)

    safe_strategy_name = (
        strategy_name.replace("/", "_").replace(" ", "_").replace("+", "")
    )
    safe_name = f"{safe_symbol}_{atr_tp_multi}_{atr_sl_multi}_{swing_period}_{optimal_threshold}".replace(
        ".", "-"
    )
    out_filename = f"{safe_name}_{safe_strategy_name}.pkl".lower()
    out_path = models_dir / out_filename

    joblib.dump(production_bundle, out_path)
    logger.info("Model exported successfully to: %s", out_path)
    logger.info("Ready for production with threshold: %s", optimal_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Training Factory.")
    parser.add_argument(
        "symbol",
        type=str,
        nargs="?",
        help="Symbol to train (e.g.: ETH_USDT). If not provided, uses settings.yaml.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe (e.g.: 4h, 1d). Uses per-symbol config if not provided.",
    )
    args = parser.parse_args()

    from src.brain.data_fetcher import get_fear_and_greed

    fng_data = get_fear_and_greed()

    if args.symbol:
        tf = args.timeframe or get_symbol_timeframe(args.symbol)
        train_factory(args.symbol, fng_data, tf)
    else:
        entries = get_active_symbols_with_timeframe()
        if not entries:
            logger.error("No symbol found in arguments or settings.yaml.")
        else:
            logger.info("Batch Mode: Training %d models...", len(entries))
            ok_count = 0
            fail_count = 0
            for entry in entries:
                try:
                    tf = args.timeframe or entry["timeframe"]
                    train_factory(entry["symbol"], fng_data, tf)
                    ok_count += 1
                except (KeyError, ValueError, IOError) as exc:
                    logger.error("Error training %s: %s", entry["symbol"], exc)
                    fail_count += 1
                    continue

            logger.info(
                "Batch training finished. Success: %d | Failed: %d",
                ok_count,
                fail_count,
            )
