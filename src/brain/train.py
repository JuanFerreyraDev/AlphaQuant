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
from src.config.settings_loader import get_active_symbols, get_project_root
from src.utils.helpers import (
    DEFAULT_HP,
    cleanup_columns,
    compute_target,
    load_csv_data,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train_factory(symbol: str, df_fg: pd.DataFrame | None = None) -> None:
    """Train a model based on the configuration saved in config.json.

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'`` or ``'BTC/USDT'``).
        df_fg: Fear & Greed DataFrame (injected by the caller).
    """
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

    logger.info("Starting Training Factory for %s...", symbol)
    logger.info("Strategy: %s", strategy_name)

    filename = f"{safe_symbol}_1d.csv"
    try:
        df = load_csv_data(filename)
    except FileNotFoundError:
        raise RuntimeError(f"Data file not found for {symbol}")

    compute_all_technicals(df)
    df, _ = add_sentiment(df, df_fg if df_fg is not None else pd.DataFrame())

    compute_target(
        df,
        swing_days=swing_period,
        atr_tp_multi=atr_tp_multi,
        atr_sl_multi=atr_sl_multi,
    )
    cleanup_columns(df)

    X = df[features_list]
    y = df["target"]

    imbalance = sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=DEFAULT_HP["n_estimators"],
        max_depth=DEFAULT_HP["max_depth"],
        learning_rate=DEFAULT_HP["learning_rate"],
        scale_pos_weight=imbalance,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training XGBClassifier with %d features...", len(features_list))
    model.fit(X, y)

    production_bundle: dict[str, Any] = {
        "model": model,
        "features": features_list,
        "threshold": optimal_threshold,
        "atr_tp_multi": atr_tp_multi,
        "atr_sl_multi": atr_sl_multi,
        "strategy_name": strategy_name,
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
    args = parser.parse_args()

    from src.brain.data_fetcher import get_fear_and_greed

    fng_data = get_fear_and_greed()

    if args.symbol:
        train_factory(args.symbol, fng_data)
    else:
        symbols = get_active_symbols()
        if not symbols:
            logger.error("No symbol found in arguments or settings.yaml.")
        else:
            logger.info("Batch Mode: Training %d models...", len(symbols))
            ok_count = 0
            fail_count = 0
            for s in symbols:
                try:
                    train_factory(s, fng_data)
                    ok_count += 1
                except (KeyError, ValueError, IOError) as exc:
                    logger.error("Error training %s: %s", s, exc)
                    fail_count += 1
                    continue

            logger.info(
                "Batch training finished. Success: %d | Failed: %d",
                ok_count,
                fail_count,
            )
