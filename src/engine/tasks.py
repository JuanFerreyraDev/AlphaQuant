"""tasks.py — Daily market evaluation orchestrator.

THIS is the ONLY module that imports from both ``src.api.binance`` and
``src.api.telegram``, serving as a bridge between the brain and external APIs.

Responsibility:
  1. Read the configuration from settings.yaml (active market, symbols).
  2. For each active symbol, load the .pkl model and evaluate the latest candle.
  3. If a signal is detected, calculate TP/SL and execute the trade via BinanceExecutor.
  4. Notify via Telegram both the signals and the execution results.
"""

import asyncio
import datetime
import glob
import json
import logging
import os
from typing import Any, Optional

import ccxt.async_support as ccxt_async
import joblib
import pandas as pd

from src.api.binance.binance_executor import BinanceExecutor
from src.api.telegram.notifier import (
    send_execution_error,
    send_execution_result,
    send_trade_signal,
)
from src.brain.data_fetcher import (
    fetch_historical_data,
    fetch_ohlcv_binance,
    get_fear_and_greed,
)
from src.brain.features import compute_all_technicals, add_sentiment
from src.brain.strategy_optimizer import optimize_strategy
from src.brain.train import train_factory
from src.config.settings_loader import (
    get_active_market,
    get_active_symbols,
    get_project_root,
)

logger = logging.getLogger(__name__)

TRAINING_COOLDOWN_DAYS: int = 14

REQUIRED_KEYS: list[str] = [
    "model",
    "features",
    "threshold",
    "atr_tp_multi",
    "atr_sl_multi",
]


def run_full_training_pipeline(symbol: str) -> tuple[bool, str, str]:
    """Execute the complete training pipeline for a symbol.

    Strict order:
      1. ``fetch_historical_data`` — download fresh data.
      2. ``optimize_strategy`` — search for a new strategy and write
         ``last_trained`` to ``config.json``.
      3. ``train_factory`` — train the model and save the ``.pkl``.

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'``).

    Raises:
        RuntimeError: If any of the 3 steps fails.

    Returns:
        Tuple ``(trained, safe_symbol, reason)``.
        If ``trained`` is ``False``, ``reason`` describes why it was skipped.
    """
    safe_symbol = _sanitize_symbol(symbol)
    needs_training, reason = _check_training_freshness(safe_symbol)
    if not needs_training:
        logger.info("[Pipeline] Skipping %s: %s", safe_symbol, reason)
        return needs_training, safe_symbol, reason
    logger.info("[Pipeline] Step 1/3: Downloading data for %s...", safe_symbol)
    fetch_historical_data(safe_symbol)

    df_fg = get_fear_and_greed()

    logger.info("[Pipeline] Step 2/3: Optimizing strategy for %s...", safe_symbol)
    optimize_strategy(safe_symbol, df_fg)

    logger.info("[Pipeline] Step 3/3: Training model for %s...", safe_symbol)
    train_factory(safe_symbol, df_fg)

    logger.info("[Pipeline] Pipeline complete for %s.", safe_symbol)

    return needs_training, safe_symbol, ""


def _sanitize_symbol(symbol: str) -> str:
    """Convert any symbol format to a file-safe format.

    Args:
        symbol: Symbol in any format (e.g. ``'BTC/USDT:USDT'``).

    Returns:
        Normalized format (e.g. ``'BTC_USDT'``).
    """
    return symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"


def _check_training_freshness(safe_symbol: str) -> tuple[bool, str]:
    """Check whether a symbol needs retraining.

    Reads ``data/models/{safe_symbol}/config.json`` and compares the
    ``last_trained`` field with the current date.

    Args:
        safe_symbol: Normalized symbol (e.g. ``'BTC_USDT'``).

    Returns:
        Tuple ``(needs_training, reason)``.  If ``needs_training`` is
        ``False``, ``reason`` describes why it was skipped.
    """
    base_dir = get_project_root()
    config_path = base_dir / "data" / "models" / safe_symbol / "config.json"

    if not config_path.exists():
        return True, ""

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return True, ""

    last_trained_str = config.get("last_trained")
    if not last_trained_str:
        return True, ""

    try:
        last_trained_dt = datetime.datetime.fromisoformat(last_trained_str)
        if last_trained_dt.tzinfo is None:
            last_trained_dt = last_trained_dt.replace(tzinfo=datetime.timezone.utc)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        days_elapsed = (now_utc - last_trained_dt).days
    except (ValueError, TypeError):
        return True, ""

    if days_elapsed < TRAINING_COOLDOWN_DAYS:
        return (
            False,
            f"Trained {days_elapsed} day(s) ago, less than {TRAINING_COOLDOWN_DAYS} days",
        )

    return True, ""


async def daily_market_evaluation(app: Any, chat_id: int) -> int:
    """Main daily market evaluation logic.

    Iterates over active symbols, evaluates each loaded .pkl model,
    and executes trades on Binance Futures when a signal is detected.

    Args:
        app: ``telegram.ext.Application`` instance.
        chat_id: Authorized Telegram chat ID.

    Returns:
        Number of signals sent.
    """
    active_market: str = get_active_market()
    active_symbols: list[str] = get_active_symbols()
    base_models_dir = os.path.join("data", "models")

    if not active_symbols:
        logger.warning("No active symbols in settings.yaml.")
        return 0

    if not os.path.exists(base_models_dir):
        logger.warning("Models directory not found: %s", base_models_dir)
        return 0

    logger.info(
        "Daily evaluation started | Market: %s | Symbols: %s",
        active_market,
        [_sanitize_symbol(s) for s in active_symbols],
    )

    executor: Optional[BinanceExecutor] = None
    if active_market in ("futures", "both"):
        executor = await asyncio.to_thread(_init_executor)

    df_fg: pd.DataFrame = await asyncio.to_thread(get_fear_and_greed)

    signals_sent: int = 0

    exchange = ccxt_async.binanceusdm({"enableRateLimit": True})
    try:
        for symbol in active_symbols:
            safe_symbol = _sanitize_symbol(symbol)
            symbol_dir = os.path.join(base_models_dir, safe_symbol)

            if not os.path.isdir(symbol_dir):
                logger.info("No trained models for %s, skipping.", safe_symbol)
                continue

            pkl_files: list[str] = glob.glob(os.path.join(symbol_dir, "*.pkl"))
            if not pkl_files:
                logger.info("No .pkl files for %s, skipping.", safe_symbol)
                continue

            for model_path in pkl_files:
                model_name = os.path.basename(model_path)
                try:
                    signals_sent += await _evaluate_model(
                        app,
                        chat_id,
                        executor,
                        active_market,
                        safe_symbol,
                        model_path,
                        model_name,
                        df_fg,
                        exchange,
                    )
                except Exception as exc:
                    logger.error(
                        "Error processing %s with model %s: %s",
                        safe_symbol,
                        model_name,
                        exc,
                    )
    finally:
        await exchange.close()

    logger.info("Evaluation finished. Signals sent: %d", signals_sent)
    return signals_sent


def _init_executor() -> Optional[BinanceExecutor]:
    """Attempt to initialize the BinanceExecutor.

    Returns:
        ``BinanceExecutor`` instance or ``None`` on failure.
    """
    try:
        executor = BinanceExecutor()
        logger.info("BinanceExecutor initialized successfully.")
        return executor
    except (ValueError, ConnectionError) as exc:
        logger.warning(
            "Could not initialize BinanceExecutor: %s. "
            "The bot will operate in signal-only mode (Telegram).",
            exc,
        )
        return None


async def _evaluate_model(
    app: Any,
    chat_id: int,
    executor: Optional[BinanceExecutor],
    active_market: str,
    safe_symbol: str,
    model_path: str,
    model_name: str,
    df_fg: pd.DataFrame,
    exchange: Optional[ccxt_async.binanceusdm] = None,
) -> int:
    """Evaluate a single .pkl model and execute a trade if a signal is detected.

    Args:
        app: Telegram Application instance.
        chat_id: Authorized chat ID.
        executor: BinanceExecutor instance (may be None).
        active_market: Current active market.
        safe_symbol: Normalized symbol.
        model_path: Path to the .pkl file.
        model_name: Name of the .pkl file.
        df_fg: Fear & Greed DataFrame (may be empty).
        exchange: Optional shared ``ccxt_async.binanceusdm`` instance.

    Returns:
        ``1`` if a signal was sent, ``0`` otherwise.
    """
    model_dict: dict[str, Any] = joblib.load(model_path)

    if not all(k in model_dict for k in REQUIRED_KEYS):
        logger.warning(
            "[IGNORED] %s does not have all the risk parameters.", model_path
        )
        return 0

    model = model_dict["model"]
    features: list[str] = model_dict["features"]
    threshold: float = model_dict["threshold"]
    atr_tp_multi: float = model_dict["atr_tp_multi"]
    atr_sl_multi: float = model_dict["atr_sl_multi"]
    strategy_name: str = model_dict.get("strategy_name", model_name.replace(".pkl", ""))

    if not model or not features:
        logger.warning("Invalid .pkl file: %s", model_path)
        return 0

    df = await fetch_ohlcv_binance(
        safe_symbol, timeframe="1d", limit=100, exchange=exchange
    )
    if df is None or df.empty:
        logger.warning("Missing or empty data for %s.", safe_symbol)
        return 0

    df.columns = df.columns.str.lower()
    df.set_index("timestamp", inplace=True)

    df = await asyncio.to_thread(compute_all_technicals, df)

    df, _ = add_sentiment(df, df_fg)
    if not df_fg.empty:
        df["fng_value"] = df["fng_value"].ffill()

    last_candle = df.iloc[-1:]
    current_price = float(last_candle["close"].iloc[0])

    atr_value: float = 0.0
    for col in ("atr_14", "atr"):
        if col in last_candle.columns:
            atr_value = float(last_candle[col].iloc[0])
            break
    if atr_value == 0:
        atr_value = float(last_candle["high"].iloc[0] - last_candle["low"].iloc[0])

    try:
        vela_filtrada = last_candle[features]
    except KeyError as exc:
        logger.warning("[%s] Missing features for %s: %s", safe_symbol, model_name, exc)
        return 0

    proba_array = model.predict_proba(vela_filtrada)
    proba: float = float(
        proba_array[0, 1] if proba_array.shape[1] > 1 else proba_array[0][0]
    )

    if proba <= threshold:
        return 0

    tp = current_price + (atr_value * atr_tp_multi)
    sl = current_price - (atr_value * atr_sl_multi)

    logger.info(
        "[%s] Signal detected by %s (Prob: %.4f > %.4f) | "
        "Price: %.4f | TP: %.4f | SL: %.4f",
        safe_symbol,
        strategy_name,
        proba,
        threshold,
        current_price,
        tp,
        sl,
    )

    signal_data: dict[str, Any] = {
        "symbol": safe_symbol.replace("_", "/"),
        "strategy": strategy_name,
        "price": current_price,
        "tp": tp,
        "sl": sl,
    }
    await send_trade_signal(app, chat_id, signal_data)

    if executor is not None and active_market in ("futures", "both"):
        await _execute_and_notify(app, chat_id, executor, safe_symbol, sl, tp)

    return 1


async def _execute_and_notify(
    app: Any,
    chat_id: int,
    executor: BinanceExecutor,
    safe_symbol: str,
    sl: float,
    tp: float,
) -> None:
    """Execute the trade on Binance and send the result via Telegram.

    Args:
        app: Telegram Application instance.
        chat_id: Authorized chat ID.
        executor: BinanceExecutor instance.
        safe_symbol: Normalized symbol.
        sl: Stop Loss price.
        tp: Take Profit price.
    """
    try:
        result = await asyncio.to_thread(
            executor.execute_futures_trade,
            safe_symbol,
            "BUY",
            sl,
            tp,
        )
        await send_execution_result(app, chat_id, result, safe_symbol)

    except (ConnectionError, TimeoutError, ValueError) as exec_err:
        logger.warning("Error executing trade for %s: %s", safe_symbol, exec_err)
        await send_execution_error(app, chat_id, safe_symbol, exec_err)
