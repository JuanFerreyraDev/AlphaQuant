"""notifier.py — Telegram notification sender.

DECOUPLING RULE: This module does NOT import anything from ``src.api.binance``.
"""

import html
import logging
from typing import Any, Optional

import telegram.error

logger = logging.getLogger(__name__)


async def send_trade_signal(app: Any, chat_id: int, signal_data: dict[str, Any]) -> None:
    """Send a trading signal via Telegram using HTML syntax.

    Args:
        app: ``telegram.ext.Application`` instance.
        chat_id: Destination chat ID.
        signal_data: Dictionary with keys ``symbol``, ``strategy``,
            ``price``, ``tp``, ``sl``.
    """
    symbol = html.escape(str(signal_data.get("symbol", "UNKNOWN")))
    strategy = html.escape(str(signal_data.get("strategy", "Unknown Strategy")))

    price: float = signal_data.get("price", 0.0)
    tp: float = signal_data.get("tp", 0.0)
    sl: float = signal_data.get("sl", 0.0)

    message = (
        f"🤖 <b>New trading signal</b>\n\n"
        f"🔹 <b>Pair:</b> {symbol}\n"
        f"🧠 <b>Strategy:</b> {strategy}\n\n"
        f"🟢 <b>Buy (Entry):</b> {price:.4f}\n"
        f"🎯 <b>Take Profit (TP):</b> {tp:.4f}\n"
        f"🛡️ <b>Stop Loss (SL):</b> {sl:.4f}\n"
    )

    try:
        await app.bot.send_message(
            chat_id=chat_id, text=message, parse_mode="HTML"
        )
    except telegram.error.TelegramError as exc:
        if isinstance(exc, telegram.error.NetworkError):
            logger.warning("Network error sending signal to Telegram: %s", exc)
        else:
            logger.error("Error sending signal to Telegram (chat_id=%d): %s", chat_id, exc)


async def send_execution_result(
    app: Any,
    chat_id: int,
    result: Optional[dict[str, Any]],
    safe_symbol: str,
) -> None:
    """Send the Binance trade execution result via Telegram.

    Args:
        app: ``telegram.ext.Application`` instance.
        chat_id: Destination chat ID.
        result: Return value of ``BinanceExecutor.execute_futures_trade()``,
            or ``None`` if the trade was skipped.
        safe_symbol: Normalized symbol (e.g. ``'BTC_USDT'``).
    """
    display = html.escape(safe_symbol.replace("_", "/"))

    if result:
        sl_id = result["stop_loss"].get("clientAlgoId", "N/A") if result.get("stop_loss") else "N/A"
        tp_id = result["take_profit"].get("clientAlgoId", "N/A") if result.get("take_profit") else "N/A"
        text = (
            f"✅ <b>Order executed in Binance Futures</b>\n\n"
            f"🔹 <b>Pair:</b> {display}\n"
            f"📋 <b>Entry ID:</b> {result['entry'].get('orderId', 'N/A')}\n"
            f"🛡️ <b>SL ID:</b> {sl_id}\n"
            f"🎯 <b>TP ID:</b> {tp_id}"
        )
    else:
        text = (
            f"⚠️ <b>Trade no executed for {display}</b>\n"
            f"Reason: Position already open, insufficient balance, "
            f"or exchange filter not met. Check logs."
        )

    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except telegram.error.TelegramError as exc:
        logger.warning("Could not send execution result to Telegram: %s", exc)


async def send_execution_error(
    app: Any,
    chat_id: int,
    safe_symbol: str,
    error: Exception,
) -> None:
    """Send a trade execution error alert via Telegram.

    Args:
        app: ``telegram.ext.Application`` instance.
        chat_id: Destination chat ID.
        safe_symbol: Normalized symbol (e.g. ``'BTC_USDT'``).
        error: The exception that caused the failure.
    """
    display = html.escape(safe_symbol.replace("_", "/"))
    text = (
        f"⚠️ <b>Signal in {display} ignored</b>\n"
        f"Execution error: <code>{html.escape(str(error))}</code>"
    )
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except telegram.error.TelegramError as exc:
        logger.warning("Could not send error alert to Telegram: %s", exc)
