"""handlers.py — Thin routing layer for the Telegram bot.

Responsibilities (only):
  1. Authentication (``_check_credentials``, ``_is_authorized``).
  2. ``/start`` entry point.
  3. ``handle_callback`` router — dispatches callback_data to ``_actions``.
  4. Text-input receivers for ConversationHandler states.
  5. ``build_conversation_handler`` factory.

UI building blocks live in ``_ui.py``.
Action implementations live in ``_actions.py``.
"""

import logging
import os
import re
from typing import Optional

from telegram import Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from src.config.settings_loader import load_settings, save_settings

from ._actions import (
    _on_balance,
    _on_clear,
    _on_margin_toggle,
    _on_panic,
    _on_positions,
    _on_scan,
    _on_status,
    _on_train,
)
from ._ui import (
    NAVIGATING,
    WAITING_ADD_SYMBOL,
    WAITING_LEVERAGE,
    WAITING_REMOVE_SYMBOL,
    WAITING_RISK,
    _current_margin,
    _kb_bot,
    _kb_futures,
    _kb_main,
    _kb_panic_confirm,
    _txt_bot,
    _txt_exchange,
    _txt_futures,
    _txt_main,
    _kb_exchange,
)

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+_USDT$")


# --- Authentication ---


def _check_credentials() -> tuple[bool, Optional[int]]:
    """Verify that AUTHORIZED_CHAT_ID, BINANCE_API_KEY, and
    BINANCE_API_SECRET are present in the environment.

    Returns:
        (ok, chat_id) where *ok* is ``True`` if all credentials
        are present and *chat_id* is the authorized chat integer.
    """
    chat_id_str = os.getenv("AUTHORIZED_CHAT_ID")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not chat_id_str or not api_key or not api_secret:
        return False, None
    try:
        return True, int(chat_id_str)
    except ValueError:
        logger.warning("AUTHORIZED_CHAT_ID is not a valid integer: %s", chat_id_str)
        return False, None


def _is_authorized(chat_id: int) -> bool:
    ok, auth_id = _check_credentials()
    if not ok or auth_id is None:
        return False
    return chat_id == auth_id


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ok, auth_id = _check_credentials()

    if not ok or auth_id is None or update.effective_chat.id != auth_id:
        await update.message.reply_text(
            "⛔ Access Denied or incomplete credentials on the server."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        _txt_main(),
        reply_markup=_kb_main(),
        parse_mode="HTML",
    )
    return NAVIGATING


# --- Callback-query router  (NAVIGATING state) ---


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    data: str = query.data

    if not _is_authorized(update.effective_chat.id):
        await query.answer("⛔ Access Denied.", show_alert=True)
        return ConversationHandler.END

    # Spot
    if data == "menu:spot":
        await query.answer("⏳ Coming Soon...", show_alert=True)
        return NAVIGATING

    await query.answer()

    # Navigation
    if data == "menu:main":
        await query.edit_message_text(
            _txt_main(),
            reply_markup=_kb_main(),
            parse_mode="HTML",
        )
        return NAVIGATING

    if data == "menu:bot":
        paused = context.application.bot_data.get("paused", False)
        await query.edit_message_text(
            _txt_bot(),
            reply_markup=_kb_bot(paused),
            parse_mode="HTML",
        )
        return NAVIGATING

    if data == "menu:exchange":
        await query.edit_message_text(
            _txt_exchange(),
            reply_markup=_kb_exchange(),
            parse_mode="HTML",
        )
        return NAVIGATING

    if data == "menu:futures":
        await query.edit_message_text(
            _txt_futures(),
            reply_markup=_kb_futures(_current_margin()),
            parse_mode="HTML",
        )
        return NAVIGATING

    # Bot actions
    if data == "action:status":
        return await _on_status(query, context)

    if data == "action:clear":
        return await _on_clear(query, context)

    if data == "action:add_symbol":
        await query.edit_message_text(
            "➕ <b>Add Symbol</b>\n\n"
            "Send the symbol in the following format: <code>BTC_USDT</code>\n"
            "Regex: <code>^[A-Z0-9]+_USDT$</code>\n\n"
            "Type /cancel to cancel.",
            parse_mode="HTML",
        )
        return WAITING_ADD_SYMBOL

    if data == "action:remove_symbol":
        symbols = load_settings().get("futures", {}).get("symbols", [])
        listing = "\n".join(f"• <code>{s}</code>" for s in symbols) or "Empty list."
        await query.edit_message_text(
            f"🗑️ <b>Remove Symbol</b>\n\nCurrent symbols:\n{listing}\n\n"
            "Send the symbol to be removed.\nType /cancel to cancel.",
            parse_mode="HTML",
        )
        return WAITING_REMOVE_SYMBOL

    if data == "action:train":
        return await _on_train(query, context)

    if data == "action:pause":
        context.application.bot_data["paused"] = True
        await query.edit_message_text(
            "⏸️ <b>Bot PAUSED</b>\n\nAutomatic operations will not be executed.",
            reply_markup=_kb_bot(True),
            parse_mode="HTML",
        )
        return NAVIGATING

    if data == "action:resume":
        context.application.bot_data["paused"] = False
        await query.edit_message_text(
            "▶️ <b>Bot RESUMED</b>\n\nAutomatic operations activated.",
            reply_markup=_kb_bot(False),
            parse_mode="HTML",
        )
        return NAVIGATING

    # Futures actions
    if data == "action:balance":
        return await _on_balance(query, context)

    if data == "action:positions":
        return await _on_positions(query, context)

    if data == "action:scan":
        return await _on_scan(query, context)

    if data == "action:leverage":
        lev = load_settings().get("futures", {}).get("default_leverage", 1)
        await query.edit_message_text(
            f"⚙️ <b>Modify Leverage</b>\n\n"
            f"Current value: <b>{lev}x</b>\n"
            f"Send an integer between 1 and 125.\n\n"
            f"Type /cancel to cancel.",
            parse_mode="HTML",
        )
        return WAITING_LEVERAGE

    if data == "action:risk":
        risk = load_settings().get("global", {}).get("risk_per_trade_pct", 1.0)
        await query.edit_message_text(
            f"⚖️ <b>Modify Risk (%)</b>\n\n"
            f"Current value: <b>{risk}%</b>\n"
            f"Send a decimal number (e.g., 1.5 for 1.5%).\n\n"
            f"Type /cancel to cancel.",
            parse_mode="HTML",
        )
        return WAITING_RISK

    if data == "action:margin_toggle":
        return await _on_margin_toggle(query)

    if data == "action:panic":
        await query.edit_message_text(
            "⚠️ <b>PANIC BUTTON</b>\n\n"
            "This will close <b>ALL</b> open positions and cancel "
            "<b>ALL</b> orders.\n\nAre you sure?",
            reply_markup=_kb_panic_confirm(),
            parse_mode="HTML",
        )
        return NAVIGATING

    if data == "action:panic_confirm":
        return await _on_panic(query, context)

    if data == "action:panic_cancel":
        await query.edit_message_text(
            _txt_futures(),
            reply_markup=_kb_futures(_current_margin()),
            parse_mode="HTML",
        )
        return NAVIGATING

    return NAVIGATING


# --- Text-input receivers  (ConversationHandler states) ---


async def receive_add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip().upper()

    if not _SYMBOL_RE.match(text):
        await update.message.reply_text(
            "❌ Invalid format. Use: <code>SYMBOL_USDT</code>\n"
            "Example: <code>BTC_USDT</code>, <code>ETH_USDT</code>\n\n"
            "Try again or type /cancel.",
            parse_mode="HTML",
        )
        return WAITING_ADD_SYMBOL

    settings = load_settings()
    symbols: list[str] = settings.get("futures", {}).get("symbols", [])

    if text in symbols:
        await update.message.reply_text(
            f"⚠️ <code>{text}</code> already exists in the list.",
            parse_mode="HTML",
        )
        return WAITING_ADD_SYMBOL

    symbols.append(text)
    settings.setdefault("futures", {})["symbols"] = symbols
    save_settings(settings)

    paused = context.application.bot_data.get("paused", False)
    await update.message.reply_text(
        f"✅ <code>{text}</code> added successfully.",
        reply_markup=_kb_bot(paused),
        parse_mode="HTML",
    )
    return NAVIGATING


async def receive_remove_symbol(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    text = update.message.text.strip().upper()
    settings = load_settings()
    symbols: list[str] = settings.get("futures", {}).get("symbols", [])

    if text not in symbols:
        await update.message.reply_text(
            f"❌ <code>{text}</code> does not exist in the list.\n"
            "Try again or type /cancel.",
            parse_mode="HTML",
        )
        return WAITING_REMOVE_SYMBOL

    symbols.remove(text)
    settings["futures"]["symbols"] = symbols
    save_settings(settings)

    paused = context.application.bot_data.get("paused", False)
    await update.message.reply_text(
        f"✅ <code>{text}</code> removed successfully.",
        reply_markup=_kb_bot(paused),
        parse_mode="HTML",
    )
    return NAVIGATING


async def receive_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()

    try:
        value = int(text)
    except ValueError:
        await update.message.reply_text(
            "❌ Must be an integer. Try again or type /cancel.",
        )
        return WAITING_LEVERAGE

    if not 1 <= value <= 125:
        await update.message.reply_text(
            "❌ Value out of range (1–125). Try again or type /cancel.",
        )
        return WAITING_LEVERAGE

    settings = load_settings()
    settings.setdefault("futures", {})["default_leverage"] = value
    save_settings(settings)

    margin = settings.get("futures", {}).get("margin_type", "ISOLATED")
    await update.message.reply_text(
        f"✅ Leverage updated to <b>{value}x</b>",
        reply_markup=_kb_futures(margin),
        parse_mode="HTML",
    )
    return NAVIGATING


async def receive_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()

    try:
        value = float(text)
    except ValueError:
        await update.message.reply_text(
            "❌ Must be a decimal number. Example: 1.5\n" "Try again or type /cancel.",
        )
        return WAITING_RISK

    if value <= 0 or value > 100:
        await update.message.reply_text(
            "❌ Value out of range (0.01–100). Try again or type /cancel.",
        )
        return WAITING_RISK

    settings = load_settings()
    settings.setdefault("global", {})["risk_per_trade_pct"] = round(value, 2)
    save_settings(settings)

    margin = settings.get("futures", {}).get("margin_type", "ISOLATED")
    await update.message.reply_text(
        f"✅ Risk updated to <b>{value}%</b>",
        reply_markup=_kb_futures(margin),
        parse_mode="HTML",
    )
    return NAVIGATING


# /cancel helpers (CommandHandlers in each state)


async def _cancel_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    paused = context.application.bot_data.get("paused", False)
    await update.message.reply_text(
        _txt_bot(),
        reply_markup=_kb_bot(paused),
        parse_mode="HTML",
    )
    return NAVIGATING


async def _cancel_to_futures(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        _txt_futures(),
        reply_markup=_kb_futures(_current_margin()),
        parse_mode="HTML",
    )
    return NAVIGATING


# --- ConversationHandler factory ---


def build_conversation_handler() -> ConversationHandler:
    """Build and return the main ``ConversationHandler`` for the bot."""
    return ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            NAVIGATING: [CallbackQueryHandler(handle_callback)],
            WAITING_ADD_SYMBOL: [
                CommandHandler("cancel", _cancel_to_bot),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_add_symbol),
            ],
            WAITING_REMOVE_SYMBOL: [
                CommandHandler("cancel", _cancel_to_bot),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_remove_symbol),
            ],
            WAITING_LEVERAGE: [
                CommandHandler("cancel", _cancel_to_futures),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_leverage),
            ],
            WAITING_RISK: [
                CommandHandler("cancel", _cancel_to_futures),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_risk),
            ],
        },
        fallbacks=[CommandHandler("start", start_command)],
        per_message=False,
    )
