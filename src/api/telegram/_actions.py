"""_actions.py — Action handler implementations for the Telegram bot.

Each ``_on_*`` function handles a single user interaction (a button press
or triggered action) delegated from the main ``handle_callback`` router in
``handlers.py``.  All functions share the signature ``(query, context) -> int``
and return a ConversationHandler state constant defined in ``_ui.py``.
"""

import asyncio
import html
import logging
from typing import Any

from telegram.ext import ContextTypes

from src.config.settings_loader import (
    load_bot_state,
    load_settings,
    save_bot_state,
    get_active_symbols,
)

from ._ui import (
    NAVIGATING,
    _current_margin,
    _kb_back,
    _kb_bot,
    _kb_futures,
    _kb_main,
    _txt_futures,
    _txt_main,
)

logger = logging.getLogger(__name__)


async def _on_status(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Display the current bot status.

    Shows whether the bot is active or paused and the number of symbols
    being monitored.  Renders an inline back button to the main menu.

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context carrying ``bot_data``.

    Returns:
        ``NAVIGATING`` state constant.
    """
    paused = context.application.bot_data.get("paused", False)
    label = "⏸️ Paused" if paused else "✅ Active"
    n_sym = len(load_bot_state().get("symbols", {}).get("futures", []))
    await query.edit_message_text(
        f"📊 <b>Bot Status</b>\n\n"
        f"Status: <b>{label}</b>\n"
        f"Monitored Symbols: <b>{n_sym}</b>\n\n"
        f"⏳ Waiting for daily market close.",
        reply_markup=_kb_back("menu:main"),
        parse_mode="HTML",
    )
    return NAVIGATING


async def _on_clear(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Delete the last 50 messages in the chat and redisplay the main menu.

    Iterates backwards from the current message ID and silently ignores
    messages that cannot be deleted (e.g. already deleted or too old).

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context used to call ``bot.delete_message``.

    Returns:
        ``NAVIGATING`` state constant.
    """
    chat_id = query.message.chat_id
    msg_id = query.message.message_id
    deleted = 0
    for i in range(50):
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id - i)
            deleted += 1
        except Exception:
            continue
    logger.info("Deleted %d messages in chat %d.", deleted, chat_id)
    await context.bot.send_message(
        chat_id=chat_id,
        text=_txt_main(),
        reply_markup=_kb_main(),
        parse_mode="HTML",
    )
    return NAVIGATING


async def _on_train(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Launch the MLOps training pipeline for all active symbols.

    Retrieves the ``train_pipeline`` callable from ``bot_data`` and runs it
    for each active symbol in a background task.  Progress is reported by
    editing the original message.  If no pipeline function is configured or
    no symbols are active, an error message is shown instead.

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context carrying ``bot_data`` (``train_pipeline`` key).

    Returns:
        ``NAVIGATING`` state constant (immediately, before training finishes).
    """
    paused = context.application.bot_data.get("paused", False)

    train_pipeline = context.application.bot_data.get("train_pipeline")

    if train_pipeline is None:
        await query.edit_message_text(
            "❌ Internal Error: Training pipeline function not configured.",
            reply_markup=_kb_bot(paused),
            parse_mode="HTML",
        )
        return NAVIGATING

    await query.edit_message_text(
        "⏳ Starting training pipeline...\n\n",
        reply_markup=_kb_back("menu:bot"),
        parse_mode="HTML",
    )

    async def _run() -> None:
        try:
            symbols = get_active_symbols()
            if not symbols:
                await query.edit_message_text(
                    "❌ There are no active symbols to train.",
                    reply_markup=_kb_bot(paused),
                    parse_mode="HTML",
                )
                return

            results: list[str] = []
            for s in symbols:
                try:

                    needs_training, safe, reason = await asyncio.to_thread(
                        train_pipeline, s
                    )

                    if not needs_training:
                        results.append(
                            f"⏭️ {html.escape(safe)}: Omitted ({html.escape(reason)})."
                        )
                        continue

                    results.append(f"✅ {html.escape(safe)}: Trained and up-to-date.")
                except Exception as exc:
                    logger.error("Pipe error %s: %s", safe, exc)
                    results.append(
                        f"❌ {html.escape(safe)}: Error during the "
                        f" optimization/training."
                    )

            summary = "📊 <b>MLOps Training Summary:</b>\n\n" + "\n".join(results)
            await query.edit_message_text(
                summary,
                reply_markup=_kb_bot(paused),
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.error("Error in training: %s", exc)
            await query.edit_message_text(
                f"❌ Error during training: <code>{html.escape(str(exc))}</code>",
                reply_markup=_kb_bot(paused),
                parse_mode="HTML",
            )

    asyncio.create_task(_run())
    return NAVIGATING


async def _on_balance(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fetch and display the available USDT futures balance.

    Calls ``executor.get_futures_balance`` in a thread and renders the
    result.  Shows an error message if the executor is not configured or
    if the exchange request fails.

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context carrying ``bot_data`` (``executor`` key).

    Returns:
        ``NAVIGATING`` state constant.
    """
    executor = context.application.bot_data.get("executor")
    if executor is None:
        await query.edit_message_text(
            "❌ Exchange not available (credentials not configured).",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
        return NAVIGATING
    try:
        balance = await asyncio.to_thread(executor.get_futures_balance)
        await query.edit_message_text(
            f"💰 <b>USDT Balance (Futures)</b>\n\n"
            f"Available: <b>{balance:.4f} USDT</b>",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Error consulting balance: %s", exc)
        await query.edit_message_text(
            "❌ Error connecting to the Exchange.",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
    return NAVIGATING


async def _on_positions(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fetch and display all currently open futures positions.

    Calls ``executor.get_open_positions`` in a thread and renders each
    position with its symbol, amount, entry price and unrealised PnL.
    Shows a friendly message when there are no open positions.

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context carrying ``bot_data`` (``executor`` key).

    Returns:
        ``NAVIGATING`` state constant.
    """
    executor = context.application.bot_data.get("executor")
    if executor is None:
        await query.edit_message_text(
            "❌ Exchange not available (credentials not configured).",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
        return NAVIGATING
    try:
        open_pos = await asyncio.to_thread(executor.get_open_positions)

        if not open_pos:
            text = "ℹ️ No open positions at the moment."
        else:
            lines: list[str] = []
            for p in open_pos:
                sym = html.escape(str(p["symbol"]))
                amt = p["positionAmt"]
                entry = p.get("entryPrice", "N/A")
                pnl = float(p.get("unRealizedProfit", 0))
                lines.append(f"• <b>{sym}</b>: {amt} @ {entry} | PnL: {pnl:.4f}")
            text = "📋 <b>Open Positions</b>\n\n" + "\n".join(lines)

        await query.edit_message_text(
            text,
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Error consulting positions: %s", exc)
        await query.edit_message_text(
            "❌ Error connecting to the Exchange.",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
    return NAVIGATING


async def _on_scan(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Trigger a manual intraday market scan and report detected signals.

    Calls the ``evaluate_fn`` callable from ``bot_data`` with the current
    application and chat ID.  The function returns the number of signals
    dispatched, which is then reported back to the user.  If no evaluation
    function is configured, an error message is shown instead.

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context carrying ``bot_data`` (``evaluate_fn`` key).

    Returns:
        ``NAVIGATING`` state constant.
    """
    evaluate_fn = context.application.bot_data.get("evaluate_fn")
    if evaluate_fn is None:
        await query.edit_message_text(
            "❌ Error internal: evaluation function not configured.",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
        return NAVIGATING

    chat_id = query.message.chat_id
    await query.edit_message_text(
        "⚠️ INTRADAY MODE (Manual Scan)\n"
        "The current daily candle has not closed yet.\n"
        "🔍 Scanning market..."
    )

    signals_sent: int = await evaluate_fn(context.application, chat_id)

    if signals_sent == 0:
        result = "📉 Scan completed: No signals detected that exceed the threshold."
    else:
        result = f"🏁 Scan completed: {signals_sent} signal(s) sent."

    await query.message.reply_text(
        result,
        reply_markup=_kb_futures(_current_margin()),
        parse_mode="HTML",
    )
    return NAVIGATING


async def _on_margin_toggle(query: Any) -> int:
    """Toggle the futures margin type between ISOLATED and CROSSED.

    Reads the current ``margin_type`` from settings, flips it to the other
    value, persists the change via ``save_settings``, and refreshes the
    futures menu to reflect the new state.

    Args:
        query: The ``CallbackQuery`` that triggered this action.

    Returns:
        ``NAVIGATING`` state constant.
    """
    state = load_bot_state()
    current = state.get("margin_type", "ISOLATED")
    new_margin = "CROSSED" if current == "ISOLATED" else "ISOLATED"
    state["margin_type"] = new_margin
    save_bot_state(state)
    await query.edit_message_text(
        _txt_futures(),
        reply_markup=_kb_futures(new_margin),
        parse_mode="HTML",
    )
    return NAVIGATING


async def _on_panic(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Execute the panic button: close all open positions immediately.

    Calls ``executor.close_all_positions`` in a thread, which closes every
    open futures position and cancels pending orders.  Reports the number
    of closed positions on success, or an error message if the exchange
    request fails or the executor is not available.

    Args:
        query: The ``CallbackQuery`` that triggered this action.
        context: PTB context carrying ``bot_data`` (``executor`` key).

    Returns:
        ``NAVIGATING`` state constant.
    """
    executor = context.application.bot_data.get("executor")
    if executor is None:
        await query.edit_message_text(
            "❌ Exchange not available (credentials not configured).",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
        return NAVIGATING
    try:
        closed = await asyncio.to_thread(executor.close_all_positions)
        await query.edit_message_text(
            f"🚨 <b>Panic executed</b>\n\n"
            f"Closed positions: <b>{closed}</b>\n"
            f"Open orders: <b>Cancelled</b>",
            reply_markup=_kb_futures(_current_margin()),
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Error in panic button: %s", exc)
        await query.edit_message_text(
            "❌ Error connecting to the Exchange.",
            reply_markup=_kb_back("menu:futures"),
            parse_mode="HTML",
        )
    return NAVIGATING
