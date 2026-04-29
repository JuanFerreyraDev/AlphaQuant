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

from src.config.settings_loader import load_settings, save_settings

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
    paused = context.application.bot_data.get("paused", False)
    label = "⏸️ Paused" if paused else "✅ Active"
    n_sym = len(load_settings().get("futures", {}).get("symbols", []))
    await query.edit_message_text(
        f"📊 <b>Bot Status</b>\n\n"
        f"Status: <b>{label}</b>\n"
        f"Monitored Symbols: <b>{n_sym}</b>\n\n"
        f"⏳ Waiting for daily market close.",
        reply_markup=_kb_back("menu:main"), parse_mode="HTML",
    )
    return NAVIGATING


async def _on_clear(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
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
        chat_id=chat_id, text=_txt_main(),
        reply_markup=_kb_main(), parse_mode="HTML",
    )
    return NAVIGATING


async def _on_train(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    paused = context.application.bot_data.get("paused", False)
    await query.edit_message_text(
        "⏳ Starting training in background...",
        reply_markup=_kb_back("menu:bot"), parse_mode="HTML",
    )

    async def _run() -> None:
        try:
            from src.config.settings_loader import get_active_symbols as _symbols
            from src.engine.tasks import (
                _check_training_freshness,
                _sanitize_symbol,
                run_full_training_pipeline,
            )

            symbols = _symbols()
            if not symbols:
                await query.edit_message_text(
                    "❌ No active symbols to train.",
                    reply_markup=_kb_bot(paused), parse_mode="HTML",
                )
                return

            results: list[str] = []
            for s in symbols:
                safe = _sanitize_symbol(s)
                needs_training, reason = _check_training_freshness(safe)

                if not needs_training:
                    results.append(
                        f"⏭️ {html.escape(safe)}: Omitted ({html.escape(reason)})."
                    )
                    continue

                try:
                    await asyncio.to_thread(run_full_training_pipeline, s)
                    results.append(
                        f"✅ {html.escape(safe)}: Trained and updated."
                    )
                except Exception as exc:
                    logger.error("Error in pipeline of %s: %s", safe, exc)
                    results.append(
                        f"❌ {html.escape(safe)}: Error during the "
                        f"optimization/training."
                    )

            summary = (
                "📊 <b>Training Summary MLOps:</b>\n\n"
                + "\n".join(results)
            )
            await query.edit_message_text(
                summary,
                reply_markup=_kb_bot(paused), parse_mode="HTML",
            )
        except Exception as exc:
            logger.error("Error in training: %s", exc)
            await query.edit_message_text(
                f"❌ Error during training: <code>{html.escape(str(exc))}</code>",
                reply_markup=_kb_bot(paused), parse_mode="HTML",
            )

    asyncio.create_task(_run())
    return NAVIGATING


async def _on_balance(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    executor = context.application.bot_data.get("executor")
    if executor is None:
        await query.edit_message_text(
            "❌ Exchange not available (credentials not configured).",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
        return NAVIGATING
    try:
        balance = await asyncio.to_thread(executor.get_futures_balance)
        await query.edit_message_text(
            f"💰 <b>USDT Balance (Futures)</b>\n\n"
            f"Available: <b>{balance:.4f} USDT</b>",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Error consulting balance: %s", exc)
        await query.edit_message_text(
            "❌ Error connecting to the Exchange.",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
    return NAVIGATING


async def _on_positions(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    executor = context.application.bot_data.get("executor")
    if executor is None:
        await query.edit_message_text(
            "❌ Exchange not available (credentials not configured).",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
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
                lines.append(
                    f"• <b>{sym}</b>: {amt} @ {entry} | PnL: {pnl:.4f}"
                )
            text = "📋 <b>Open Positions</b>\n\n" + "\n".join(lines)

        await query.edit_message_text(
            text, reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Error consulting positions: %s", exc)
        await query.edit_message_text(
            "❌ Error connecting to the Exchange.",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
    return NAVIGATING


async def _on_scan(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    evaluate_fn = context.application.bot_data.get("evaluate_fn")
    if evaluate_fn is None:
        await query.edit_message_text(
            "❌ Error internal: evaluation function not configured.",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
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
        result, reply_markup=_kb_futures(_current_margin()), parse_mode="HTML",
    )
    return NAVIGATING


async def _on_margin_toggle(query: Any) -> int:
    settings = load_settings()
    current = settings.get("futures", {}).get("margin_type", "ISOLATED")
    new_margin = "CROSSED" if current == "ISOLATED" else "ISOLATED"
    settings.setdefault("futures", {})["margin_type"] = new_margin
    save_settings(settings)
    await query.edit_message_text(
        _txt_futures(), reply_markup=_kb_futures(new_margin), parse_mode="HTML",
    )
    return NAVIGATING


async def _on_panic(query: Any, context: ContextTypes.DEFAULT_TYPE) -> int:
    executor = context.application.bot_data.get("executor")
    if executor is None:
        await query.edit_message_text(
            "❌ Exchange not available (credentials not configured).",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
        return NAVIGATING
    try:
        closed = await asyncio.to_thread(executor.close_all_positions)
        await query.edit_message_text(
            f"🚨 <b>PPanic executed</b>\n\n"
            f"Closed positions: <b>{closed}</b>\n"
            f"Open orders: <b>Cancelled</b>",
            reply_markup=_kb_futures(_current_margin()), parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Error in panic button: %s", exc)
        await query.edit_message_text(
            "❌ Error connecting to the Exchange.",
            reply_markup=_kb_back("menu:futures"), parse_mode="HTML",
        )
    return NAVIGATING
