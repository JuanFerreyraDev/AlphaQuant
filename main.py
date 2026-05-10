"""main.py — Minimal entry point for AlphaQuant.

Responsibilities:
  1. Set up centralized logging.
  2. Load environment variables.
  3. Build the Telegram application.
  4. Inject dependencies into bot_data.
  5. Schedule the daily evaluation with APScheduler.
  6. Start the bot.
"""

import logging
import os

from dotenv import load_dotenv
from pytz import timezone
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from src.api.telegram.handlers import build_conversation_handler
from src.config.settings_loader import load_bot_state
from src.engine.tasks import daily_market_evaluation, run_full_training_pipeline
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


async def _error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch global bot errors and log them without crashing."""
    logger.error("Bot error detected: %s", context.error)


async def _post_init(app: Application) -> None:
    """Post-initialization hook: configure the daily evaluation scheduler
    and inject shared dependencies into ``bot_data``."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    app.bot_data["auth_chat_id"] = int(os.getenv("AUTHORIZED_CHAT_ID", "0"))
    app.bot_data["paused"] = not load_bot_state().get("bot_active", True)
    logger.info(
        "Bot active state loaded from bot_state.json (paused=%s).",
        app.bot_data["paused"],
    )

    try:
        from src.api.binance.binance_executor import BinanceExecutor

        app.bot_data["executor"] = BinanceExecutor()
        logger.info("BinanceExecutor injected into bot_data.")
    except (ValueError, Exception) as exc:
        app.bot_data["executor"] = None
        logger.warning(
            "BinanceExecutor not available: %s — running in signal-only mode.", exc
        )

    logger.info("Configuring AsyncIOScheduler...")
    tz = timezone("America/Argentina/Cordoba")
    scheduler = AsyncIOScheduler(timezone=tz)

    chat_id: int = int(os.getenv("AUTHORIZED_CHAT_ID", "0"))

    scheduler.add_job(
        daily_market_evaluation,
        trigger="cron",
        hour=21,
        minute=0,
        args=[app, chat_id],
    )

    scheduler.start()
    logger.info("AsyncIOScheduler started. Task scheduled at 21:00.")


def main() -> None:
    """Main application entry point."""
    setup_logging(log_level="INFO")

    load_dotenv()

    token: str = os.getenv("TELEGRAM_TOKEN", "")
    chat_id_str: str = os.getenv("AUTHORIZED_CHAT_ID", "")

    if not token or not chat_id_str:
        logger.error(
            "Missing environment variables TELEGRAM_TOKEN or AUTHORIZED_CHAT_ID."
        )
        return

    try:
        chat_id = int(chat_id_str)
    except ValueError:
        logger.error("AUTHORIZED_CHAT_ID must be an integer.")
        return

    logger.info("Initializing ApplicationBuilder...")
    app: Application = (
        ApplicationBuilder()
        .token(token)
        .read_timeout(30)
        .write_timeout(30)
        .post_init(_post_init)
        .build()
    )

    app.bot_data["evaluate_fn"] = daily_market_evaluation
    app.bot_data["train_pipeline"] = run_full_training_pipeline

    app.add_error_handler(_error_handler)
    app.add_handler(build_conversation_handler())

    logger.info("Algorithmic Trading Bot started successfully.")
    app.run_polling()


if __name__ == "__main__":
    main()
