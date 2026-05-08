"""logging_config.py — Centralized logging configuration for AlphaQuant.

Invoked ONCE from main.py at application startup.
Configures a StreamHandler (console) and a RotatingFileHandler (logs/alphaquant.log).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the global logging system for the application.

    Args:
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"alphaquant_{datetime.now().strftime('%Y-%m-%d')}.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
