"""timeframe_utils.py — Timeframe parsing and validation utilities.

These helpers ensure that every timeframe string used in the pipeline is
valid on the target exchange and can be converted to a numeric hours value
for downstream calculations (e.g. ``compute_target``'s lookahead window).
"""

import logging
import re
from typing import Optional

import ccxt

from src.config.settings_loader import get_active_market

logger = logging.getLogger(__name__)

# Regex that covers all ccxt timeframe strings we're likely to encounter.
# Examples: "1m", "5m", "15m", "1h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"
_TF_PATTERN = re.compile(r"^(\d+)([smhdwM])$")

_UNIT_TO_HOURS: dict[str, float] = {
    "s": 1.0 / 3600,
    "m": 1.0 / 60,
    "h": 1.0,
    "d": 24.0,
    "w": 168.0,      # 7 * 24
    "M": 720.0,       # 30 * 24 (approximate)
}


def parse_timeframe_hours(timeframe: str) -> float:
    """Convert a timeframe string to its duration in hours.

    Args:
        timeframe: Candle interval string (e.g. ``'4h'``, ``'1d'``).

    Returns:
        Duration in hours as a float (e.g. ``4.0``, ``24.0``).

    Raises:
        ValueError: If the string does not match the expected pattern.
    """
    match = _TF_PATTERN.match(timeframe)
    if not match:
        raise ValueError(
            f"Cannot parse timeframe '{timeframe}'. "
            f"Expected format like '1h', '4h', '1d', etc."
        )
    quantity = int(match.group(1))
    unit = match.group(2)
    return quantity * _UNIT_TO_HOURS[unit]


def validate_timeframe(
    timeframe: str,
    exchange: Optional[object] = None,
) -> None:
    """Validate *timeframe* against the live exchange's supported intervals.

    This function **never** falls back silently — it raises on any
    invalid input so that misconfigured timeframes are caught at
    config-load time, not deep inside a training loop.

    Args:
        timeframe: Candle interval string to validate (e.g. ``'4h'``).
        exchange: An already-instantiated ``ccxt`` exchange object.
            When ``None``, a temporary instance is created based on the
            currently configured ``active_market``.

    Raises:
        ValueError: If the timeframe is not supported by the exchange.
    """
    # Always validate the string is parseable first.
    parse_timeframe_hours(timeframe)

    owns_exchange = exchange is None
    if owns_exchange:
        active_market = get_active_market()
        if active_market == "futures":
            exchange = ccxt.binanceusdm({"enableRateLimit": True})
        else:
            exchange = ccxt.binance({"enableRateLimit": True})

    # ccxt exchanges expose .timeframes as a dict[str, str] after
    # instantiation (no load_markets() call needed for the timeframes
    # dict — it's statically defined on the exchange class).
    supported = getattr(exchange, "timeframes", None) or {}

    if timeframe not in supported:
        sorted_tfs = sorted(supported.keys(), key=lambda t: parse_timeframe_hours(t))
        raise ValueError(
            f"Timeframe '{timeframe}' is not supported by the exchange. "
            f"Supported timeframes: {sorted_tfs}"
        )
