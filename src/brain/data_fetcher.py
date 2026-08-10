"""data_fetcher.py — Historical and real-time data download from Binance.

Also includes Fear & Greed index retrieval.
"""

import argparse
import logging
import time
from typing import Optional

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import requests

from src.config.paths import get_raw_csv_path
from src.config.settings_loader import (
    get_active_market,
    get_active_symbols_with_timeframe,
    get_project_root,
)
from src.utils.timeframe_utils import validate_timeframe

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 2.0  # seconds


def fetch_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    start_date: str = "2020-01-01T00:00:00Z",
) -> Optional[pd.DataFrame]:
    """Download complete historical data from Binance (Futures or Spot).

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'``).
        timeframe: Candlestick timeframe (e.g. ``'1d'``).
        start_date: Start date in ISO 8601 format.

    Returns:
        DataFrame with OHLCV columns or ``None`` on failure.
    """
    active_market = get_active_market()

    ccxt_symbol = symbol.replace("_", "/")
    if active_market == "futures":
        logger.info(
            "Downloading %s with timeframe %s (Futures USD-M)...", symbol, timeframe
        )
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        if ":" not in ccxt_symbol:
            ccxt_symbol = f"{ccxt_symbol}:USDT"
    else:
        logger.info("Downloading %s with timeframe %s (Spot)...", symbol, timeframe)
        exchange = ccxt.binance({"enableRateLimit": True})

    validate_timeframe(timeframe, exchange=exchange)

    since_milliseconds: int = exchange.parse8601(start_date)
    all_candles: list[list] = []

    while True:
        candles: Optional[list[list]] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                candles = exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe, since=since_milliseconds, limit=1000
                )
                break
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error downloading %s (attempt %d/%d): %s. Retrying in %.0fs...",
                    symbol,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error downloading %s: %s", symbol, exc)
                raise RuntimeError(
                    f"Exchange error downloading {symbol}: {exc}"
                ) from exc

        if candles is None:
            raise RuntimeError(
                f"Download of {symbol} failed after {_MAX_RETRIES} attempts."
            )

        if len(candles) == 0:
            break

        all_candles.extend(candles)
        last_time: int = candles[-1][0]
        since_milliseconds = last_time + 1

        logger.debug("Downloaded %d candles so far...", len(all_candles))

        if len(candles) < 1000:
            break

        time.sleep(1)

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    full_path = get_raw_csv_path(safe_symbol, timeframe)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(full_path)

    logger.info("Saved %d bars of history to: %s", len(df), full_path)
    return df


def get_fear_and_greed() -> pd.DataFrame:
    """Download the Fear & Greed index from alternative.me.

    Returns:
        DataFrame with ``fng_value`` column indexed by timestamp,
        or empty DataFrame on failure.
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])

            if not data:
                logger.warning("Empty response from Fear & Greed API.")
                return pd.DataFrame()

            df_fg = pd.DataFrame(data)
            df_fg["timestamp"] = pd.to_datetime(
                df_fg["timestamp"].astype(int), unit="s"
            ).dt.normalize()
            df_fg["fng_value"] = df_fg["value"].astype(float)
            df_fg = df_fg[["timestamp", "fng_value"]].set_index("timestamp")
            df_fg.sort_index(inplace=True)
            df_fg.columns = df_fg.columns.str.lower()
            return df_fg

        except requests.ConnectionError as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Fear & Greed connection error (attempt %d/%d): %s. Retrying in %.0fs...",
                attempt,
                _MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
        except requests.Timeout as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Fear & Greed timeout (attempt %d/%d): %s. Retrying in %.0fs...",
                attempt,
                _MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
        except requests.HTTPError as exc:
            logger.error("Fear & Greed HTTP error: %s", exc)
            return pd.DataFrame()
        except (ValueError, KeyError) as exc:
            logger.error("Error parsing Fear & Greed response: %s", exc)
            return pd.DataFrame()

    logger.error("Could not download Fear & Greed after %d attempts.", _MAX_RETRIES)
    return pd.DataFrame()


async def fetch_ohlcv_binance(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 100,
    exchange: Optional[ccxt_async.binanceusdm] = None,
) -> Optional[pd.DataFrame]:
    """Download the latest candles from Binance USD-M Futures (async, no credentials).

    Args:
        symbol: Trading pair (e.g. ``'BTC_USDT'``).
        timeframe: Timeframe (e.g. ``'1d'``).
        limit: Maximum number of candles to download.
        exchange: Optional shared ``ccxt_async.binanceusdm`` instance.
            When provided the caller is responsible for closing it.
            When ``None`` a temporary instance is created and closed
            automatically.

    Returns:
        DataFrame with OHLCV columns or ``None`` on failure.
    """
    ccxt_symbol = symbol.replace("_", "/")
    if ":" not in ccxt_symbol:
        ccxt_symbol = f"{ccxt_symbol}:USDT"

    _owns_exchange = exchange is None
    if _owns_exchange:
        exchange = ccxt_async.binanceusdm({"enableRateLimit": True})
    try:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                candles = await exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe, limit=limit
                )
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.columns = df.columns.str.lower()
                return df
            except ccxt.NetworkError as exc:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "Error downloading data for %s after %d attempts: %s",
                        symbol,
                        _MAX_RETRIES,
                        exc,
                    )
                    return None
                logger.warning(
                    "Network error for %s (attempt %d/%d): %s",
                    symbol,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error for %s: %s", symbol, exc)
                return None
    finally:
        if _owns_exchange:
            await exchange.close()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical data downloader.")
    parser.add_argument(
        "symbol",
        type=str,
        nargs="?",
        help="Symbol (e.g.: ETH_USDT). If not provided, uses settings.yaml.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe (e.g.: 4h, 1d). Uses per-symbol config if not provided.",
    )
    args = parser.parse_args()
    
    from src.utils.logging_config import setup_logging
    setup_logging()

    if args.symbol:
        from src.config.settings_loader import get_symbol_timeframe

        tf = args.timeframe or get_symbol_timeframe(args.symbol)
        fetch_historical_data(
            symbol=args.symbol, timeframe=tf, start_date="2020-01-01T00:00:00Z"
        )
    else:
        entries = get_active_symbols_with_timeframe()
        if not entries:
            logger.warning("No symbols found in settings.yaml.")
        else:
            for entry in entries:
                tf = args.timeframe or entry["timeframe"]
                fetch_historical_data(
                    symbol=entry["symbol"], timeframe=tf, start_date="2020-01-01T00:00:00Z"
                )
