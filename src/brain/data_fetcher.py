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

from src.config.settings_loader import (
    get_active_market,
    get_active_symbols,
    get_project_root,
)

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

    # 1. Initialize exchange and normalize symbol
    ccxt_symbol = symbol.replace("_", "/")
    if active_market == "futures":
        logger.info(
            "Downloading %s with timeframe %s (Futures USD-M)...", symbol, timeframe
        )
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        if ":" not in ccxt_symbol:
            ccxt_symbol = f"{ccxt_symbol}:USDT"
    else:
        logger.info(
            "Downloading %s with timeframe %s (Spot)...", symbol, timeframe
        )
        exchange = ccxt.binance({"enableRateLimit": True})

    since_milliseconds: int = exchange.parse8601(start_date)
    all_candles: list[list] = []

    # 2. Loop to download in batches of 1000
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
                    symbol, attempt, _MAX_RETRIES, exc, delay,
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

    # 3. Convert to DataFrame
    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # 4. Save to data/raw_csv/
    base_dir = get_project_root()
    folder_path = base_dir / "data" / "raw_csv"
    folder_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = (
        symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    )
    file_name = f"{safe_symbol}_{timeframe}.csv"
    full_path = folder_path / file_name

    df.to_csv(full_path)

    logger.info(
        "Saved %d days of history to: %s", len(df), full_path
    )
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
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except requests.Timeout as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Fear & Greed timeout (attempt %d/%d): %s. Retrying in %.0fs...",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except requests.HTTPError as exc:
            logger.error("Fear & Greed HTTP error: %s", exc)
            return pd.DataFrame()
        except (ValueError, KeyError) as exc:
            logger.error("Error parsing Fear & Greed response: %s", exc)
            return pd.DataFrame()

    logger.error(
        "Could not download Fear & Greed after %d attempts.", _MAX_RETRIES
    )
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
                        symbol, _MAX_RETRIES, exc,
                    )
                    return None
                logger.warning(
                    "Network error for %s (attempt %d/%d): %s",
                    symbol, attempt, _MAX_RETRIES, exc,
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
    args = parser.parse_args()

    if args.symbol:
        fetch_historical_data(
            symbol=args.symbol, timeframe="1d", start_date="2020-01-01T00:00:00Z"
        )
    else:
        symbols = get_active_symbols()
        if not symbols:
            logger.warning("No symbols found in settings.yaml.")
        else:
            for s in symbols:
                fetch_historical_data(
                    symbol=s, timeframe="1d", start_date="2020-01-01T00:00:00Z"
                )
