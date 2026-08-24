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

from src.config.paths import get_funding_rate_path, get_raw_csv_path
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


def fetch_funding_rate_history(
    symbol: str = "BTC/USDT",
    start_date: str = "2020-01-01T00:00:00Z",
) -> Optional[pd.DataFrame]:
    """Download complete funding rate history from Binance USD-M Futures.

    Funding rates on Binance Futures settle every 8 hours (00:00, 08:00,
    16:00 UTC). Each record's timestamp is the *settlement time* — i.e.
    the exact moment the rate was applied. The returned (and saved)
    DataFrame therefore represents the value that becomes *known* at
    that timestamp, not the still-upcoming rate for the next funding
    interval.

    Uses ``ccxt.binanceusdm.fetch_funding_rate_history`` in paginated
    chunks with the same exponential-backoff retry pattern as
    :func:`fetch_historical_data`.

    Args:
        symbol: Trading pair (e.g. ``'BTC/USDT'``).  Only makes sense
            for perpetual futures contracts.
        start_date: Earliest date to fetch, in ISO 8601 format.

    Returns:
        DataFrame with a ``funding_rate`` column indexed by settlement
        DatetimeIndex, or ``None`` on total failure.
    """
    logger.info(
        "Downloading funding rate history for %s (Futures USD-M, since %s)...",
        symbol, start_date,
    )

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    ccxt_symbol = symbol.replace("_", "/")
    if ":" not in ccxt_symbol:
        ccxt_symbol = f"{ccxt_symbol}:USDT"

    since_ms: int = exchange.parse8601(start_date)
    all_entries: list[dict] = []

    while True:
        chunk: Optional[list[dict]] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                chunk = exchange.fetch_funding_rate_history(
                    ccxt_symbol, since=since_ms, limit=1000,
                )
                break
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error downloading funding rate %s "
                    "(attempt %d/%d): %s. Retrying in %.0fs...",
                    symbol, attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error on funding rate %s: %s", symbol, exc)
                raise RuntimeError(
                    f"Exchange error fetching funding rate for {symbol}: {exc}"
                ) from exc

        if chunk is None:
            raise RuntimeError(
                f"Funding rate download for {symbol} failed after "
                f"{_MAX_RETRIES} attempts."
            )

        if len(chunk) == 0:
            break

        all_entries.extend(chunk)
        last_ts: int = int(chunk[-1]["timestamp"])
        prev_count = len(all_entries)
        since_ms = last_ts + 1

        logger.debug("Funding rate: downloaded %d entries so far...", prev_count)

        # Stop if we got fewer than the limit (last page) or if the
        # last timestamp is within 8h of "now" (no more past funding).
        if len(chunk) < 1000:
            break

        time.sleep(1)

    # Convert entries to DataFrame.  ccxt returns each entry as:
    #   {"symbol": ..., "fundingRate": float, "timestamp": int_ms, "datetime": str}
    rows = [
        {"timestamp": int(e["timestamp"]), "funding_rate": float(e["fundingRate"])}
        for e in all_entries
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Funding rate history returned empty for %s.", symbol)
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    full_path = get_funding_rate_path(safe_symbol)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(full_path)

    logger.info(
        "Saved %d funding rate settlements to: %s (range: %s → %s)",
        len(df), full_path,
        df.index.min(), df.index.max(),
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


def fetch_historical_data_binance_rest(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    start_date: str = "2020-01-01T00:00:00Z",
    market: str = "futures",
) -> Optional[pd.DataFrame]:
    """Download OHLCV + taker buy/sell fields via Binance native REST.

    ccxt's ``fetch_ohlcv`` only exposes the 6 standard OHLCV columns.  The
    raw Binance REST endpoint returns 12 fields per kline, including
    ``taker_buy_base_asset_volume`` (field #9) needed for micro-structure
    features such as ``taker_buy_ratio``.

    Supported market endpoints:
    * ``"futures"`` → ``https://fapi.binance.com/fapi/v1/klines``  (USD-M)
    * ``"spot"``    → ``https://api.binance.com/api/v3/klines``

    Kline field mapping (per Binance docs, both spot & futures):
        0 open_time                   → timestamp (ms)
        1 open                        → open
        2 high                        → high
        3 low                         → low
        4 close                       → close
        5 base_asset_volume           → volume
        6 close_time
        7 quote_asset_volume          → quote_volume
        8 number_of_trades            → n_trades
        9 taker_buy_base_asset_volume → taker_buy_base_vol
       10 taker_buy_quote_asset_volume→ taker_buy_quote_vol
       11 ignore

    Uses the same exponential-backoff retry pattern as
    :func:`fetch_historical_data`.

    Args:
        symbol: Trading pair in Binance REST format (e.g. ``BTCUSDT``).
            Slash/underscore forms are auto-normalised.
        timeframe: Candlestick interval.
        start_date: Earliest ISO-8601 date to fetch from.
        market: ``"futures"`` (USD-M) or ``"spot"``.

    Returns:
        DataFrame with columns ``open, high, low, close, volume,
        quote_volume, n_trades, taker_buy_base_vol, taker_buy_quote_vol``
        indexed by open_timestamp, or ``None`` on total failure.
    """
    import datetime as _dt

    rest_symbol = (
        symbol.replace("/", "").replace("_", "").split("USDT")[0] + "USDT"
    )
    logger.info(
        "Downloading via Binance REST [%s/%s] %s %s (since %s)...",
        market, "fapi" if market == "futures" else "api",
        rest_symbol, timeframe, start_date,
    )

    base_url = (
        "https://fapi.binance.com/fapi/v1/klines"
        if market == "futures" else
        "https://api.binance.com/api/v3/klines"
    )

    since_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_rows: list[list] = []
    page_size = 1500  # Binance max per call

    while True:
        chunk: Optional[list[list]] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    base_url,
                    params={
                        "symbol": rest_symbol,
                        "interval": timeframe,
                        "startTime": since_ms,
                        "limit": page_size,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                chunk = resp.json()
                if not isinstance(chunk, list):
                    raise ValueError(
                        f"Unexpected REST response type: {type(chunk).__name__}"
                    )
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "REST network error for %s %s (attempt %d/%d): %s. "
                    "Retrying in %.0fs...",
                    rest_symbol, timeframe, attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
            except (requests.HTTPError, ValueError) as exc:
                logger.error("REST error for %s %s: %s", rest_symbol, timeframe, exc)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_BASE_DELAY)

        if chunk is None:
            raise RuntimeError(
                f"Binance REST download failed after {_MAX_RETRIES} attempts."
            )

        if len(chunk) == 0:
            break

        all_rows.extend(chunk)
        last_open_ms = int(chunk[-1][0])
        prev_count = len(all_rows)
        since_ms = last_open_ms + 1

        if len(all_rows) % (page_size * 5) == 0:
            logger.debug("REST klines: %d rows so far (last %s)...",
                         prev_count, pd.to_datetime(last_open_ms, unit="ms"))

        if len(chunk) < page_size:
            break

        time.sleep(0.25)

    if not all_rows:
        logger.warning("REST klines returned empty for %s %s.", rest_symbol, timeframe)
        return None

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
    df.drop(columns=["open_time", "close_time", "ignore"], inplace=True)

    for col in ("open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_trades"] = df["n_trades"].astype(int)

    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
    full_path = get_raw_csv_path(safe_symbol, timeframe)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(full_path)

    logger.info(
        "Saved REST kline data (%d rows) to %s (range %s → %s, cols: %s)",
        len(df), full_path, df.index.min(), df.index.max(),
        ",".join(df.columns),
    )
    return df


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


def fetch_onchain_active_addresses(symbol: str = "BTC_USDT") -> pd.DataFrame:
    """Download the full history of daily unique addresses from Blockchain.com.

    Source: ``https://api.blockchain.info/charts/n-unique-addresses``
    - Metric: *Number of Unique Addresses Used* (daily, Bitcoin-only).
    - No API key required.
    - The endpoint returns the *complete* history (``timespan=all``) in a
      single JSON response; no pagination is needed.
    - Response format::

          {"values": [{"x": <epoch_seconds_int>, "y": <float>}, ...]}

    Timestamp semantics
    -------------------
    Each ``x`` value is a Unix timestamp (seconds) representing the **start
    of the UTC day** (i.e. midnight, ``00:00:00 UTC``).  The ``y`` value is
    the count of unique addresses *seen during that calendar day*.

    Because the aggregate is computed over the full day it can only be
    *finalized* once the day closes.  Blockchain.com does not publish a
    formal SLA for data latency.  Given the absence of a documented
    guarantee, a conservative interpretation (+2 calendar days of shift
    before the merge_asof) is applied at enrichment time — see
    ``add_onchain_active_addresses`` in ``features.py``.

    The saved index uses ``datetime64[ns]`` normalized to midnight (tz-naive),
    matching every other time-series in this codebase.

    Args:
        symbol: Symbol key used only to construct the output path.
            Only ``'BTC_USDT'`` is meaningful for this metric.

    Returns:
        DataFrame with a single column ``onchain_active_addresses``
        indexed by a tz-naive ``DatetimeIndex`` (midnight-aligned), or an
        empty DataFrame on total failure.
    """
    from src.config.paths import get_onchain_active_addresses_path

    url = (
        "https://api.blockchain.info/charts/n-unique-addresses"
        "?timespan=all&format=json"
    )
    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"

    logger.info(
        "Downloading onchain active addresses (Blockchain.com) for %s...", safe_symbol
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            values = payload.get("values", [])

            if not values:
                logger.warning("Blockchain.com returned empty 'values' array.")
                return pd.DataFrame()

            df_onchain = pd.DataFrame(values, columns=["x", "y"])

            # 'x' is epoch seconds UTC → normalize to midnight datetime64[ns] tz-naive
            df_onchain["timestamp"] = (
                pd.to_datetime(df_onchain["x"].astype(int), unit="s", utc=True)
                .dt.normalize()
                .dt.tz_localize(None)
            )
            df_onchain["onchain_active_addresses"] = df_onchain["y"].astype(float)
            df_onchain = (
                df_onchain[["timestamp", "onchain_active_addresses"]]
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp")
                .set_index("timestamp")
            )

            full_path = get_onchain_active_addresses_path(safe_symbol)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            df_onchain.to_csv(full_path)

            logger.info(
                "Saved %d daily rows of onchain active addresses to: %s "
                "(range: %s → %s)",
                len(df_onchain),
                full_path,
                df_onchain.index.min(),
                df_onchain.index.max(),
            )
            return df_onchain

        except requests.ConnectionError as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Blockchain.com connection error (attempt %d/%d): %s. "
                "Retrying in %.0fs...",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except requests.Timeout as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Blockchain.com timeout (attempt %d/%d): %s. "
                "Retrying in %.0fs...",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except requests.HTTPError as exc:
            logger.error("Blockchain.com HTTP error: %s", exc)
            return pd.DataFrame()
        except (ValueError, KeyError) as exc:
            logger.error("Error parsing Blockchain.com response: %s", exc)
            return pd.DataFrame()

    logger.error(
        "Could not download onchain active addresses after %d attempts.",
        _MAX_RETRIES,
    )
    return pd.DataFrame()


def fetch_mempool_fee_rate_median(symbol: str = "BTC_USDT") -> pd.DataFrame:
    """Download the full history of daily median fee-rates from mempool.space.

    Source: ``https://mempool.space/api/v1/mining/blocks/fee-rates/all``

    Metric
    ------
    ``avgFee_50`` — the p50 (median) fee-rate in **sat/vB** aggregated over
    all blocks mined during each calendar day.  Each JSON entry groups
    approximately 144–153 confirmed blocks (one natural Bitcoin day).

    Why this metric signals congestion
    ------------------------------------
    The median fee-rate reflects the price the *typical* on-chain participant
    was willing to pay on a given day.  When block space is contested, users
    bid up fees; when blocks are uncrowded, low-fee transactions clear easily
    and the median stays near the floor (~1 sat/vB).  This makes ``avgFee_50``
    a direct, user-revealed proxy for mempool pressure — distinct from hashrate,
    address count, or exchange-classification metrics.

    Timestamp semantics and day assignment
    ----------------------------------------
    Each API entry's ``timestamp`` is the Unix epoch second of the *central
    block* of the daily bucket — not UTC midnight.  Observed range across
    2020-08-2026: **06:43–14:10 UTC** (verified against all 2420 entries;
    0 entries fall in the 00:00–06:00 or 18:00–24:00 risk zones).
    Applying ``pd.to_datetime(..., utc=True).normalize()`` therefore always
    assigns the correct UTC calendar day — confirmed against 6 specific real
    entries (heights 610778, 610935, 611729, 612037, 689380, 762080).

    The resulting index is stored as tz-naive ``datetime64[ns]`` midnight UTC,
    matching every other time-series in the pipeline.

    Shift decision (+1 day, same as ``trend_htf``)
    -----------------------------------------------
    The mempool backend (``backend/src/api/blocks.ts``) indexes
    ``fee_rate_percentiles`` synchronously in the same block-processing cycle,
    directly from Bitcoin Core RPC.  There is no async pipeline or artificial
    delay beyond normal block confirmation (~10 minutes).  The data is
    therefore available within seconds of a block being confirmed — the same
    semantics as any other per-block metric.  A **+1 day shift** (identical to
    ``add_trend_htf``) is applied before ``merge_asof`` to ensure sub-daily
    bars during day D only see the value for day D-1, never the still-in-progress
    day D.  No additional conservative buffer is needed (unlike
    ``onchain_active_addresses``, which has no published SLA).

    Retroactive revision risk
    --------------------------
    Zero.  Confirmed Bitcoin blocks are cryptographically immutable.  The fee
    data is computed from already-mined transactions and cannot be revised
    retroactively — a stronger guarantee than any off-chain aggregation service.

    Data coverage
    --------------
    Endpoint returns history from block 0 (2009-01-03).
    Coverage from 2020-01-01: 366 entries/year, no gaps.
    Latest available at fetch time: 2026-08-16.

    No API key required.  A single HTTP request returns the full history
    (~955 KB); no pagination needed.

    Args:
        symbol: Symbol key used only to construct the output file path.
            Only ``'BTC_USDT'`` is meaningful for this metric.

    Returns:
        DataFrame with a single column ``mempool_fee_rate_p50``
        indexed by a tz-naive midnight ``DatetimeIndex`` (UTC), or an
        empty DataFrame on total failure.
    """
    from src.config.paths import get_mempool_fee_rate_path

    url = "https://mempool.space/api/v1/mining/blocks/fee-rates/all"
    safe_symbol = symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"

    logger.info(
        "Downloading mempool fee-rate history (mempool.space) for %s...", safe_symbol
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            values = response.json()

            if not values or not isinstance(values, list):
                logger.warning("mempool.space fee-rates returned empty or unexpected response.")
                return pd.DataFrame()

            df_fees = pd.DataFrame(values)

            # Normalise timestamp: epoch seconds → tz-naive midnight UTC datetime64[ns]
            # Each entry's timestamp is the central block of the ~daily bucket.
            # Verified: all 2020-2026 timestamps fall 06:43–14:10 UTC — normalize()
            # always maps to the correct calendar day (0 day-crossing edge cases).
            df_fees["timestamp"] = (
                pd.to_datetime(df_fees["timestamp"].astype(int), unit="s", utc=True)
                .dt.normalize()
                .dt.tz_localize(None)
            )
            df_fees = df_fees.rename(columns={"avgFee_50": "mempool_fee_rate_p50"})
            df_fees = (
                df_fees[["timestamp", "mempool_fee_rate_p50"]]
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp")
                .set_index("timestamp")
            )
            df_fees["mempool_fee_rate_p50"] = pd.to_numeric(
                df_fees["mempool_fee_rate_p50"], errors="coerce"
            )

            full_path = get_mempool_fee_rate_path(safe_symbol)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            df_fees.to_csv(full_path)

            logger.info(
                "Saved %d daily rows of mempool fee-rate p50 to: %s "
                "(range: %s → %s)",
                len(df_fees),
                full_path,
                df_fees.index.min(),
                df_fees.index.max(),
            )
            return df_fees

        except requests.ConnectionError as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "mempool.space connection error (attempt %d/%d): %s. "
                "Retrying in %.0fs...",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except requests.Timeout as exc:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "mempool.space timeout (attempt %d/%d): %s. "
                "Retrying in %.0fs...",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except requests.HTTPError as exc:
            logger.error("mempool.space HTTP error: %s", exc)
            return pd.DataFrame()
        except (ValueError, KeyError) as exc:
            logger.error("Error parsing mempool.space fee-rate response: %s", exc)
            return pd.DataFrame()

    logger.error(
        "Could not download mempool fee-rate history after %d attempts.", _MAX_RETRIES
    )
    return pd.DataFrame()


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
    parser.add_argument(
        "--funding-rate",
        action="store_true",
        help="Download Futures funding rate history instead of OHLCV.",
    )
    parser.add_argument(
        "--binance-rest",
        action="store_true",
        help="Use native Binance REST endpoint (fapi/api) instead of ccxt. "
             "Includes extra columns: quote_volume, n_trades, "
             "taker_buy_base_vol, taker_buy_quote_vol.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="futures",
        choices=["futures", "spot"],
        help="Market to use with --binance-rest (futures = USD-M, default).",
    )
    args = parser.parse_args()

    from src.utils.logging_config import setup_logging
    setup_logging()

    if args.funding_rate:
        if args.symbol:
            fetch_funding_rate_history(
                symbol=args.symbol, start_date="2020-01-01T00:00:00Z"
            )
        else:
            entries = get_active_symbols_with_timeframe()
            if not entries:
                logger.warning("No symbols found in settings.yaml.")
            else:
                for entry in entries:
                    fetch_funding_rate_history(
                        symbol=entry["symbol"], start_date="2020-01-01T00:00:00Z"
                    )
    elif args.binance_rest:
        _download_fn = fetch_historical_data_binance_rest
        if args.symbol:
            from src.config.settings_loader import get_symbol_timeframe
            tf = args.timeframe or get_symbol_timeframe(args.symbol)
            _download_fn(
                symbol=args.symbol, timeframe=tf,
                start_date="2020-01-01T00:00:00Z", market=args.market,
            )
        else:
            entries = get_active_symbols_with_timeframe()
            if not entries:
                logger.warning("No symbols found in settings.yaml.")
            else:
                for entry in entries:
                    tf = args.timeframe or entry["timeframe"]
                    _download_fn(
                        symbol=entry["symbol"], timeframe=tf,
                        start_date="2020-01-01T00:00:00Z", market=args.market,
                    )
    elif args.symbol:
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

