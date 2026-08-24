"""features.py — Technical indicator computation on OHLCV data.

Groups functions for momentum, trend, volatility, and volume
using the ``pandas_ta`` library.
"""

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum indicators: RSI, MACD, Stochastic.

    Args:
        df: DataFrame with ``close``, ``high``, ``low`` columns.

    Returns:
        DataFrame with momentum columns added.
    """
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    try:
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
        else:
            df["stoch_k"] = 0
    except (KeyError, ValueError) as exc:
        logger.debug("Stochastic not available: %s", exc)
        df["stoch_k"] = 0

    return df


def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend indicators: EMA 50, distance to EMA, ADX.

    Args:
        df: DataFrame with ``close``, ``high``, ``low`` columns.

    Returns:
        DataFrame with trend columns added.
    """
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]

    adx = ta.adx(df["high"], df["low"], df["close"])
    if adx is not None:
        df["adx_14"] = adx["ADX_14"]
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility indicators: ATR, Bollinger Bands.

    Args:
        df: DataFrame with ``close``, ``high``, ``low`` columns.

    Returns:
        DataFrame with volatility columns added.
    """
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"])

    try:
        bb = ta.bbands(df["close"], length=20)
        if bb is not None:
            col_bbu = [c for c in bb.columns if "BBU" in c][0]
            col_bbb = [c for c in bb.columns if "BBB" in c][0]
            df["bb_width"] = bb[col_bbb]
            df["bb_pos"] = df["close"] / bb[col_bbu]
        else:
            df["bb_width"] = 0
            df["bb_pos"] = 0
    except (KeyError, IndexError, ValueError) as exc:
        logger.debug("Bollinger Bands not available: %s", exc)
        df["bb_width"] = 0
        df["bb_pos"] = 0

    return df


def compute_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume indicators: OBV, relative volume.

    Args:
        df: DataFrame with ``close``, ``volume`` columns.

    Returns:
        DataFrame with volume columns added.
    """
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["rel_volume"] = df["volume"] / df["vol_sma_20"]
    return df


def compute_all_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on the DataFrame.

    Args:
        df: DataFrame with OHLCV columns (lowercase).

    Returns:
        DataFrame with all technical indicators added.
    """
    compute_momentum(df)
    compute_trend(df)
    compute_volatility(df)
    compute_volume(df)
    return df


def add_sentiment(df: pd.DataFrame, df_fg: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Attempt to add sentiment columns to the DataFrame.

    Uses an as-of (backward) merge rather than an exact-index join so that
    sub-daily candles (``4h``, ``1h``, etc.) correctly inherit the most
    recent daily Fear & Greed value regardless of whether the candle grid
    happens to align with midnight UTC. An exact-index join would leave
    every non-aligned candle's sentiment columns as NaN — which, combined
    with ``rolling(14)`` needing 14 non-NaN values in-window, previously
    produced all-NaN ``fng_sma_14``/``fng_vol_14`` columns for any sub-daily
    timeframe and caused ``cleanup_columns``'s ``dropna()`` to silently
    remove every row.

    Args:
        df: DataFrame with OHLCV data and technical indicators.
        df_fg: DataFrame containing the Fear and Greed index data.

    Returns:
        Tuple ``(df, has_sentiment)`` indicating whether sentiment was added.
    """
    if not df_fg.empty:
        df_fg = df_fg[~df_fg.index.duplicated(keep="last")].sort_index()
        df_fg_computed = df_fg.copy()
        df_fg_computed["fng_sma_14"] = df_fg_computed["fng_value"].rolling(14).mean()
        df_fg_computed["fng_vol_14"] = df_fg_computed["fng_value"].rolling(14).std()

        df = df.sort_index()
        df.index = df.index.astype("datetime64[ns]")
        df_fg_computed.index = df_fg_computed.index.astype("datetime64[ns]")

        df = pd.merge_asof(
            df,
            df_fg_computed[["fng_value", "fng_sma_14", "fng_vol_14"]],
            left_index=True,
            right_index=True,
            direction="backward",
        )
        return df, True

    logger.warning("Sentiment data is empty. The arena will be technical-only.")
    return df, False


def add_trend_htf(df: pd.DataFrame, df_1d: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Add higher-timeframe trend context using daily EMA200 distance.

    Computes ``trend_htf = (close_1d - EMA200_1d) / EMA200_1d`` on the daily
    bars, then as-of-merges (backward) the value onto each sub-daily candle so
    that a 4h or 1h bar only inherits a **fully closed** daily bar — never the
    daily bar still in progress (which would leak same-day future information).

    Follows the exact ``merge_asof(direction="backward")`` pattern used by
    :func:`add_sentiment` for the Fear & Greed index, including the mandatory
    ``datetime64[ns]`` normalisation on both indexes before the join.

    Args:
        df: Sub-daily DataFrame (e.g. 4h, 1h) with OHLCV data and
            technical indicators.  Index must be a ``DatetimeIndex``.
        df_1d: Daily OHLCV DataFrame for the same symbol.  Index must be
            a ``DatetimeIndex`` (normally midnight-aligned).

    Returns:
        Tuple ``(df, has_trend_htf)`` indicating whether the feature was added.
    """
    if df_1d.empty:
        logger.warning("Daily (1d) data is empty. trend_htf will be skipped.")
        return df, False

    required_cols = {"close"}
    missing = required_cols - set(df_1d.columns)
    if missing:
        logger.warning(
            "Daily data missing required columns %s for trend_htf — skipping.",
            missing,
        )
        return df, False

    df_1d = df_1d[~df_1d.index.duplicated(keep="last")].sort_index()
    df_1d_computed = df_1d.copy()

    df_1d_computed["ema_200_1d"] = ta.ema(df_1d_computed["close"], length=200)
    df_1d_computed["trend_htf"] = (
        df_1d_computed["close"] - df_1d_computed["ema_200_1d"]
    ) / df_1d_computed["ema_200_1d"]

    # CRITICAL LEAKAGE PREVENTION:
    # A daily bar's timestamp (e.g. 2024-01-02 00:00) is its **OPEN** time.
    # The bar only closes 24 h later (at 2024-01-03 00:00) and its close/EMA
    # should NOT be visible to sub-daily bars during the day itself.
    # Shift the daily index forward by 1 calendar day so a bar's trend_htf
    # value becomes "available" for merging only at the bar's close time.
    # Then merge_asof(backward) will correctly pick the most recent *closed*
    # daily bar for every sub-daily timestamp.
    df_1d_computed.index = df_1d_computed.index + pd.Timedelta(days=1)

    df = df.sort_index()
    df.index = df.index.astype("datetime64[ns]")
    df_1d_computed.index = df_1d_computed.index.astype("datetime64[ns]")

    df = pd.merge_asof(
        df,
        df_1d_computed[["trend_htf"]],
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return df, True


def add_taker_buy_ratio(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Compute ``taker_buy_ratio`` = taker_buy_base_vol / volume per candle.

    This is a native bar-level micro-structure feature: it quantifies the
    fraction of total volume that was executed as *aggressive buy* (taker
    lifting the ask).  Values live in ``[0, 1]``; ``0.5`` is neutral
    equilibrium, values ``> 0.5`` signal buy-imbalance, ``< 0.5`` signal
    sell-imbalance.

    Temporal semantics — **no leakage possible by construction**:
    ``taker_buy_base_vol`` is an aggregate field of the candle itself,
    exactly like ``close`` / ``high`` / ``low`` / ``volume``.  Binance
    populates it at candle *close* time, and the value is stored under
    the candle's open-time index alongside the other OHLCV fields.  A
    model that consumes feature row ``i`` (index = bar open) therefore
    sees only taker data from bars whose close time has already passed —
    identical to the leakage model of every other technical indicator.

    Requires the input DataFrame to have been loaded from the CSV
    produced by :func:`fetch_historical_data_binance_rest`, which
    contains columns ``volume`` and ``taker_buy_base_vol``.

    Args:
        df: OHLCV DataFrame indexed by candle open-time, with at least
            columns ``volume`` and ``taker_buy_base_vol``.

    Returns:
        Tuple ``(df, has_taker_ratio)``.  On success ``df`` has a new
        column ``taker_buy_ratio``.  On failure the original df is
        returned and the bool is ``False``.
    """
    required = {"volume", "taker_buy_base_vol"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            "add_taker_buy_ratio skipping: missing columns %s "
            "(available: %s). Re-download via data_fetcher --binance-rest "
            "to obtain taker fields.", missing, list(df.columns),
        )
        return df, False

    df = df.copy()
    total_vol = df["volume"].astype(float)
    taker_buy = df["taker_buy_base_vol"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_vol > 0.0, taker_buy / total_vol, np.nan)
    df["taker_buy_ratio"] = ratio

    zero_vol_rows = int((total_vol <= 0.0).sum())
    if zero_vol_rows:
        logger.debug(
            "taker_buy_ratio: %d rows had volume=0 → set to NaN", zero_vol_rows,
        )

    ok = bool(np.isfinite(df["taker_buy_ratio"]).any())
    if not ok:
        logger.warning("taker_buy_ratio: no finite values after compute.")
    return df, ok


def add_funding_rate(df: pd.DataFrame, df_funding: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Add Futures funding rate (`funding_rate_current`) onto sub-daily candles.

    Uses an as-of (backward) merge so every sub-daily bar inherits the most
    recent **already-settled** funding rate known at the bar's open time.

    Leakage model — Binance USD-M Futures funding settles every 8 hours at
    00:00, 08:00, 16:00 UTC.  The CSV loaded from
    ``data/raw_csv/{symbol}/funding_rate.csv`` stores each settlement with
    its index set to the *settlement timestamp* — i.e. the exact moment the
    rate becomes publicly known and applied.  Therefore, unlike
    :func:`add_trend_htf`, **no index shift is needed** — the CSV index is
    already the "availability" timestamp.  ``merge_asof(direction="backward")``
    then correctly matches:

    * A 04:00 UTC 4h bar → sees the 00:00 funding (the 08:00 funding has NOT
      yet settled when the 04:00 bar opens → no leakage).
    * A 08:00 UTC 4h bar → sees the 08:00 funding (settlement just happened,
      value is available at bar open → correct, no future peek).
    * A 12:00 UTC 4h bar → sees the 08:00 funding (16:00 not yet settled).

    This is the exact same merge_asof(direction="backward") pattern used in
    :func:`add_sentiment` and :func:`add_trend_htf`, with the mandatory
    ``datetime64[ns]`` normalisation.

    Args:
        df: Sub-daily DataFrame (e.g. 4h, 1h) with OHLCV / technicals.
            Index must be a DatetimeIndex at bar *open* time (the standard
            convention in this codebase).
        df_funding: Funding rate history DataFrame, typically loaded via
            :func:`src.config.paths.load_funding_rate_csv`.  Must have a
            ``funding_rate`` column and a DatetimeIndex set to each
            *settlement* timestamp.

    Returns:
        Tuple ``(df, has_funding_rate)``.  On success, ``df`` has a new
        column ``funding_rate_current`` with the most-recent-settled rate
        for each row.  On failure, the original df is returned and
        ``has_funding_rate`` is ``False``.
    """
    if df_funding.empty:
        logger.warning("Funding rate data is empty. funding_rate_current skipped.")
        return df, False

    if "funding_rate" not in df_funding.columns:
        logger.warning(
            "Funding rate df missing required 'funding_rate' column — skipping."
        )
        return df, False

    df_funding = df_funding[~df_funding.index.duplicated(keep="last")].sort_index()

    # NOTE: No index shift here.  Unlike OHLC candles (where index = OPEN
    # time and the CLOSE value is only known N hours later), a funding rate
    # CSV entry's index IS the exact settlement moment — the value is
    # publicly available from that microsecond onwards.  merge_asof(backward)
    # therefore gives exactly the right temporal semantics.

    df = df.sort_index()
    df.index = df.index.astype("datetime64[ns]")
    df_funding.index = df_funding.index.astype("datetime64[ns]")

    df_merged = pd.merge_asof(
        df,
        df_funding[["funding_rate"]],
        left_index=True,
        right_index=True,
        direction="backward",
    )
    df_merged.rename(columns={"funding_rate": "funding_rate_current"}, inplace=True)
    return df_merged, True


def add_onchain_active_addresses(
    df: pd.DataFrame, df_onchain: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """Add daily onchain active addresses onto sub-daily candles.

    Replicates the exact ``shift + merge_asof(direction="backward")`` pattern
    of :func:`add_trend_htf` — the only difference is the number of days
    shifted before the merge.

    Why +2 days (conservative shift)
    ---------------------------------
    Blockchain.com's ``n-unique-addresses`` metric is a *daily aggregate*:
    it counts every unique address that appeared in any transaction during
    the UTC calendar day.  Temporal availability reasoning:

    * The day's data can only be **finalized** once the day closes at
      ``23:59:59 UTC`` — so the earliest *theoretical* availability is
      ``D+1 00:00 UTC`` (same as :func:`add_trend_htf`'s +1 day shift).
    * Blockchain.com **does not publish a latency SLA** for this endpoint.
      Without a documented guarantee, a +1 day shift cannot be relied
      upon: the aggregation pipeline may not complete by D+1 00:00 UTC.
    * The **conservative choice is +2 days**: this assumes up to 24 hours
      of aggregation delay beyond the day close, with no empirical
      verification of the actual Blockchain.com publish time.  It trades
      one extra day of signal lag for a hard leakage-free guarantee.
    * Decision documented here and in the leakage test
      ``tests/features/test_onchain_active_addresses_leakage.py``.
      Revisit if Blockchain.com publishes an official SLA in future.

    Resulting temporal semantics
    ----------------------------
    * Any sub-daily bar opening during calendar day ``D`` inherits the
      onchain value from day ``D-2`` (the last *conservatively available*
      closed day).
    * A bar at ``D 00:00 UTC`` inherits ``D-2``'s value (NOT ``D-1``).
    * The first ``2 * 24h`` of bars in a series are ``NaN`` — confirming
      the shift is real and no pre-release data sneaks through.

    Args:
        df: Sub-daily DataFrame (e.g. 4h, 1h) with OHLCV data and
            technical indicators.  Index must be a ``DatetimeIndex``.
        df_onchain: Daily onchain DataFrame with a single column
            ``onchain_active_addresses``, typically loaded via
            :func:`src.config.paths.load_onchain_active_addresses_csv`.
            Index must be a midnight-aligned ``DatetimeIndex``.

    Returns:
        Tuple ``(df, has_onchain)`` indicating whether the feature was
        added.  On failure the original ``df`` is returned unchanged.
    """
    if df_onchain.empty:
        logger.warning(
            "Onchain active addresses data is empty — feature will be skipped."
        )
        return df, False

    required_col = "onchain_active_addresses"
    if required_col not in df_onchain.columns:
        logger.warning(
            "Onchain DataFrame missing column '%s' — skipping. "
            "Available columns: %s",
            required_col,
            list(df_onchain.columns),
        )
        return df, False

    df_onchain = df_onchain[~df_onchain.index.duplicated(keep="last")].sort_index()

    # CONSERVATIVE LEAKAGE PREVENTION (+2 days):
    # Each daily entry's index is midnight of day D (its start / open time).
    # The aggregate is only fully finalized at D+1 00:00 at the earliest.
    # Blockchain.com publishes no latency SLA for this endpoint, so the
    # actual publish time is unknown — up to 24h of delay beyond day close
    # is assumed conservatively (no empirical verification of actual
    # publish time). Shifting the index forward by 2 calendar days means
    # a sub-daily bar during day D can at most see the value for day D-2.
    df_onchain = df_onchain.copy()
    df_onchain.index = df_onchain.index + pd.Timedelta(days=2)

    df = df.sort_index()
    df.index = df.index.astype("datetime64[ns]")
    df_onchain.index = df_onchain.index.astype("datetime64[ns]")

    df = pd.merge_asof(
        df,
        df_onchain[[required_col]],
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return df, True


def add_mempool_fee_rate_p50(
    df: pd.DataFrame, df_mempool: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """Add daily median mempool fee-rate (sat/vB) onto sub-daily candles.

    Replicates the exact ``+1 day shift + merge_asof(direction="backward")``
    pattern of :func:`add_trend_htf`.

    Shift decision (+1 day)
    ------------------------
    The mempool.space backend indexes ``fee_rate_percentiles`` synchronously
    in the same block-processing cycle as block confirmation, directly from
    Bitcoin Core RPC (verified in ``backend/src/api/blocks.ts``).  There is
    no async pipeline or artificial delay.  Data is available within seconds
    of block confirmation.

    However, each entry in the ``fee-rates/all`` endpoint represents a full
    calendar day of ~144 confirmed blocks.  The daily aggregate is only
    complete once the day has closed — i.e. no earlier than ``D+1 00:00 UTC``.
    A **+1 day shift** is therefore both necessary and sufficient:

    * +1 day (like ``trend_htf``): appropriate because mempool.space has no
      publication SLA but the open-source code shows no artificial delay —
      the data is available as soon as the last block of the day is indexed.
    * +2 days (like ``onchain_active_addresses``) would be overly conservative
      given the transparent, synchronous indexing architecture.

    Resulting temporal semantics
    ----------------------------
    * Sub-daily bars opening during day D → inherit value from day D-1.
    * Bar at D+1 00:00 UTC → inherits the just-closed day D value.
    * First calendar day of bars → NaN (no closed day available yet).

    Args:
        df: Sub-daily DataFrame (e.g. 4h, 1h) with OHLCV data and
            technical indicators.  Index must be a ``DatetimeIndex``.
        df_mempool: Daily mempool fee-rate DataFrame with a single column
            ``mempool_fee_rate_p50`` (sat/vB), typically loaded via
            :func:`src.config.paths.load_mempool_fee_rate_csv`.
            Index must be a midnight-aligned ``DatetimeIndex``.

    Returns:
        Tuple ``(df, has_mempool_fee_rate)`` indicating whether the feature
        was added.  On failure the original ``df`` is returned unchanged.
    """
    if df_mempool.empty:
        logger.warning(
            "Mempool fee-rate data is empty — feature will be skipped."
        )
        return df, False

    required_col = "mempool_fee_rate_p50"
    if required_col not in df_mempool.columns:
        logger.warning(
            "Mempool DataFrame missing column '%s' — skipping. "
            "Available columns: %s",
            required_col,
            list(df_mempool.columns),
        )
        return df, False

    df_mempool = df_mempool[~df_mempool.index.duplicated(keep="last")].sort_index()

    # +1 DAY SHIFT (same rationale as add_trend_htf):
    # Each daily entry's index is midnight of day D (its start / open time).
    # The daily aggregate is complete at D+1 00:00 UTC at the earliest.
    # Shifting the index forward by 1 day makes the value "available" only
    # from D+1 00:00 onward — merge_asof(backward) then correctly picks
    # the most recent fully-closed day for every sub-daily bar.
    df_mempool = df_mempool.copy()
    df_mempool.index = df_mempool.index + pd.Timedelta(days=1)

    df = df.sort_index()
    df.index = df.index.astype("datetime64[ns]")
    df_mempool.index = df_mempool.index.astype("datetime64[ns]")

    df = pd.merge_asof(
        df,
        df_mempool[[required_col]],
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return df, True
