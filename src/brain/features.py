"""features.py — Technical indicator computation on OHLCV data.

Groups functions for momentum, trend, volatility, and volume
using the ``pandas_ta`` library.
"""

import logging

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
