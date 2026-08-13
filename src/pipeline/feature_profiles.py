"""Declarative feature enrichment profiles for experiment datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.brain.data_fetcher import get_fear_and_greed
from src.brain.features import (
    add_funding_rate,
    add_sentiment,
    add_taker_buy_ratio,
    add_trend_htf,
    compute_all_technicals,
)
from src.config.paths import load_funding_rate_csv
from src.utils.helpers import SENTIMENT_COLS, load_csv_data

EnrichmentFn = Callable[[pd.DataFrame, str], pd.DataFrame]

REQUIRED_BASE_FEATURES: frozenset[str] = frozenset(
    {"rsi_14", "atr_14", "bb_width", "bb_pos", "obv", "rel_volume"}
)

COLS_TO_DROP: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "volume",
    "ema_50",
    "vol_sma_20",
    "max_high_future",
    "min_low_future",
    "quote_volume",
    "n_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
)


@dataclass(frozen=True)
class FeatureProfile:
    """Describes which enrichments to apply and the A/B treatment column."""

    name: str
    enrichments: tuple[str, ...]
    treatment_col: str | None = None
    extra_csv_requirements: tuple[str, ...] = ()


def _apply_sentiment(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df_fg = get_fear_and_greed()
    df, has_sentiment = add_sentiment(df, df_fg)
    if not has_sentiment:
        raise RuntimeError(f"sentiment not loaded for {symbol}")
    return df


def _apply_trend_htf(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df_1d = load_csv_data(symbol, "1d")
    df, has_trend = add_trend_htf(df, df_1d)
    if not has_trend:
        raise RuntimeError(f"trend_htf not added for {symbol} — need 1d.csv")
    return df


def _apply_funding_rate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df_funding = load_funding_rate_csv(symbol)
    df, has_funding = add_funding_rate(df, df_funding)
    if not has_funding:
        raise RuntimeError(
            f"funding_rate_current not added for {symbol} — "
            f"need data/raw_csv/{symbol}/funding_rate.csv"
        )
    return df


def _apply_taker_buy_ratio(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df, has_taker = add_taker_buy_ratio(df)
    if not has_taker:
        raise RuntimeError(
            f"taker_buy_ratio not computed for {symbol} — "
            "re-download CSV with --binance-rest"
        )
    return df


ENRICHMENT_REGISTRY: dict[str, EnrichmentFn] = {
    "technicals": lambda df, _symbol: compute_all_technicals(df),
    "sentiment": _apply_sentiment,
    "trend_htf": _apply_trend_htf,
    "funding_rate": _apply_funding_rate,
    "taker_buy_ratio": _apply_taker_buy_ratio,
}


FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "control": FeatureProfile(
        name="control",
        enrichments=("technicals", "sentiment"),
    ),
    "trend_htf": FeatureProfile(
        name="trend_htf",
        enrichments=("technicals", "sentiment", "trend_htf"),
        treatment_col="trend_htf",
        extra_csv_requirements=("1d.csv",),
    ),
    "funding_rate": FeatureProfile(
        name="funding_rate",
        enrichments=("technicals", "sentiment", "funding_rate"),
        treatment_col="funding_rate_current",
        extra_csv_requirements=("funding_rate.csv",),
    ),
    "taker_buy_ratio": FeatureProfile(
        name="taker_buy_ratio",
        enrichments=("technicals", "sentiment", "taker_buy_ratio"),
        treatment_col="taker_buy_ratio",
    ),
}


def get_profile(name: str) -> FeatureProfile:
    """Return a registered feature profile by name."""
    try:
        return FEATURE_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Available: {available}") from exc


def list_profile_names() -> list[str]:
    """Return sorted registered profile names."""
    return sorted(FEATURE_PROFILES)
