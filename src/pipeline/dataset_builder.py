"""Build walk-forward-ready DataFrames for any symbol and feature profile."""

from __future__ import annotations

import pandas as pd

from src.config.experiment_defaults import ExperimentConfig
from src.pipeline.feature_profiles import (
    COLS_TO_DROP,
    ENRICHMENT_REGISTRY,
    REQUIRED_BASE_FEATURES,
    FeatureProfile,
    get_profile,
)
from src.utils.helpers import SENTIMENT_COLS, compute_target, load_csv_data
from src.utils.timeframe_utils import parse_timeframe_hours


def build_dataset(
    symbol: str,
    timeframe: str,
    profile: str | FeatureProfile = "control",
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV data, apply enrichments, compute targets, and infer feature columns.

    Args:
        symbol: Trading pair (e.g. ``'SOL_USDT'``).
        timeframe: Candle interval (e.g. ``'4h'``).
        profile: Registered profile name or ``FeatureProfile`` instance.
        config: Walk-forward target parameters. Uses defaults when ``None``.

    Returns:
        Tuple of ``(df, control_features)`` where ``control_features`` excludes
        the profile's treatment column (if any). The treatment column remains
        in ``df`` for A/B experiments.
    """
    cfg = config or ExperimentConfig()
    prof = profile if isinstance(profile, FeatureProfile) else get_profile(profile)
    tf_hours = parse_timeframe_hours(timeframe)

    df = load_csv_data(symbol, timeframe)
    for enrichment in prof.enrichments:
        df = ENRICHMENT_REGISTRY[enrichment](df, symbol)

    compute_target(
        df,
        swing_days=cfg.swing_period,
        atr_tp_multi=cfg.tp_multi,
        atr_sl_multi=cfg.sl_multi,
        timeframe_hours=tf_hours,
    )

    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    exclude = {"close", "target", "target_ret"}
    if prof.treatment_col:
        exclude.add(prof.treatment_col)

    control_features = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    missing_base = REQUIRED_BASE_FEATURES - set(control_features)
    if missing_base:
        raise RuntimeError(
            f"Missing base features {missing_base} for {symbol}/{timeframe} "
            f"profile={prof.name} — available: {list(df.columns)}"
        )

    if not any(c in control_features for c in SENTIMENT_COLS):
        raise RuntimeError(f"Sentiment columns missing for {symbol}/{timeframe}")

    if prof.treatment_col and prof.treatment_col not in df.columns:
        raise RuntimeError(
            f"Treatment column {prof.treatment_col!r} missing after enrichments"
        )

    return df, control_features
