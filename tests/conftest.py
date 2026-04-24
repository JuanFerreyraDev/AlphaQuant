"""Shared fixtures for the AlphaQuant test suite."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Synthetic 100-row OHLCV DataFrame with realistic price action."""
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1000, 5000, n),
        }
    )

@pytest.fixture
def ohlcv_df_with_technicals(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame with all technical indicators pre-computed."""
    from src.brain.features import compute_all_technicals

    return compute_all_technicals(ohlcv_df.copy())

@pytest.fixture
def sample_model_dict() -> dict[str, Any]:
    """Valid model dict with all REQUIRED_KEYS expected by tasks.py."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return {
        "model": mock_model,
        "features": ["rsi_14", "macd"],
        "threshold": 0.6,
        "atr_tp_multi": 2.0,
        "atr_sl_multi": 1.0,
        "strategy_name": "TestStrategy",
    }