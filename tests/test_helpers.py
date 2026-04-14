"""Tests for src.utils.helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.helpers import (
    SENTIMENT_COLS,
    build_strategies,
    cleanup_columns,
    compute_target,
    find_optimal_threshold,
    load_csv_data,
    temporal_split_with_embargo,
    train_and_evaluate,
)


@pytest.fixture
def sample_ohlcv_csv(tmp_path: Path) -> Path:
    """Create a synthetic OHLCV CSV for tests."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(200, 300, 100),
            "low": np.random.uniform(50, 100, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 5000, 100),
        }
    )
    csv_dir = tmp_path / "data" / "raw_csv"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "TEST_USDT_1d.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


class TestLoadCsvData:
    def test_loads_valid_csv(self, sample_ohlcv_csv: Path) -> None:
        from unittest.mock import patch

        with patch(
            "src.utils.helpers.get_project_root", return_value=sample_ohlcv_csv
        ):
            df = load_csv_data("TEST_USDT_1d.csv")
            assert isinstance(df, pd.DataFrame)
            assert "close" in df.columns
            assert df.index.name == "timestamp"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        with patch("src.utils.helpers.get_project_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError):
                load_csv_data("NONEXISTENT.csv")


class TestComputeTarget:
    def test_adds_target_column(self) -> None:
        df = pd.DataFrame(
            {
                "close": [100.0, 110.0, 105.0, 115.0, 120.0] * 10,
                "high": [105.0, 115.0, 110.0, 120.0, 125.0] * 10,
                "low": [95.0, 105.0, 100.0, 110.0, 115.0] * 10,
                "atr_14": [5.0] * 50,
            }
        )
        result = compute_target(df, swing_days=3, atr_tp_multi=1.0, atr_sl_multi=1.0)
        assert "target" in result.columns

    def test_raises_without_atr(self) -> None:
        df = pd.DataFrame({"close": [100.0], "high": [105.0], "low": [95.0]})
        with pytest.raises(ValueError, match="atr_14"):
            compute_target(df, swing_days=3)


class TestTemporalSplitWithEmbargo:
    def test_split_sizes(self) -> None:
        df = pd.DataFrame({"a": range(100)})
        train, val, test = temporal_split_with_embargo(
            df, train_pct=0.7, val_pct=0.1, embargo_days=5
        )
        assert len(train) == 70
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) < 100

    def test_embargo_gap_exists(self) -> None:
        df = pd.DataFrame({"a": range(100)}, index=range(100))
        train, val, test = temporal_split_with_embargo(df, embargo_days=5)
        assert val.index[0] > train.index[-1]
        assert test.index[0] > val.index[-1]


class TestComputeTargetEdgeCases:
    def test_target_contains_only_binary_values(self) -> None:
        """target must contain only 0 and 1."""
        df = pd.DataFrame(
            {
                "close": np.random.uniform(100, 200, 50),
                "high": np.random.uniform(200, 300, 50),
                "low": np.random.uniform(50, 100, 50),
                "atr_14": [5.0] * 50,
            }
        )

        result = compute_target(df, swing_days=3, atr_tp_multi=1.0, atr_sl_multi=1.0)

        assert set(result["target"].dropna().unique()).issubset({0, 1})

    def test_with_zero_atr_multipliers(self) -> None:
        """Zero ATR multipliers must not cause errors."""
        df = pd.DataFrame(
            {
                "close": [100.0] * 20,
                "high": [105.0] * 20,
                "low": [95.0] * 20,
                "atr_14": [5.0] * 20,
            }
        )

        result = compute_target(df, swing_days=3, atr_tp_multi=0.0, atr_sl_multi=0.0)

        assert "target" in result.columns


class TestCleanupColumns:
    def test_drops_ohlcv_and_auxiliary_columns(self) -> None:
        """Drops OHLCV, ema_50, vol_sma_20, max/min_future columns."""
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100.0],
                "ema_50": [1.2],
                "vol_sma_20": [90.0],
                "max_high_future": [2.1],
                "min_low_future": [0.4],
                "rsi_14": [50.0],
                "target": [1],
            }
        )

        result = cleanup_columns(df)

        for col in ["open", "high", "low", "close", "volume", "ema_50", "vol_sma_20"]:
            assert col not in result.columns
        assert "rsi_14" in result.columns
        assert "target" in result.columns

    def test_drops_nan_rows(self) -> None:
        """Rows containing NaN are removed."""
        df = pd.DataFrame(
            {
                "rsi_14": [50.0, np.nan, 60.0],
                "target": [1, 0, 1],
            }
        )

        result = cleanup_columns(df)

        assert len(result) == 2
        assert not result.isna().any().any()


class TestTemporalSplitEdgeCases:
    def test_with_very_small_dataframe(self) -> None:
        """Very small DataFrame does not cause errors."""
        df = pd.DataFrame({"a": range(10)})

        train, val, test = temporal_split_with_embargo(df, embargo_days=1)

        assert len(train) > 0


class TestFindOptimalThreshold:
    def test_returns_default_when_no_signals_pass(self) -> None:
        """If no threshold produces enough signals, returns (0.65, 0.0)."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]] * 20)
        X_val = pd.DataFrame({"f1": range(20)})
        y_val = pd.Series([0] * 20)

        threshold, profit = find_optimal_threshold(mock_model, X_val, y_val, 2.0, 1.0)

        assert threshold == 0.65
        assert profit == 0.0

    def test_picks_best_profit_threshold(self) -> None:
        """Selects the threshold with the highest net profit."""
        mock_model = MagicMock()
        probas = np.column_stack([
            np.zeros(50),
            np.linspace(0.4, 0.9, 50),
        ])
        mock_model.predict_proba.return_value = probas
        X_val = pd.DataFrame({"f1": range(50)})
        y_val = pd.Series([1] * 30 + [0] * 20)

        threshold, profit = find_optimal_threshold(mock_model, X_val, y_val, 2.0, 1.0)

        assert 0.50 <= threshold < 0.85
        assert profit > 0


class TestBuildStrategies:
    def test_without_sentiment_excludes_fng_columns(self) -> None:
        """Without sentiment, strategies do not include FNG columns."""
        df = pd.DataFrame(
            {
                "rsi_14": [50.0],
                "macd": [0.1],
                "macd_hist": [0.05],
                "stoch_k": [60.0],
                "dist_ema_50": [0.01],
                "adx_14": [25.0],
                "atr_14": [5.0],
                "bb_width": [0.1],
                "bb_pos": [0.5],
                "obv": [1000.0],
                "rel_volume": [1.2],
                "target": [1],
            }
        )

        strategies = build_strategies(df, has_sentiment=False)

        for name, features in strategies.items():
            for col in SENTIMENT_COLS:
                assert col not in features
        assert "Puramente Momentum + Sentimiento" not in strategies

    def test_with_sentiment_adds_fng_columns(self) -> None:
        """With sentiment, '+Sentimiento' variants including FNG are created."""
        df = pd.DataFrame(
            {
                "rsi_14": [50.0],
                "macd": [0.1],
                "macd_hist": [0.05],
                "stoch_k": [60.0],
                "dist_ema_50": [0.01],
                "adx_14": [25.0],
                "atr_14": [5.0],
                "bb_width": [0.1],
                "bb_pos": [0.5],
                "obv": [1000.0],
                "rel_volume": [1.2],
                "target": [1],
            }
        )

        strategies = build_strategies(df, has_sentiment=True)

        assert "Puramente Momentum + Sentimiento" in strategies
        sentiment_features = strategies["Puramente Momentum + Sentimiento"]
        for col in SENTIMENT_COLS:
            assert col in sentiment_features


class TestTrainAndEvaluate:
    def test_returns_correct_tuple_structure(self, ohlcv_df_with_technicals: pd.DataFrame) -> None:
        """Verifies return structure: (model, metrics, preds_test, buy_dates, threshold)."""
        df = ohlcv_df_with_technicals.copy()
        df["target"] = np.random.randint(0, 2, len(df))
        df.dropna(inplace=True)
        features = ["rsi_14", "macd", "atr_14"]
        n = len(df)
        split = int(n * 0.6)
        val_split = int(n * 0.8)

        X_train = df[features].iloc[:split]
        X_val = df[features].iloc[split:val_split]
        X_test = df[features].iloc[val_split:]
        y_train = df["target"].iloc[:split]
        y_val = df["target"].iloc[split:val_split]
        y_test = df["target"].iloc[val_split:]

        result = train_and_evaluate(
            X_train, X_val, X_test, y_train, y_val, y_test, tp_val=2.0, sl_val=1.0
        )

        assert len(result) == 5
        model, metrics, preds_test, buy_dates, threshold = result
        assert hasattr(model, "predict_proba")
        assert isinstance(metrics, dict)
        assert "Profit_Neto" in metrics
        assert isinstance(preds_test, np.ndarray)
        assert isinstance(threshold, float)

    def test_handles_fully_imbalanced_target_all_zeros(
        self, ohlcv_df_with_technicals: pd.DataFrame
    ) -> None:
        """All-zero target does not cause errors (imbalance = 1)."""
        df = ohlcv_df_with_technicals.copy()
        df["target"] = 0
        df.dropna(inplace=True)
        features = ["rsi_14", "macd"]
        n = len(df)
        split = int(n * 0.6)
        val_split = int(n * 0.8)

        X_train = df[features].iloc[:split]
        X_val = df[features].iloc[split:val_split]
        X_test = df[features].iloc[val_split:]
        y_train = df["target"].iloc[:split]
        y_val = df["target"].iloc[split:val_split]
        y_test = df["target"].iloc[val_split:]

        result = train_and_evaluate(
            X_train, X_val, X_test, y_train, y_val, y_test, tp_val=2.0, sl_val=1.0
        )
        assert result is not None