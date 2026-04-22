"""Tests for src.brain.strategy_optimizer — Strategy optimization pipeline."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.brain.strategy_optimizer import optimize_strategy


def _make_df() -> pd.DataFrame:
    """Synthetic DataFrame with features, target, and expected columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "rsi_14": np.random.uniform(20, 80, n),
            "macd": np.random.randn(n),
            "macd_hist": np.random.randn(n),
            "stoch_k": np.random.uniform(0, 100, n),
            "dist_ema_50": np.random.randn(n) * 0.01,
            "adx_14": np.random.uniform(10, 50, n),
            "atr_14": np.random.uniform(1, 10, n),
            "bb_width": np.random.uniform(0, 0.5, n),
            "bb_pos": np.random.uniform(0, 1, n),
            "obv": np.random.uniform(1000, 5000, n),
            "rel_volume": np.random.uniform(0.5, 2, n),
            "target": np.random.randint(0, 2, n),
        }
    )


class TestOptimizeStrategy:
    @patch("src.brain.strategy_optimizer.load_csv_data", side_effect=FileNotFoundError("not found"))
    def test_raises_when_csv_not_found(self, _mock: MagicMock) -> None:
        """CSV not found raises RuntimeError."""
        with pytest.raises(RuntimeError, match="File data/raw_csv/.* not found"):
            optimize_strategy("NONEXISTENT_USDT")

    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies", return_value={})
    @patch("src.brain.strategy_optimizer.temporal_split_with_embargo")
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment", return_value=(None, False))
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_returns_early_when_no_results_found(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Without satisfactory results, raises RuntimeError."""
        # Arrange
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        train_df = df.iloc[:70]
        val_df = df.iloc[75:85]
        test_df = df.iloc[90:]
        mock_split.return_value = (train_df, val_df, test_df)
        mock_strategies.return_value = {}  # No strategies → no results

        # Act / Assert
        with pytest.raises(RuntimeError, match="No satisfactory results found for"):
            optimize_strategy("BTC_USDT")

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.temporal_split_with_embargo")
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_writes_best_config_to_json(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Writes the best config.json with the winning strategy."""
        # Arrange
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        train_df = df.iloc[:70]
        val_df = df.iloc[75:85]
        test_df = df.iloc[90:]
        mock_split.return_value = (train_df, val_df, test_df)

        mock_strategies.return_value = {
            "Momentum": ["rsi_14", "macd"],
        }

        mock_model = MagicMock()
        preds_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        metrics = {"Profit_Neto": 5.0}
        mock_train.return_value = (mock_model, metrics, preds_test, [], 0.65)

        # Act
        optimize_strategy("BTC_USDT")

        # Assert
        config_path = tmp_path / "data" / "models" / "BTC_USDT" / "config.json"
        assert config_path.exists()
        with config_path.open("r") as fh:
            config = json.load(fh)
        assert "strategy_name" in config
        assert "features" in config
        assert "optimal_threshold" in config
        assert "last_trained" in config

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.temporal_split_with_embargo")
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_skips_strategies_with_no_valid_features(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Strategies with non-existent features in df are skipped."""
        # Arrange
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        train_df = df.iloc[:70]
        val_df = df.iloc[75:85]
        test_df = df.iloc[90:]
        mock_split.return_value = (train_df, val_df, test_df)

        mock_strategies.return_value = {
            "Invalid": ["nonexistent_col_1", "nonexistent_col_2"],
        }

        # Act / Assert — no valid features → no results → RuntimeError
        with pytest.raises(RuntimeError, match="No satisfactory results found for"):
            optimize_strategy("BTC_USDT")
        mock_train.assert_not_called()

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.temporal_split_with_embargo")
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_selects_highest_profit_config(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Selects the configuration with the highest Profit_Neto."""
        # Arrange
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        train_df = df.iloc[:70]
        val_df = df.iloc[75:85]
        test_df = df.iloc[90:]
        mock_split.return_value = (train_df, val_df, test_df)

        mock_strategies.return_value = {
            "Low": ["rsi_14"],
            "High": ["macd"],
        }

        preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        # The optimizer iterates over SWING_RANGE × ATR_TP_RANGE × ATR_SL_RANGE combos (36),
        # each time calling train_and_evaluate for each strategy (2).
        # Return low profit for "Low" and high profit for "High" on every call.
        def _side_effect(*args, **kwargs):
            # We can detect which strategy by the X_train columns
            X_train = args[0]
            if "macd" in X_train.columns:
                return (MagicMock(), {"Profit_Neto": 10.0}, preds, [], 0.70)
            return (MagicMock(), {"Profit_Neto": 1.0}, preds, [], 0.60)

        mock_train.side_effect = _side_effect

        # Act
        optimize_strategy("BTC_USDT")

        # Assert
        config_path = tmp_path / "data" / "models" / "BTC_USDT" / "config.json"
        with config_path.open("r") as fh:
            config = json.load(fh)
        assert config["strategy_name"] == "High"
        assert config["optimal_threshold"] == 0.70
