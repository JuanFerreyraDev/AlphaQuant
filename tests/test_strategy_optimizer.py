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
            "close": np.random.uniform(100, 200, n),
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
    @patch(
        "src.brain.strategy_optimizer.load_csv_data",
        side_effect=FileNotFoundError("not found"),
    )
    def test_raises_when_csv_not_found(self, _mock: MagicMock) -> None:
        """CSV not found raises RuntimeError."""
        with pytest.raises(RuntimeError, match="File data/raw_csv/.* not found"):
            optimize_strategy("NONEXISTENT_USDT")

    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies", return_value={})
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
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
        mock_dynamic_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Without satisfactory results, raises RuntimeError."""
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        mock_strategies.return_value = {}

        with pytest.raises(RuntimeError, match="No satisfactory results found for"):
            optimize_strategy("BTC_USDT")

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
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
        mock_dynamic_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Writes the best config.json with the winning strategy."""
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        mock_strategies.return_value = {
            "Momentum": ["rsi_14", "macd"],
        }

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.zeros((10, 2))
        preds_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        metrics = {
            "net_profit_pct": 5.0,
            "val_fitness_score": 3.5,
            "test_fitness_score": 2.0,
            "test_profit_factor": 2.1,
            "test_max_drawdown": 0.12,
            "test_trade_count": 10,
            "val_profit_factor": 2.5,
            "val_max_drawdown": 0.10,
            "val_trade_count": 14,
        }
        mock_train.return_value = (mock_model, metrics, preds_test, [], 0.65)

        optimize_strategy("BTC_USDT")

        config_path = tmp_path / "data" / "models" / "BTC_USDT" / "config.json"
        assert config_path.exists()
        with config_path.open("r") as fh:
            config = json.load(fh)
        assert "strategy_name" in config
        assert "features" in config
        assert "optimal_threshold" in config
        assert "last_trained" in config
        assert "test_profit_factor" in config
        assert "test_max_drawdown" in config
        assert "test_trade_count" in config

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
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
        mock_dynamic_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Strategies with non-existent features in df are skipped."""
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        mock_strategies.return_value = {
            "Invalid": ["nonexistent_col_1", "nonexistent_col_2"],
        }

        with pytest.raises(RuntimeError, match="No satisfactory results found for"):
            optimize_strategy("BTC_USDT")
        mock_train.assert_not_called()

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
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
        mock_dynamic_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Selects the configuration with the highest val_fitness_score."""
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        mock_strategies.return_value = {
            "Low": ["rsi_14"],
            "High": ["macd"],
        }

        preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # The optimizer iterates over SWING_RANGE × ATR_TP_RANGE × ATR_SL_RANGE combos (36),
        # each time calling train_and_evaluate for each strategy (2).
        # Return low fitness for "Low" and high fitness for "High" on every call.
        def _side_effect(*args, **kwargs):
            # Detect which strategy by the X_train columns
            X_train = args[0]
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.zeros((len(preds), 2))
            if "macd" in X_train.columns:
                return (
                    mock_model,
                    {
                        "net_profit_pct": 10.0,
                        "val_fitness_score": 5.0,
                        "test_fitness_score": 4.0,
                        "test_profit_factor": 3.0,
                        "test_max_drawdown": 0.10,
                        "test_trade_count": 15,
                        "val_profit_factor": 3.5,
                        "val_max_drawdown": 0.08,
                        "val_trade_count": 18,
                    },
                    preds,
                    [],
                    0.70,
                )
            return (
                mock_model,
                {
                    "net_profit_pct": 1.0,
                    "val_fitness_score": 0.5,
                    "test_fitness_score": 0.3,
                    "test_profit_factor": 1.1,
                    "test_max_drawdown": 0.40,
                    "test_trade_count": 6,
                    "val_profit_factor": 1.2,
                    "val_max_drawdown": 0.35,
                    "val_trade_count": 8,
                },
                preds,
                [],
                0.60,
            )

        mock_train.side_effect = _side_effect

        optimize_strategy("BTC_USDT")

        config_path = tmp_path / "data" / "models" / "BTC_USDT" / "config.json"
        with config_path.open("r") as fh:
            config = json.load(fh)
        assert config["strategy_name"] == "High"
        assert config["optimal_threshold"] == 0.70
