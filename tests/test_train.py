"""Tests for src.brain.train — Model training factory."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.brain.train import train_factory

SAMPLE_CONFIG: dict[str, Any] = {
    "strategy_name": "Momentum",
    "features": ["rsi_14", "macd"],
    "optimal_threshold": 0.65,
    "atr_tp_multi": 2.0,
    "atr_sl_multi": 1.0,
    "swing_period": 7,
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
}


def _make_training_df() -> pd.DataFrame:
    """Synthetic DataFrame with features and target for training."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "rsi_14": np.random.uniform(20, 80, n),
            "macd": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )


class TestTrainFactory:
    @patch("src.brain.train.get_project_root")
    def test_raises_when_config_not_found(
        self, mock_root: MagicMock, tmp_path: Path
    ) -> None:
        """Without config.json, raises RuntimeError."""
        mock_root.return_value = tmp_path

        with pytest.raises(RuntimeError, match="Configuration not found at"):
            train_factory("BTC_USDT")

    @patch("src.brain.train.get_project_root")
    def test_raises_on_invalid_json(self, mock_root: MagicMock, tmp_path: Path) -> None:
        """Malformed JSON raises RuntimeError."""
        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text("{{invalid json", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Error parsing config.json"):
            train_factory("BTC_USDT")

    @patch("src.brain.train.load_csv_data", side_effect=FileNotFoundError("not found"))
    @patch("src.brain.train.get_project_root")
    def test_raises_when_csv_not_found(
        self, mock_root: MagicMock, _mock_csv: MagicMock, tmp_path: Path
    ) -> None:
        """CSV not found raises RuntimeError."""
        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps(SAMPLE_CONFIG), encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="Data file not found for"):
            train_factory("BTC_USDT")

    @patch("src.brain.train.joblib.dump")
    @patch("src.brain.train.cleanup_columns")
    @patch("src.brain.train.compute_target")
    @patch("src.brain.train.add_sentiment", return_value=(None, False))
    @patch("src.brain.train.compute_all_technicals")
    @patch("src.brain.train.load_csv_data")
    @patch("src.brain.train.get_project_root")
    def test_trains_and_exports_model_on_success(
        self,
        mock_root: MagicMock,
        mock_csv: MagicMock,
        mock_technicals: MagicMock,
        mock_sentiment: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_dump: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successful training exports a model with joblib.dump."""
        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps(SAMPLE_CONFIG), encoding="utf-8"
        )

        df = _make_training_df()
        mock_csv.return_value = df
        mock_technicals.return_value = df
        mock_sentiment.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        train_factory("BTC_USDT")

        mock_dump.assert_called_once()
        saved_dict = mock_dump.call_args[0][0]
        assert "model" in saved_dict
        assert "features" in saved_dict
        assert "threshold" in saved_dict
        assert "atr_tp_multi" in saved_dict
        assert "atr_sl_multi" in saved_dict
        assert "strategy_name" in saved_dict

    @patch("src.brain.train.joblib.dump")
    @patch("src.brain.train.cleanup_columns")
    @patch("src.brain.train.compute_target")
    @patch("src.brain.train.add_sentiment")
    @patch("src.brain.train.compute_all_technicals")
    @patch("src.brain.train.load_csv_data")
    @patch("src.brain.train.get_project_root")
    def test_handles_imbalanced_data_all_zeros(
        self,
        mock_root: MagicMock,
        mock_csv: MagicMock,
        mock_technicals: MagicMock,
        mock_sentiment: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_dump: MagicMock,
        tmp_path: Path,
    ) -> None:
        """All-zero target (imbalance=1) does not cause errors."""
        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        (config_dir / "config.json").write_text(
            json.dumps(SAMPLE_CONFIG), encoding="utf-8"
        )

        df = _make_training_df()
        df["target"] = 0
        mock_csv.return_value = df
        mock_technicals.return_value = df
        mock_sentiment.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df

        train_factory("BTC_USDT")
        mock_dump.assert_called_once()
