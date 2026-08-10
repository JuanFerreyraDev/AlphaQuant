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
        with pytest.raises(RuntimeError, match="not found"):
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

        optimize_strategy("BTC_USDT", audit=True)

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

        optimize_strategy("BTC_USDT", audit=True)

        config_path = tmp_path / "data" / "models" / "BTC_USDT" / "config.json"
        with config_path.open("r") as fh:
            config = json.load(fh)
        assert config["strategy_name"] == "High"
        assert config["optimal_threshold"] == 0.70


class TestTimeframeCalibrationWiring:
    """Regression tests ensuring get_calibrated_constants(timeframe) is
    actually threaded into BOTH compute_dynamic_split call sites (the
    main grid-search loop and the fast-mode winner re-split) AND into
    compute_min_val_trades — previously the kill-switch call silently
    used the 1d-only module defaults regardless of timeframe."""

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.train_and_evaluate_val_only")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_min_val_trades", return_value=5)
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_4h_uses_4h_calibration_everywhere(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_dynamic_split: MagicMock,
        mock_min_val_trades: MagicMock,
        mock_strategies: MagicMock,
        mock_train_val_only: MagicMock,
        mock_train: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        mock_strategies.return_value = {"Momentum": ["rsi_14", "macd"]}

        mock_train_val_only.return_value = (
            MagicMock(),
            {
                "val_fitness_score": 3.5,
                "val_profit_factor": 2.5,
                "val_max_drawdown": 0.10,
                "val_trade_count": 14,
                "net_profit_pct": 5.0,
            },
            0.65,
        )
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.zeros((10, 2))
        preds_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mock_train.return_value = (
            mock_model,
            {
                "net_profit_pct": 5.0,
                "test_fitness_score": 2.0,
                "test_profit_factor": 2.1,
                "test_max_drawdown": 0.12,
                "test_trade_count": 10,
            },
            preds_test,
            [],
            0.65,
        )

        optimize_strategy("BTC_USDT", timeframe="4h")

        # Every compute_dynamic_split call (grid loop + fast-mode winner
        # re-split) must use the 4h-calibrated factor, not the 1d default.
        for call in mock_dynamic_split.call_args_list:
            assert call.kwargs["bars_per_trade_safety_factor"] == 2

        # compute_min_val_trades must receive the 4h-calibrated factor/floor.
        mock_min_val_trades.assert_called_once()
        _, kwargs = mock_min_val_trades.call_args
        assert kwargs["bars_per_trade_safety_factor"] == 2
        assert kwargs["absolute_floor"] == 8


class TestQuickMode:
    """Regression tests for the --quick flag (drastically reduced grid for
    fast smoke-testing/diagnostics, without touching the production grid)."""

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_quick_mode_uses_reduced_grid(
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
        """quick=True exercises QUICK_* grid sizes, not the production grid."""
        from src.brain.strategy_optimizer import (
            ATR_SL_RANGE,
            ATR_TP_RANGE,
            QUICK_ATR_SL_RANGE,
            QUICK_ATR_TP_RANGE,
            QUICK_SWING_RANGE,
            SWING_RANGE,
        )

        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        mock_strategies.return_value = {"Momentum": ["rsi_14"]}

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

        # quick=True: compute_target is called once per (swing,tp,sl) combo,
        # which must match the QUICK_* grid size, not the production one.
        optimize_strategy("BTC_USDT", audit=True, quick=True)
        quick_combo_count = (
            len(QUICK_SWING_RANGE) * len(QUICK_ATR_TP_RANGE) * len(QUICK_ATR_SL_RANGE)
        )
        assert mock_target.call_count == quick_combo_count

        mock_target.reset_mock()

        # quick=False (default): falls back to the full production grid.
        optimize_strategy("BTC_USDT", audit=True, quick=False)
        prod_combo_count = len(SWING_RANGE) * len(ATR_TP_RANGE) * len(ATR_SL_RANGE)
        assert mock_target.call_count == prod_combo_count
        assert prod_combo_count > quick_combo_count

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_quick_swing_overrides_swing_but_not_split_sizing(
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
        """--quick-swing changes the swing used for compute_target, but
        compute_dynamic_split must still be sized for the production
        worst-case swing (max(SWING_RANGE)), not quick_swing."""
        from src.brain.strategy_optimizer import SWING_RANGE

        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        mock_strategies.return_value = {"Momentum": ["rsi_14"]}

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

        optimize_strategy("BTC_USDT", audit=True, quick=True, quick_swing=5)

        for call in mock_target.call_args_list:
            assert call.kwargs["swing_days"] == 5

        for call in mock_dynamic_split.call_args_list:
            assert call.kwargs["swing_period"] == max(SWING_RANGE)


class TestFailureDiagnostics:
    """Regression tests for always-write-a-failure-report (Issue: raising
    RuntimeError on failure previously discarded all diagnostic data)."""

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate_val_only")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split", return_value=(70, 10, 20))
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_failure_csv_written_without_audit_flag(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_dynamic_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train_val_only: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Case 2 (not valid_results): the FAILED diagnostic CSV must be
        written even when audit=False (default) — it must not depend on
        remembering to pass --audit."""
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        mock_strategies.return_value = {"Momentum": ["rsi_14"]}

        mock_model = MagicMock()
        val_metrics = {
            "val_fitness_score": -999.0,
            "val_profit_factor": 0.42,
            "val_max_drawdown": 0.5,
            "val_trade_count": 3,
            "val_net_profit_pct": -1.0,
        }
        mock_train_val_only.return_value = (mock_model, val_metrics, 0.65)

        with pytest.raises(RuntimeError, match="No valid configurations found for"):
            optimize_strategy("BTC_USDT", audit=False, quick=True)

        failed_path = (
            tmp_path / "data" / "models" / "BTC_USDT" / "optimization_report_FAILED.csv"
        )
        assert failed_path.exists()
        failed_df = pd.read_csv(failed_path)
        assert len(failed_df) > 0
        assert "val_profit_factor" in failed_df.columns
        pf_values = list(failed_df["val_profit_factor"])
        assert pf_values == sorted(pf_values, reverse=True)

    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.strategy_optimizer.train_and_evaluate_val_only")
    @patch("src.brain.strategy_optimizer.build_strategies")
    @patch("src.brain.strategy_optimizer.compute_dynamic_split")
    @patch("src.brain.strategy_optimizer.cleanup_columns")
    @patch("src.brain.strategy_optimizer.compute_target")
    @patch("src.brain.strategy_optimizer.add_sentiment")
    @patch("src.brain.strategy_optimizer.compute_all_technicals")
    @patch("src.brain.strategy_optimizer.load_csv_data")
    def test_split_none_skips_only_failing_swing(
        self,
        mock_csv: MagicMock,
        mock_tech: MagicMock,
        mock_sent: MagicMock,
        mock_target: MagicMock,
        mock_cleanup: MagicMock,
        mock_dynamic_split: MagicMock,
        mock_strategies: MagicMock,
        mock_train_val_only: MagicMock,
        mock_root: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A swing_period with insufficient data (split=None) is skipped on
        its own — earlier/later swing values in the grid that DO have a
        valid split must still produce a usable result instead of the
        whole run being discarded."""
        mock_root.return_value = tmp_path
        df = _make_df()
        mock_csv.return_value = df
        mock_tech.return_value = df
        mock_sent.return_value = (df, False)
        mock_target.return_value = df
        mock_cleanup.return_value = df
        mock_strategies.return_value = {"Momentum": ["rsi_14"]}

        def _split_side_effect(*args: Any, **kwargs: Any) -> tuple[int, int, int] | None:
            if kwargs.get("embargo_days") == 10:
                return None
            return (70, 10, 20)

        mock_dynamic_split.side_effect = _split_side_effect

        mock_model = MagicMock()
        val_metrics = {
            "val_fitness_score": 3.5,
            "val_profit_factor": 2.5,
            "val_max_drawdown": 0.10,
            "val_trade_count": 14,
            "val_net_profit_pct": 5.0,
        }
        mock_train_val_only.return_value = (mock_model, val_metrics, 0.65)

        # Should NOT raise despite swing_period=10 being skipped for every
        # (tp,sl) combo — swing_period=5 and 7 still produce results.
        df_report = optimize_strategy("BTC_USDT", audit=False)

        assert isinstance(df_report, pd.DataFrame)
        assert not df_report.empty
        config_path = tmp_path / "data" / "models" / "BTC_USDT" / "config.json"
        assert config_path.exists()

class TestOosSanityCheck:
    def test_passes_when_test_metrics_are_healthy(self) -> None:
        from src.brain.strategy_optimizer import _passes_oos_sanity_check
        config = {
            "val_fitness_score": 10.0,
            "test_fitness_score": 8.0,
        }
        assert _passes_oos_sanity_check(config) is True

    def test_fails_when_pf_too_low(self) -> None:
        from src.brain.strategy_optimizer import _passes_oos_sanity_check
        config = {
            "test_profit_factor": 0.9,
            "test_trade_count": 12,
            "test_max_drawdown": 0.20,
            "recent_signals": 2,
        }
        assert _passes_oos_sanity_check(config) is False


class TestCalculateProductionScore:
    def test_calculates_valid_score(self) -> None:
        from src.brain.strategy_optimizer import _calculate_production_score
        config = {
            "val_trade_count": 15,
            "val_profit_factor": 2.0,
            "val_max_drawdown": 0.10,
        }
        score = _calculate_production_score(config)
        # Expected: base (18.0) * pf_ratio (0.8) = 14.4
        # Plus recent signal bonus
        assert score > 14.0

    def test_zero_val_pf_returns_very_low_score(self) -> None:
        from src.brain.strategy_optimizer import _calculate_production_score
        config = {
            "val_trade_count": 15,
            "val_profit_factor": 0.0,
            "val_max_drawdown": 0.10,
        }
        score = _calculate_production_score(config)
        assert score >= -9999.0  # Should return a very low score for invalid config


class TestMainBlock:
    @patch("src.brain.strategy_optimizer.optimize_strategy")
    @patch("src.brain.strategy_optimizer.get_active_symbols_with_timeframe")
    @patch("src.brain.strategy_optimizer.get_project_root")
    @patch("src.brain.data_fetcher.get_fear_and_greed")
    def test_main_batch_mode_success(
        self, mock_fng, mock_root, mock_symbols, mock_opt, tmp_path
    ) -> None:
        """Test batch mode execution."""
        mock_root.return_value = tmp_path
        mock_fng.return_value = pd.DataFrame()
        mock_symbols.return_value = [
            {"symbol": "BTC_USDT", "timeframe": "4h"},
            {"symbol": "ETH_USDT", "timeframe": "1d"}
        ]
        
        from src.brain.strategy_optimizer import main
        import sys
        
        test_args = ["strategy_optimizer.py"]
        with patch.object(sys, "argv", test_args):
            main()
            
        assert mock_opt.call_count == 2
        mock_opt.assert_any_call(
            "BTC_USDT", mock_fng.return_value, audit=False, timeframe="4h",
            quick=False, quick_swing=10,
        )
        mock_opt.assert_any_call(
            "ETH_USDT", mock_fng.return_value, audit=False, timeframe="1d",
            quick=False, quick_swing=10,
        )
