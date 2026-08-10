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
    fitness_score,
    load_csv_data,
    train_and_evaluate
)

SAMPLE_HIPERPARAMS: dict[str, float] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
}

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
    csv_dir = tmp_path / "data" / "raw_csv" / "TEST_USDT"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "1d.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


class TestLoadCsvData:
    def test_loads_valid_csv(self, sample_ohlcv_csv: Path) -> None:
        from unittest.mock import patch

        with patch("src.config.paths.get_project_root", return_value=sample_ohlcv_csv):
            df = load_csv_data("TEST_USDT", "1d")
            assert isinstance(df, pd.DataFrame)
            assert "close" in df.columns
            assert df.index.name == "timestamp"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        with patch("src.utils.helpers.get_project_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="data_fetcher"):
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

    def test_lower_tf_df_determines_tp_sl_order(self) -> None:
        """lower_tf_df walks bars chronologically and fires TP before SL."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0] * 10,
                "high": [101.0] * 10,   # daily high never reaches tp_price=110
                "low": [99.0] * 10,
                "atr_14": [10.0] * 10,
            },
            index=dates,
        )
        # Window for row 0 (2023-01-01): 2023-01-02 to 2023-01-06
        # First bar: low=98 does NOT hit sl_price=90, high=115 hits tp_price=110 → TP fires
        # Second bar would hit SL, but we already exited
        lower_tf = pd.DataFrame(
            {"high": [115.0, 80.0], "low": [98.0, 75.0]},
            index=pd.to_datetime(["2023-01-02 00:00", "2023-01-02 01:00"]),
        )

        result = compute_target(
            df, swing_days=5, atr_tp_multi=1.0, atr_sl_multi=1.0,
            lower_tf_df=lower_tf,
        )

        # Daily fallback would give target=0 (daily high 101 < tp_price 110)
        # lower_tf correctly resolves target=1 for row 0
        assert result["target"].iloc[0] == 1

    def test_lower_tf_pessimistic_intrabar_sl_beats_tp(self) -> None:
        """If a single bar touches both SL and TP, SL is assumed to fire first."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "atr_14": [10.0] * 5,
            },
            index=dates,
        )
        # tp_price = 100 + 10*1.0 = 110, sl_price = 100 - 10*1.0 = 90
        # This bar touches BOTH levels (high >= tp AND low <= sl)
        lower_tf = pd.DataFrame(
            {"high": [115.0], "low": [85.0]},
            index=pd.to_datetime(["2023-01-02 00:00"]),
        )

        result = compute_target(
            df, swing_days=3, atr_tp_multi=1.0, atr_sl_multi=1.0,
            lower_tf_df=lower_tf,
        )

        # Pessimistic rule: SL checked first → target must be 0
        assert result["target"].iloc[0] == 0

    def test_lower_tf_df_does_not_add_extra_columns(self) -> None:
        """lower_tf path must NOT add max_high_future or min_low_future to df."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "atr_14": [10.0] * 5,
            },
            index=dates,
        )
        lower_tf = pd.DataFrame(
            {"high": [115.0], "low": [98.0]},
            index=pd.to_datetime(["2023-01-02 00:00"]),
        )

        compute_target(df, swing_days=3, atr_tp_multi=1.0, atr_sl_multi=1.0, lower_tf_df=lower_tf)

        assert "max_high_future" not in df.columns
        assert "min_low_future" not in df.columns

    def test_lower_tf_df_raises_on_missing_columns(self) -> None:
        """lower_tf_df without high/low raises a ValueError."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "atr_14": [10.0] * 5,
            },
            index=dates,
        )
        bad_ltf = pd.DataFrame(
            {"open": [100.0]}, index=[pd.Timestamp("2023-01-02")]
        )

        with pytest.raises(ValueError, match="missing required columns"):
            compute_target(df, swing_days=3, lower_tf_df=bad_ltf)


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

    def test_no_row_loss_when_all_feature_columns_are_healthy(self) -> None:
        """Regression test: if every feature column is fully populated (no
        NaN), cleanup_columns must not drop any rows. This guards against
        the dynamic-timeframe Issue 1 regression, where an all-NaN
        sentiment column (fng_sma_14/fng_vol_14, produced by joining daily
        Fear & Greed data onto a sub-daily index) caused dropna() to
        silently remove every single row instead of just the expected
        warmup rows.
        """
        n = 50
        df = pd.DataFrame(
            {
                "rsi_14": np.random.uniform(20, 80, n),
                "macd": np.random.randn(n),
                "adx_14": np.random.uniform(10, 50, n),
                "atr_14": np.random.uniform(1, 10, n),
                "fng_value": np.random.uniform(0, 100, n),
                "fng_sma_14": np.random.uniform(0, 100, n),
                "fng_vol_14": np.random.uniform(0, 20, n),
                "target": np.random.randint(0, 2, n),
            }
        )

        result = cleanup_columns(df)

        assert len(result) == n


class TestFindOptimalThreshold:
    def test_returns_default_when_no_signals_pass(self) -> None:
        """If no threshold produces enough signals, returns (-1.0, 0.0)."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]] * 20)
        X_val = pd.DataFrame({"f1": range(20)})
        y_val = pd.Series([0] * 20)
        prices_val = pd.DataFrame(
            {"close": [100.0] * 20, "atr_14": [5.0] * 20}, index=range(20)
        )

        threshold, profit = find_optimal_threshold(
            mock_model, X_val, y_val, 2.0, 1.0, prices_val, 0.001, 0.0005,
            swing_period=5,
        )

        assert threshold == -1.0
        assert profit == 0.0

    def test_picks_best_profit_threshold(self) -> None:
        """Selects the threshold with the highest net return."""
        mock_model = MagicMock()
        probas = np.column_stack(
            [
                np.zeros(50),
                np.linspace(0.4, 0.9, 50),
            ]
        )
        mock_model.predict_proba.return_value = probas
        X_val = pd.DataFrame({"f1": range(50)})
        y_val = pd.Series([1] * 30 + [0] * 20)
        prices_val = pd.DataFrame(
            {"close": [100.0] * 50, "atr_14": [5.0] * 50}, index=range(50)
        )

        threshold, profit = find_optimal_threshold(
            mock_model, X_val, y_val, 2.0, 1.0, prices_val, 0.001, 0.0005,
            swing_period=5,
        )

        assert 0.50 <= threshold < 0.85
        assert profit > 0


class TestFitnessScore:
    """Tests for fitness_score() — validates all three scoring scenarios."""

    def _make_model(self, n: int, all_fire: bool = True) -> MagicMock:
        """Return a mock model whose probas always exceed any threshold."""
        model = MagicMock()
        high = np.column_stack([np.zeros(n), np.ones(n)])
        low = np.column_stack([np.ones(n), np.zeros(n)])
        model.predict_proba.return_value = high if all_fire else low
        return model

    def test_healthy_model_returns_positive_score(self) -> None:
        """PF > 1.0 and enough sequential trades → positive score with profit_factor > 1.0."""
        n = 20
        model = self._make_model(n)
        X_val = pd.DataFrame({"f1": range(n)})
        # 15 wins, 5 losses
        y_val = pd.Series([1] * 15 + [0] * 5, index=range(n))
        prices_val = pd.DataFrame(
            {"close": [100.0] * n, "atr_14": [5.0] * n}, index=range(n)
        )

        score, metrics = fitness_score(
            model, X_val, y_val, prices_val,
            tp_val=2.0, sl_val=1.0, fee_rate=0.001, slippage=0.0005, threshold=0.5,
            swing_period=5,
        )

        # With swing_period=5 and n=20, sequential simulation yields 4 trades
        # (bars 0, 5, 10, 15).  All are wins (y_val[0..14]==1), so PF > 1.
        assert score > 0
        assert metrics["profit_factor"] > 1.0
        assert metrics["trade_count"] == 4

    def test_pf_below_one_returns_sentinel(self) -> None:
        """PF ≤ 1.0 returns the sentinel -999.0 score."""
        n = 20
        model = self._make_model(n)
        X_val = pd.DataFrame({"f1": range(n)})
        # 4 wins, 16 losses → tiny tp=0.5 vs large sl=2.0 guarantees PF < 1
        y_val = pd.Series([1] * 4 + [0] * 16, index=range(n))
        prices_val = pd.DataFrame(
            {"close": [100.0] * n, "atr_14": [1.0] * n}, index=range(n)
        )

        score, metrics = fitness_score(
            model, X_val, y_val, prices_val,
            tp_val=0.5, sl_val=2.0, fee_rate=0.001, slippage=0.0005, threshold=0.5,
            swing_period=5,
        )

        assert score == -999.0
        assert metrics["profit_factor"] <= 1.0

    def test_linear_frequency_penalty(self) -> None:
        """Frequency penalty is a linear ramp: fewer executable trades → lower score."""
        # We test the penalty in isolation by calling fitness_score twice with
        # identical inputs but different trade_count outcomes, achieved by
        # varying swing_period on a dataset where every bar is a win so PF and
        # MDD stay identical regardless of how many bars actually execute.
        # All-wins: PF = gross_profit / 1e-9 (proportional to trade_count).
        # We therefore verify the *direction* (score grows with trade_count)
        # and that a sub-threshold trade_count produces a strictly lower score.

        n = 50
        model = self._make_model(n)
        X_val = pd.DataFrame({"f1": range(n)})
        y_val = pd.Series([1] * n, index=range(n))          # all wins
        prices_val = pd.DataFrame(
            {"close": [100.0] * n, "atr_14": [1.0] * n}, index=range(n)
        )

        # swing_period=5  → 10 executable trades,
        # target_min = max(5, 50//(5+15)) = 5, penalty = min(1.0, 10/5) = 1.0
        score_full, metrics_full = fitness_score(
            model, X_val, y_val, prices_val,
            tp_val=2.0, sl_val=1.0, fee_rate=0.001, slippage=0.0005, threshold=0.5,
            swing_period=5,
        )

        # swing_period=24 → 3 executable trades (bars 0, 24, 48),
        # target_min = max(5, 50//(24+15)) = max(5,1) = 5, penalty = min(1.0, 3/5) = 0.6
        score_partial, metrics_partial = fitness_score(
            model, X_val, y_val, prices_val,
            tp_val=2.0, sl_val=1.0, fee_rate=0.001, slippage=0.0005, threshold=0.5,
            swing_period=24,
        )

        assert metrics_full["trade_count"] == 10
        assert metrics_partial["trade_count"] == 3
        # Both PF >> 1 (all wins), so neither returns -999 sentinel
        assert score_full > 0
        assert score_partial > 0
        # Fully-penalised score must exceed partially-penalised one
        assert score_full > score_partial


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
        assert "Pure Momentum + Sentiment" not in strategies

    def test_with_sentiment_adds_fng_columns(self) -> None:
        """With sentiment, '+Sentiment' variants including FNG are created."""
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

        assert "Pure Momentum + Sentiment" in strategies
        sentiment_features = strategies["Pure Momentum + Sentiment"]
        for col in SENTIMENT_COLS:
            assert col in sentiment_features


class TestTrainAndEvaluate:
    def test_returns_correct_tuple_structure(
        self, ohlcv_df_with_technicals: pd.DataFrame
    ) -> None:
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
        prices_val = df[["close", "atr_14"]].iloc[split:val_split]
        prices_test = df[["close", "atr_14"]].iloc[val_split:]

        with patch(
            "src.utils.helpers.get_trading_settings",
            return_value={"fee_rate": 0.001, "slippage": 0.0005},
        ):
            result = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test,
                tp_val=2.0, sl_val=1.0, prices_val=prices_val,
                prices_test=prices_test, hyperparams=SAMPLE_HIPERPARAMS
            )

        assert len(result) == 5
        model, metrics, preds_test, buy_dates, threshold = result
        assert hasattr(model, "predict_proba")
        assert isinstance(metrics, dict)
        assert "net_profit_pct" in metrics
        assert "val_fitness_score" in metrics
        assert "test_fitness_score" in metrics
        assert "test_profit_factor" in metrics
        assert "test_max_drawdown" in metrics
        assert "test_trade_count" in metrics
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
        prices_val = df[["close", "atr_14"]].iloc[split:val_split]
        prices_test = df[["close", "atr_14"]].iloc[val_split:]

        with patch(
            "src.utils.helpers.get_trading_settings",
            return_value={"fee_rate": 0.001, "slippage": 0.0005},
        ):
            result = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test,
                tp_val=2.0, sl_val=1.0, prices_val=prices_val,
                prices_test=prices_test, hyperparams=SAMPLE_HIPERPARAMS
            )
        assert result is not None

    def test_train_and_evaluate_val_only_returns_expected_tuple(
        self, ohlcv_df_with_technicals: pd.DataFrame
    ) -> None:
        """train_and_evaluate_val_only returns (model, metrics, threshold)."""
        from src.utils.helpers import train_and_evaluate_val_only
        df = ohlcv_df_with_technicals.copy()
        df["target"] = np.random.randint(0, 2, len(df))
        df.dropna(inplace=True)
        features = ["rsi_14", "macd"]
        n = len(df)
        split = int(n * 0.7)

        X_train = df[features].iloc[:split]
        X_val = df[features].iloc[split:]
        y_train = df["target"].iloc[:split]
        y_val = df["target"].iloc[split:]
        prices_val = df[["close", "atr_14"]].iloc[split:]

        with patch(
            "src.utils.helpers.get_trading_settings",
            return_value={"fee_rate": 0.001, "slippage": 0.0005},
        ):
            result = train_and_evaluate_val_only(
                X_train, X_val, y_train, y_val,
                tp_val=2.0, sl_val=1.0, prices_val=prices_val,
                hyperparams=SAMPLE_HIPERPARAMS, swing_period=5
            )
        
        assert len(result) == 3
        model, metrics, threshold = result
        assert hasattr(model, "predict_proba")
        assert isinstance(metrics, dict)
        assert isinstance(threshold, float)

    @patch("src.utils.helpers._train_core")
    def test_train_and_evaluate_with_valid_threshold_calculates_test_metrics(
        self, mock_train_core, ohlcv_df_with_technicals: pd.DataFrame
    ) -> None:
        """When a valid threshold is found, test metrics are calculated correctly."""
        df = ohlcv_df_with_technicals.copy()
        df["target"] = np.random.randint(0, 2, len(df))
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
        prices_val = df[["close", "atr_14"]].iloc[split:val_split]
        prices_test = df[["close", "atr_14"]].iloc[val_split:]

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.array([[0.1, 0.9]] * len(X))
        val_metrics = {"val_profit_factor": 2.0}
        
        # _train_core returns (model, best_threshold, val_metrics)
        mock_train_core.return_value = (mock_model, 0.5, val_metrics)

        with patch(
            "src.utils.helpers.get_trading_settings",
            return_value={"fee_rate": 0.001, "slippage": 0.0005},
        ):
            result = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test,
                tp_val=2.0, sl_val=1.0, prices_val=prices_val,
                prices_test=prices_test, hyperparams=SAMPLE_HIPERPARAMS
            )

        model, metrics, preds_test, buy_dates, threshold = result
        assert threshold == 0.5
        assert metrics["val_profit_factor"] == 2.0
        assert "test_profit_factor" in metrics
