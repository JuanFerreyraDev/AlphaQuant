"""Tests for src.engine.tasks — Daily market evaluation orchestrator."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.engine.tasks import (
    TRAINING_COOLDOWN_DAYS,
    _check_training_freshness,
    _evaluate_model,
    _execute_and_notify,
    _init_executor,
    _sanitize_symbol,
    daily_market_evaluation,
    run_full_training_pipeline,
)


class TestSanitizeSymbol:
    def test_converts_slash_format(self) -> None:
        assert _sanitize_symbol("BTC/USDT") == "BTC_USDT"

    def test_converts_colon_format(self) -> None:
        assert _sanitize_symbol("BTC/USDT:USDT") == "BTC_USDT"

    def test_preserves_underscore_format(self) -> None:
        assert _sanitize_symbol("BTC_USDT") == "BTC_USDT"

    def test_handles_complex_format(self) -> None:
        assert _sanitize_symbol("ETH/USDT:USDT") == "ETH_USDT"


class TestInitExecutor:
    @patch("src.engine.tasks.BinanceExecutor")
    def test_returns_instance_on_success(self, mock_cls: MagicMock) -> None:
        """Successful initialization returns an instance."""
        mock_cls.return_value = MagicMock()

        result = _init_executor()

        assert result is not None

    @patch("src.engine.tasks.BinanceExecutor", side_effect=ValueError("missing keys"))
    def test_returns_none_on_value_error(self, _mock: MagicMock) -> None:
        """ValueError returns None (signal-only mode)."""
        result = _init_executor()
        assert result is None

    @patch("src.engine.tasks.BinanceExecutor", side_effect=ConnectionError("refused"))
    def test_returns_none_on_connection_error(self, _mock: MagicMock) -> None:
        """ConnectionError returns None."""
        result = _init_executor()
        assert result is None


class TestDailyMarketEvaluation:
    @pytest.mark.asyncio
    @patch("src.engine.tasks.get_active_symbols", return_value=[])
    @patch("src.engine.tasks.get_active_market", return_value="futures")
    async def test_returns_zero_when_no_symbols(
        self, _mock_market: MagicMock, _mock_symbols: MagicMock
    ) -> None:
        """No active symbols returns 0 signals."""
        result = await daily_market_evaluation(MagicMock(), 123)

        assert result == 0

    @pytest.mark.asyncio
    @patch("src.engine.tasks.os.path.exists", return_value=False)
    @patch("src.engine.tasks.get_active_symbols", return_value=["BTC_USDT"])
    @patch("src.engine.tasks.get_active_market", return_value="futures")
    async def test_returns_zero_when_no_models_dir(
        self, _m1: MagicMock, _m2: MagicMock, _m3: MagicMock
    ) -> None:
        """No models directory returns 0."""
        result = await daily_market_evaluation(MagicMock(), 123)
        assert result == 0

    @pytest.mark.asyncio
    @patch("src.engine.tasks._evaluate_model", new_callable=AsyncMock, return_value=1)
    @patch(
        "src.engine.tasks.glob.glob", return_value=["/data/models/BTC_USDT/model.pkl"]
    )
    @patch("src.engine.tasks.os.path.isdir", return_value=True)
    @patch("src.engine.tasks.os.path.exists", return_value=True)
    @patch("src.engine.tasks.get_fear_and_greed", return_value=pd.DataFrame())
    @patch("src.engine.tasks._init_executor", return_value=None)
    @patch("src.engine.tasks.get_active_symbols", return_value=["BTC_USDT"])
    @patch("src.engine.tasks.get_active_market", return_value="futures")
    async def test_processes_all_symbols_and_models(self, *mocks: MagicMock) -> None:
        """Processes each model found for every symbol."""
        mock_exchange = MagicMock()
        mock_exchange.close = AsyncMock()
        with (
            patch(
                "src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread,
            patch(
                "src.engine.tasks.ccxt_async.binanceusdm", return_value=mock_exchange
            ),
        ):
            mock_thread.side_effect = [None, pd.DataFrame()]
            result = await daily_market_evaluation(MagicMock(), 123)

        assert result == 1
        mock_exchange.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "src.engine.tasks._evaluate_model",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    )
    @patch(
        "src.engine.tasks.glob.glob", return_value=["/data/models/BTC_USDT/model.pkl"]
    )
    @patch("src.engine.tasks.os.path.isdir", return_value=True)
    @patch("src.engine.tasks.os.path.exists", return_value=True)
    @patch("src.engine.tasks.get_fear_and_greed", return_value=pd.DataFrame())
    @patch("src.engine.tasks._init_executor", return_value=None)
    @patch("src.engine.tasks.get_active_symbols", return_value=["BTC_USDT"])
    @patch("src.engine.tasks.get_active_market", return_value="futures")
    async def test_continues_on_model_error(self, *mocks: MagicMock) -> None:
        """Error in a model does not stop the evaluation."""
        mock_exchange = MagicMock()
        mock_exchange.close = AsyncMock()
        with (
            patch(
                "src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread,
            patch(
                "src.engine.tasks.ccxt_async.binanceusdm", return_value=mock_exchange
            ),
        ):
            mock_thread.side_effect = [None, pd.DataFrame()]

            result = await daily_market_evaluation(MagicMock(), 123)
            assert result == 0
        mock_exchange.close.assert_awaited_once()


def _make_model_dict(proba: float = 0.8, threshold: float = 0.6) -> dict[str, Any]:
    """Create a valid model_dict with configurable probability."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - proba, proba]])
    return {
        "model": mock_model,
        "features": ["rsi_14", "macd"],
        "threshold": threshold,
        "atr_tp_multi": 2.0,
        "atr_sl_multi": 1.0,
        "strategy_name": "TestStrategy",
    }


def _make_ohlcv_df() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for _evaluate_model."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0] * 5,
            "high": [110.0] * 5,
            "low": [90.0] * 5,
            "close": [105.0] * 5,
            "volume": [1000.0] * 5,
            "rsi_14": [55.0] * 5,
            "macd": [0.5] * 5,
            "atr_14": [5.0] * 5,
        }
    )


class TestEvaluateModel:
    @pytest.mark.asyncio
    @patch("src.engine.tasks.joblib.load")
    async def test_returns_zero_when_model_dict_missing_keys(
        self, mock_load: MagicMock
    ) -> None:
        """Model dict missing required keys returns 0."""
        mock_load.return_value = {"model": MagicMock()}

        result = await _evaluate_model(
            MagicMock(),
            123,
            None,
            "futures",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        assert result == 0

    @pytest.mark.asyncio
    @patch("src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock)
    @patch(
        "src.engine.tasks.fetch_ohlcv_binance",
        new_callable=AsyncMock,
        return_value=None,
    )
    @patch("src.engine.tasks.joblib.load")
    async def test_returns_zero_when_ohlcv_data_is_none(
        self, mock_load: MagicMock, _mock_fetch: MagicMock, _mock_thread: MagicMock
    ) -> None:
        """OHLCV data None returns 0."""
        mock_load.return_value = _make_model_dict()

        result = await _evaluate_model(
            MagicMock(),
            123,
            None,
            "futures",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        assert result == 0

    @pytest.mark.asyncio
    @patch("src.engine.tasks.send_trade_signal", new_callable=AsyncMock)
    @patch("src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock)
    @patch("src.engine.tasks.fetch_ohlcv_binance", new_callable=AsyncMock)
    @patch("src.engine.tasks.joblib.load")
    async def test_returns_zero_when_probability_below_threshold(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_thread: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Probability below threshold returns 0, without sending a signal."""
        mock_load.return_value = _make_model_dict(proba=0.3, threshold=0.6)
        mock_fetch.return_value = _make_ohlcv_df()
        mock_thread.return_value = _make_ohlcv_df()

        result = await _evaluate_model(
            MagicMock(),
            123,
            None,
            "futures",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        assert result == 0
        mock_signal.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.engine.tasks.send_trade_signal", new_callable=AsyncMock)
    @patch("src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock)
    @patch("src.engine.tasks.fetch_ohlcv_binance", new_callable=AsyncMock)
    @patch("src.engine.tasks.joblib.load")
    async def test_returns_one_and_sends_signal_when_above_threshold(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_thread: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Probability above threshold sends a signal and returns 1."""
        mock_load.return_value = _make_model_dict(proba=0.8, threshold=0.6)
        df = _make_ohlcv_df()
        mock_fetch.return_value = df
        mock_thread.return_value = df

        result = await _evaluate_model(
            MagicMock(),
            123,
            None,
            "spot",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        assert result == 1
        mock_signal.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.engine.tasks._execute_and_notify", new_callable=AsyncMock)
    @patch("src.engine.tasks.send_trade_signal", new_callable=AsyncMock)
    @patch("src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock)
    @patch("src.engine.tasks.fetch_ohlcv_binance", new_callable=AsyncMock)
    @patch("src.engine.tasks.joblib.load")
    async def test_executes_binance_trade_when_executor_present(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_thread: MagicMock,
        mock_signal: MagicMock,
        mock_exec_notify: MagicMock,
    ) -> None:
        """With executor present and futures market, executes a trade."""
        mock_load.return_value = _make_model_dict(proba=0.9, threshold=0.5)
        df = _make_ohlcv_df()
        mock_fetch.return_value = df
        mock_thread.return_value = df
        mock_executor = MagicMock()

        result = await _evaluate_model(
            MagicMock(),
            123,
            mock_executor,
            "futures",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        assert result == 1
        mock_exec_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.engine.tasks._execute_and_notify", new_callable=AsyncMock)
    @patch("src.engine.tasks.send_trade_signal", new_callable=AsyncMock)
    @patch("src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock)
    @patch("src.engine.tasks.fetch_ohlcv_binance", new_callable=AsyncMock)
    @patch("src.engine.tasks.joblib.load")
    async def test_skips_binance_when_executor_is_none(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_thread: MagicMock,
        mock_signal: MagicMock,
        mock_exec_notify: MagicMock,
    ) -> None:
        """Without executor, does not attempt to execute on Binance."""
        mock_load.return_value = _make_model_dict(proba=0.9, threshold=0.5)
        df = _make_ohlcv_df()
        mock_fetch.return_value = df
        mock_thread.return_value = df

        await _evaluate_model(
            MagicMock(),
            123,
            None,
            "futures",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        mock_exec_notify.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock)
    @patch("src.engine.tasks.fetch_ohlcv_binance", new_callable=AsyncMock)
    @patch("src.engine.tasks.joblib.load")
    async def test_returns_zero_on_missing_features(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_thread: MagicMock,
    ) -> None:
        """Missing features in the last candle returns 0."""
        model_dict = _make_model_dict()
        model_dict["features"] = ["nonexistent_feature_1", "nonexistent_feature_2"]
        mock_load.return_value = model_dict
        df = _make_ohlcv_df()
        mock_fetch.return_value = df
        mock_thread.return_value = df

        result = await _evaluate_model(
            MagicMock(),
            123,
            None,
            "futures",
            "BTC_USDT",
            "/path/model.pkl",
            "model.pkl",
            pd.DataFrame(),
        )

        assert result == 0


class TestExecuteAndNotify:
    @pytest.mark.asyncio
    async def test_sends_success_message_on_trade(
        self, mock_telegram_app: MagicMock
    ) -> None:
        """Successful trade sends a confirmation message."""
        mock_executor = MagicMock()
        mock_executor.execute_futures_trade.return_value = {
            "entry": {"orderId": "E1"},
            "stop_loss": {"clientAlgoId": "SL1"},
            "take_profit": {"clientAlgoId": "TP1"},
        }

        with patch(
            "src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = mock_executor.execute_futures_trade.return_value
            await _execute_and_notify(
                mock_telegram_app, 123, mock_executor, "BTC_USDT", 58000.0, 65000.0
            )

        mock_telegram_app.bot.send_message.assert_awaited_once()
        text = mock_telegram_app.bot.send_message.call_args[1]["text"]
        assert "Order executed" in text

    @pytest.mark.asyncio
    async def test_sends_warning_message_when_trade_returns_none(
        self, mock_telegram_app: MagicMock
    ) -> None:
        """Trade returning None sends a warning message."""
        mock_executor = MagicMock()

        with patch(
            "src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = None
            await _execute_and_notify(
                mock_telegram_app, 123, mock_executor, "BTC_USDT", 58000.0, 65000.0
            )

        mock_telegram_app.bot.send_message.assert_awaited_once()
        text = mock_telegram_app.bot.send_message.call_args[1]["text"]
        assert "Trade no" in text or "not executed" in text.lower()

    @pytest.mark.asyncio
    async def test_sends_alert_on_execution_error(
        self, mock_telegram_app: MagicMock
    ) -> None:
        """Execution error sends an alert."""
        mock_executor = MagicMock()

        with patch(
            "src.engine.tasks.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.side_effect = ConnectionError("network down")
            await _execute_and_notify(
                mock_telegram_app, 123, mock_executor, "BTC_USDT", 58000.0, 65000.0
            )

        mock_telegram_app.bot.send_message.assert_awaited_once()
        text = mock_telegram_app.bot.send_message.call_args[1]["text"]
        assert "ignored" in text


class TestCheckTrainingFreshness:
    @patch("src.engine.tasks.get_project_root")
    def test_needs_training_when_config_missing(
        self,
        mock_root: MagicMock,
        tmp_path,
    ) -> None:
        """Without config.json, indicates training is needed."""
        mock_root.return_value = tmp_path
        needs, reason = _check_training_freshness("BTC_USDT")
        assert needs is True
        assert reason == ""

    @patch("src.engine.tasks.get_project_root")
    def test_needs_training_when_no_last_trained_field(
        self,
        mock_root: MagicMock,
        tmp_path,
    ) -> None:
        """Config without last_trained field indicates training is needed."""
        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        import json

        (config_dir / "config.json").write_text(
            json.dumps({"symbol": "BTC_USDT"}),
            encoding="utf-8",
        )
        needs, reason = _check_training_freshness("BTC_USDT")
        assert needs is True

    @patch("src.engine.tasks.get_project_root")
    def test_skips_when_recently_trained(
        self,
        mock_root: MagicMock,
        tmp_path,
    ) -> None:
        """Trained less than 14 days ago indicates NO training needed."""
        import datetime
        import json

        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        recent = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=5
        )
        (config_dir / "config.json").write_text(
            json.dumps({"last_trained": recent.isoformat()}),
            encoding="utf-8",
        )
        needs, reason = _check_training_freshness("BTC_USDT")
        assert needs is False
        assert "less than" in reason and "day" in reason

    @patch("src.engine.tasks.get_project_root")
    def test_needs_training_when_cooldown_expired(
        self,
        mock_root: MagicMock,
        tmp_path,
    ) -> None:
        """Trained more than 14 days ago indicates training IS needed."""
        import datetime
        import json

        mock_root.return_value = tmp_path
        config_dir = tmp_path / "data" / "models" / "BTC_USDT"
        config_dir.mkdir(parents=True)
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=20)
        (config_dir / "config.json").write_text(
            json.dumps({"last_trained": old.isoformat()}),
            encoding="utf-8",
        )
        needs, reason = _check_training_freshness("BTC_USDT")
        assert needs is True


class TestRunFullTrainingPipeline:
    @patch("src.engine.tasks.train_factory")
    @patch("src.engine.tasks.optimize_strategy")
    @patch("src.engine.tasks.get_fear_and_greed", return_value=pd.DataFrame())
    @patch("src.engine.tasks.fetch_historical_data")
    @patch("src.engine.tasks._check_training_freshness", return_value=(True, ""))
    def test_runs_all_three_steps_in_order(
        self,
        _mock_freshness: MagicMock,
        mock_fetch: MagicMock,
        _mock_fg: MagicMock,
        mock_optimize: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Pipeline executes fetch → optimize → train in order."""
        call_order: list[str] = []
        mock_fetch.side_effect = lambda *a, **kw: call_order.append("fetch")
        mock_optimize.side_effect = lambda *a, **kw: call_order.append("optimize")
        mock_train.side_effect = lambda *a, **kw: call_order.append("train")

        trained, safe_symbol, reason = run_full_training_pipeline("BTC_USDT")

        assert call_order == ["fetch", "optimize", "train"]
        assert trained is True
        assert safe_symbol == "BTC_USDT"
        assert reason == ""

    @patch(
        "src.engine.tasks.fetch_historical_data",
        side_effect=RuntimeError("download failed"),
    )
    @patch("src.engine.tasks._check_training_freshness", return_value=(True, ""))
    def test_propagates_fetch_error(
        self,
        _mock_freshness: MagicMock,
        _mock: MagicMock,
    ) -> None:
        """Error in fetch_historical_data propagates as RuntimeError."""
        with pytest.raises(RuntimeError, match="download failed"):
            run_full_training_pipeline("BTC_USDT")

    @patch(
        "src.engine.tasks.optimize_strategy",
        side_effect=RuntimeError("optimization failed"),
    )
    @patch("src.engine.tasks.get_fear_and_greed", return_value=pd.DataFrame())
    @patch("src.engine.tasks.fetch_historical_data")
    @patch("src.engine.tasks._check_training_freshness", return_value=(True, ""))
    def test_propagates_optimize_error(
        self,
        _mock_freshness: MagicMock,
        _mock_fetch: MagicMock,
        _mock_fg: MagicMock,
        _mock_opt: MagicMock,
    ) -> None:
        """Error in optimize_strategy propagates."""
        with pytest.raises(RuntimeError, match="optimization failed"):
            run_full_training_pipeline("BTC_USDT")

    @patch(
        "src.engine.tasks.train_factory", side_effect=RuntimeError("training failed")
    )
    @patch("src.engine.tasks.optimize_strategy")
    @patch("src.engine.tasks.get_fear_and_greed", return_value=pd.DataFrame())
    @patch("src.engine.tasks.fetch_historical_data")
    @patch("src.engine.tasks._check_training_freshness", return_value=(True, ""))
    def test_propagates_train_error(
        self,
        _mock_freshness: MagicMock,
        _mock_fetch: MagicMock,
        _mock_fg: MagicMock,
        _mock_opt: MagicMock,
        _mock_train: MagicMock,
    ) -> None:
        """Error in train_factory propagates."""
        with pytest.raises(RuntimeError, match="training failed"):
            run_full_training_pipeline("BTC_USDT")

    @patch("src.engine.tasks.train_factory")
    @patch("src.engine.tasks.optimize_strategy")
    @patch("src.engine.tasks.fetch_historical_data")
    @patch(
        "src.engine.tasks._check_training_freshness",
        return_value=(False, "Trained 3 day(s) ago, less than 14 days"),
    )
    def test_skips_when_recently_trained(
        self,
        _mock_freshness: MagicMock,
        mock_fetch: MagicMock,
        mock_optimize: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Pipeline skips all steps when cooldown has not expired."""
        trained, safe_symbol, reason = run_full_training_pipeline("BTC_USDT")

        assert trained is False
        assert safe_symbol == "BTC_USDT"
        assert "less than" in reason
        mock_fetch.assert_not_called()
        mock_optimize.assert_not_called()
        mock_train.assert_not_called()
