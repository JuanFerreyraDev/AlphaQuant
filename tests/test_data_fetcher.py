"""Tests for src.brain.data_fetcher"""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import requests

from src.brain.data_fetcher import (
    fetch_historical_data,
    fetch_ohlcv_binance,
    get_fear_and_greed,
)

SAMPLE_FNG_RESPONSE = {
    "data": [
        {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1711843200"},
        {"value": "30", "value_classification": "Fear", "timestamp": "1711756800"},
    ]
}


class TestGetFearAndGreed:
    @patch("src.brain.data_fetcher.requests.get")
    def test_returns_dataframe_on_success(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_FNG_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_fear_and_greed()
        assert isinstance(result, pd.DataFrame)
        assert "fng_value" in result.columns
        assert len(result) == 2

    @patch("src.brain.data_fetcher.requests.get")
    def test_returns_empty_on_empty_data(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_fear_and_greed()
        assert result.empty

    @patch("src.brain.data_fetcher.requests.get")
    def test_returns_empty_on_timeout(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.Timeout("timeout")

        result = get_fear_and_greed()
        assert result.empty

    @patch("src.brain.data_fetcher.requests.get")
    def test_returns_empty_on_connection_error(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.ConnectionError("connection refused")

        result = get_fear_and_greed()
        assert result.empty


class TestGetFearAndGreedEdgeCases:
    """Edge cases for get_fear_and_greed()."""

    @patch("src.brain.data_fetcher.requests.get")
    def test_returns_empty_on_http_error(self, mock_get: MagicMock) -> None:
        """HTTPError returns an empty DataFrame without retries."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        result = get_fear_and_greed()

        assert result.empty

    @patch("src.brain.data_fetcher.requests.get")
    def test_returns_empty_on_invalid_json(self, mock_get: MagicMock) -> None:
        """ValueError when parsing JSON returns an empty DataFrame."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_get.return_value = mock_response

        result = get_fear_and_greed()

        assert result.empty

    @patch("src.brain.data_fetcher.time.sleep", return_value=None)
    @patch("src.brain.data_fetcher.requests.get")
    def test_retries_on_connection_error_up_to_max(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Retries 3 times on ConnectionError then returns empty."""
        mock_get.side_effect = requests.ConnectionError("refused")

        result = get_fear_and_greed()

        assert result.empty
        assert mock_get.call_count == 3


class TestFetchHistoricalData:
    @patch("src.brain.data_fetcher.get_project_root")
    @patch("src.brain.data_fetcher.get_active_market", return_value="futures")
    @patch("src.brain.data_fetcher.ccxt.binanceusdm")
    def test_returns_dataframe_on_success(
        self,
        mock_exchange_cls: MagicMock,
        _mock_market: MagicMock,
        mock_root: MagicMock,
        tmp_path,
    ) -> None:
        """Successful download returns a DataFrame with OHLCV columns."""
        mock_root.return_value = tmp_path
        (tmp_path / "data" / "raw_csv").mkdir(parents=True)

        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        mock_exchange.parse8601.return_value = 1577836800000
        candles = [
            [1577836800000, 100.0, 105.0, 95.0, 102.0, 5000.0],
            [1577923200000, 102.0, 108.0, 99.0, 106.0, 6000.0],
        ]
        mock_exchange.fetch_ohlcv.return_value = candles

        result = fetch_historical_data("BTC_USDT", "1d", "2020-01-01T00:00:00Z")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "close" in result.columns

    @patch("src.brain.data_fetcher.time.sleep", return_value=None)
    @patch("src.brain.data_fetcher.get_active_market", return_value="futures")
    @patch("src.brain.data_fetcher.ccxt.binanceusdm")
    def test_raises_on_exchange_error(
        self,
        mock_exchange_cls: MagicMock,
        _mock_market: MagicMock,
        _mock_sleep: MagicMock,
    ) -> None:
        """ExchangeError raises RuntimeError immediately."""
        import ccxt

        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        mock_exchange.parse8601.return_value = 1577836800000
        mock_exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("symbol not found")

        with pytest.raises(RuntimeError, match="Exchange error downloading"):
            fetch_historical_data("INVALID_USDT")

    @patch("src.brain.data_fetcher.get_project_root")
    @patch("src.brain.data_fetcher.get_active_market", return_value="spot")
    @patch("src.brain.data_fetcher.ccxt.binance")
    def test_handles_empty_candles(
        self,
        mock_exchange_cls: MagicMock,
        _mock_market: MagicMock,
        mock_root: MagicMock,
        tmp_path,
    ) -> None:
        """Empty candle list produces an empty DataFrame (not None)."""
        mock_root.return_value = tmp_path
        (tmp_path / "data" / "raw_csv").mkdir(parents=True)

        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        mock_exchange.parse8601.return_value = 1577836800000
        mock_exchange.fetch_ohlcv.return_value = []

        result = fetch_historical_data("BTC_USDT", "1d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.brain.data_fetcher.time.sleep", return_value=None)
    @patch("src.brain.data_fetcher.get_active_market", return_value="futures")
    @patch("src.brain.data_fetcher.ccxt.binanceusdm")
    def test_raises_on_network_error_exhaustion(
        self,
        mock_exchange_cls: MagicMock,
        _mock_market: MagicMock,
        _mock_sleep: MagicMock,
    ) -> None:
        """Retries on NetworkError and raises RuntimeError after exhausting attempts."""
        import ccxt

        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        mock_exchange.parse8601.return_value = 1577836800000
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")

        with pytest.raises(RuntimeError, match="Download of .* failed after"):
            fetch_historical_data("BTC_USDT")
        assert mock_exchange.fetch_ohlcv.call_count == 3


class TestFetchOhlcvBinance:
    @pytest.mark.asyncio
    @patch("src.brain.data_fetcher.ccxt_async.binanceusdm")
    async def test_returns_dataframe_on_success(
        self, mock_exchange_cls: MagicMock
    ) -> None:
        """Successful async download returns a DataFrame."""
        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        candles = [
            [1577836800000, 100.0, 105.0, 95.0, 102.0, 5000.0],
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=candles)
        mock_exchange.close = AsyncMock()

        result = await fetch_ohlcv_binance("BTC_USDT", "1d", limit=1)

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        mock_exchange.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.brain.data_fetcher.ccxt_async.binanceusdm")
    async def test_returns_none_on_exchange_error(
        self, mock_exchange_cls: MagicMock
    ) -> None:
        """ExchangeError returns None."""
        import ccxt

        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=ccxt.ExchangeError("err"))
        mock_exchange.close = AsyncMock()

        result = await fetch_ohlcv_binance("BTC_USDT")

        assert result is None
        mock_exchange.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.brain.data_fetcher.ccxt_async.binanceusdm")
    async def test_retries_on_network_error(
        self, mock_exchange_cls: MagicMock
    ) -> None:
        """Retries on NetworkError and returns None after exhausting attempts."""
        import ccxt

        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=ccxt.NetworkError("timeout"))
        mock_exchange.close = AsyncMock()

        result = await fetch_ohlcv_binance("BTC_USDT")

        assert result is None
        assert mock_exchange.fetch_ohlcv.await_count == 3

    @pytest.mark.asyncio
    async def test_reuses_injected_exchange_without_closing(self) -> None:
        """When an exchange is injected, it is used and NOT closed by the function."""
        candles = [[1577836800000, 100.0, 105.0, 95.0, 102.0, 5000.0]]
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=candles)
        mock_exchange.close = AsyncMock()

        result = await fetch_ohlcv_binance("BTC_USDT", "1d", limit=1, exchange=mock_exchange)

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        mock_exchange.fetch_ohlcv.assert_awaited_once()
        mock_exchange.close.assert_not_awaited()
