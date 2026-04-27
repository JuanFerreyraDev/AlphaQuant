"""Tests for src.api.binance.binance_executor — BinanceExecutor class."""

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from binance.exceptions import BinanceAPIException

from src.api.binance.binance_executor import BinanceExecutor


def _make_api_exception(code: int, message: str) -> BinanceAPIException:
    """Build a BinanceAPIException with code and message."""
    exc = BinanceAPIException.__new__(BinanceAPIException)
    exc.code = code
    exc.message = message
    exc.status_code = 400
    return exc


def _build_executor(
    api_key: str = "test_key",
    api_secret: str = "test_secret",
    use_testnet: str = "True",
    leverage: int = 5,
    risk_per_trade_pct: float = 1.0,
) -> BinanceExecutor:
    """Build a BinanceExecutor with injected mocks."""
    env_vars = {
        "BINANCE_API_KEY": api_key,
        "BINANCE_API_SECRET": api_secret,
        "USE_TESTNET": use_testnet,
    }
    with patch("src.api.binance.binance_executor.load_dotenv"), \
         patch("src.api.binance.binance_executor.os.getenv", side_effect=lambda k, d="": env_vars.get(k, d)), \
         patch("src.api.binance.binance_executor.Client") as mock_client_cls, \
         patch("src.api.binance.binance_executor.get_market_config", return_value={"default_leverage": leverage}), \
         patch("src.api.binance.binance_executor.load_settings", return_value={"global": {"risk_per_trade_pct": risk_per_trade_pct}}):
        mock_client_cls.return_value = MagicMock()
        executor = BinanceExecutor()
    return executor


class TestBinanceExecutorInit:
    def test_raises_value_error_when_api_keys_missing(self) -> None:
        """Missing API keys raise ValueError."""
        with patch("src.api.binance.binance_executor.load_dotenv"), \
             patch("src.api.binance.binance_executor.os.getenv", return_value=""), \
             pytest.raises(ValueError, match="BINANCE_API_KEY"):
            BinanceExecutor()

    def test_connects_to_testnet_by_default(self) -> None:
        """USE_TESTNET='True' initializes Client with testnet=True."""
        env_vars = {"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s", "USE_TESTNET": "True"}

        with patch("src.api.binance.binance_executor.load_dotenv"), \
             patch("src.api.binance.binance_executor.os.getenv", side_effect=lambda k, d="": env_vars.get(k, d)), \
             patch("src.api.binance.binance_executor.Client") as mock_cls, \
             patch("src.api.binance.binance_executor.get_market_config", return_value={"default_leverage": 5}):
            mock_cls.return_value = MagicMock()
            executor = BinanceExecutor()

        mock_cls.assert_called_once_with("k", "s", testnet=True)

    def test_connects_to_mainnet_when_testnet_disabled(self) -> None:
        """USE_TESTNET='false' initializes Client with testnet=False."""
        env_vars = {"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s", "USE_TESTNET": "false"}

        with patch("src.api.binance.binance_executor.load_dotenv"), \
             patch("src.api.binance.binance_executor.os.getenv", side_effect=lambda k, d="": env_vars.get(k, d)), \
             patch("src.api.binance.binance_executor.Client") as mock_cls, \
             patch("src.api.binance.binance_executor.get_market_config", return_value={"default_leverage": 3}):
            mock_cls.return_value = MagicMock()
            BinanceExecutor()

        mock_cls.assert_called_once_with("k", "s", testnet=False)

    def test_reads_leverage_from_settings(self) -> None:
        """Leverage is read from get_market_config."""
        executor = _build_executor(leverage=10)

        assert executor.leverage == 10


class TestHasOpenPosition:
    def test_returns_false_when_no_positions(self) -> None:
        """No open position returns False."""
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"positionAmt": "0"}
        ]

        result = executor._has_open_position("BTCUSDT")

        assert result is False

    def test_returns_true_when_position_exists(self) -> None:
        """Open position returns True."""
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"positionAmt": "0.05"}
        ]

        result = executor._has_open_position("BTCUSDT")

        assert result is True

    def test_returns_true_on_api_error_as_failsafe(self) -> None:
        """API error returns True (do not trade when in doubt)."""
        executor = _build_executor()
        executor.client.futures_position_information.side_effect = _make_api_exception(
            -1, "internal error"
        )

        result = executor._has_open_position("BTCUSDT")

        assert result is True


class TestConfigureSymbol:
    def test_sets_isolated_margin_and_leverage(self) -> None:
        """Correctly configures margin type and leverage."""
        executor = _build_executor()

        result = executor._configure_symbol("BTCUSDT")
        assert result is True
        executor.client.futures_change_margin_type.assert_called_once_with(
            symbol="BTCUSDT", marginType="ISOLATED"
        )
        executor.client.futures_change_leverage.assert_called_once()

    def test_tolerates_already_isolated_error_code_minus_4046(self) -> None:
        """Error -4046 (already ISOLATED) is tolerated."""
        executor = _build_executor()
        executor.client.futures_change_margin_type.side_effect = _make_api_exception(
            -4046, "No need to change margin type."
        )

        result = executor._configure_symbol("BTCUSDT")

        assert result is True

    def test_returns_false_on_margin_type_api_error(self) -> None:
        """API error (not -4046) when changing margin returns False."""
        executor = _build_executor()
        executor.client.futures_change_margin_type.side_effect = _make_api_exception(
            -1000, "unknown error"
        )

        result = executor._configure_symbol("BTCUSDT")

        assert result is False

    def test_returns_false_on_leverage_api_error(self) -> None:
        """API error when changing leverage returns False."""
        executor = _build_executor()
        executor.client.futures_change_leverage.side_effect = _make_api_exception(
            -1, "leverage error"
        )

        result = executor._configure_symbol("BTCUSDT")

        assert result is False


class TestGetFuturesBalance:
    def test_returns_available_usdt(self) -> None:
        """Returns the available USDT balance."""
        executor = _build_executor()
        executor.client.futures_account_balance.return_value = [
            {"asset": "BTC", "availableBalance": "0.5"},
            {"asset": "USDT", "availableBalance": "1234.56"},
        ]

        result = executor._get_futures_balance()

        assert result == 1234.56

    def test_returns_zero_when_no_usdt_asset(self) -> None:
        """If no USDT asset exists in balance, returns 0.0."""
        executor = _build_executor()
        executor.client.futures_account_balance.return_value = [
            {"asset": "BTC", "availableBalance": "0.5"},
        ]

        result = executor._get_futures_balance()

        assert result == 0.0

    def test_returns_zero_on_api_error(self) -> None:
        """API error returns 0.0."""
        executor = _build_executor()
        executor.client.futures_account_balance.side_effect = _make_api_exception(
            -1, "error"
        )

        result = executor._get_futures_balance()

        assert result == 0.0


class TestRoundStepSize:
    def test_rounds_down_correctly(self) -> None:
        """Rounds down correctly respecting step_size."""
        executor = _build_executor()

        result = executor._round_step_size(1.567, "0.01")

        assert result == 1.56

    def test_with_small_step(self) -> None:
        """Small step size (0.001) rounds correctly."""
        executor = _build_executor()

        result = executor._round_step_size(0.12345, "0.001")

        assert result == 0.123


class TestRoundTickSize:
    def test_returns_string_rounded_down(self) -> None:
        """Returns string with price rounded down."""
        executor = _build_executor()

        result = executor._round_tick_size(65432.789, "0.01")
        assert result == "65432.78"
        assert isinstance(result, str)


class TestCalculateQuantity:
    def test_with_valid_balance_and_price(self) -> None:
        """Calculates correct quantity with valid balance and price."""
        executor = _build_executor(leverage=5)
        executor.client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "10000.0"},
        ]
        executor.client.futures_symbol_ticker.return_value = {"price": "50000.0"}
        executor.client.futures_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    ],
                }
            ]
        }

        result = executor._calculate_quantity("BTCUSDT")

        # 10000 * 0.01 = 100 (margin), 100 * 5 = 500 (notional), 500/50000 = 0.01
        assert result == 0.01

    def test_returns_none_on_zero_balance(self) -> None:
        """Zero balance returns None."""
        executor = _build_executor()
        executor.client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "0.0"},
        ]

        result = executor._calculate_quantity("BTCUSDT")

        assert result is None

    def test_returns_none_on_zero_price(self) -> None:
        """Zero price returns None."""
        executor = _build_executor()
        executor.client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "10000.0"},
        ]
        executor.client.futures_symbol_ticker.return_value = {"price": "0.0"}

        result = executor._calculate_quantity("BTCUSDT")

        assert result is None

    def test_returns_none_when_below_min_qty(self) -> None:
        """Quantity below minQty returns None."""
        executor = _build_executor(leverage=1)
        executor.client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "100.0"},
        ]
        executor.client.futures_symbol_ticker.return_value = {"price": "50000.0"}
        executor.client.futures_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "1.0"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    ],
                }
            ]
        }

        result = executor._calculate_quantity("BTCUSDT")

        assert result is None

    def test_returns_none_when_below_min_notional(self) -> None:
        """Actual notional below MIN_NOTIONAL returns None."""
        executor = _build_executor(leverage=1)
        executor.client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "100.0"},
        ]
        executor.client.futures_symbol_ticker.return_value = {"price": "0.01"}
        executor.client.futures_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
                        {"filterType": "MIN_NOTIONAL", "notional": "100.0"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    ],
                }
            ]
        }

        result = executor._calculate_quantity("BTCUSDT")

        assert result is None


class TestPlaceAlgoOrder:
    def test_stop_market_success(self) -> None:
        """Places STOP_MARKET correctly."""
        executor = _build_executor()
        executor.client._request_futures_api = MagicMock(
            return_value={"orderId": "12345"}
        )

        result = executor._place_algo_order("BTCUSDT", "SELL", "STOP", "59000.00")

        assert result is not None
        assert result["orderId"] == "12345"

    def test_take_profit_market_success(self) -> None:
        """Places TAKE_PROFIT_MARKET correctly."""
        executor = _build_executor()
        executor.client._request_futures_api = MagicMock(
            return_value={"clientAlgoId": "abc"}
        )
        result = executor._place_algo_order("BTCUSDT", "SELL", "TAKE_PROFIT", "65000.00")

        assert result is not None

    def test_returns_none_on_invalid_order_type(self) -> None:
        """Invalid order type returns None."""
        executor = _build_executor()

        result = executor._place_algo_order("BTCUSDT", "SELL", "INVALID", "60000.00")

        assert result is None

    def test_returns_none_on_api_error(self) -> None:
        """API error returns None."""
        executor = _build_executor()
        executor.client._request_futures_api = MagicMock(
            side_effect=_make_api_exception(-1, "algo order error")
        )

        result = executor._place_algo_order("BTCUSDT", "SELL", "STOP", "59000.00")

        assert result is None


class TestExecuteFuturesTrade:
    def test_full_success_flow(self) -> None:
        """Successful complete trade returns dict with entry, stop_loss, take_profit."""
        executor = _build_executor()
        executor._has_open_position = MagicMock(return_value=False)
        executor._configure_symbol = MagicMock(return_value=True)
        executor._calculate_quantity = MagicMock(return_value=0.01)
        executor._get_symbol_filters = MagicMock(return_value={
            "step_size": "0.001", "min_qty": "0.001",
            "min_notional": 5.0, "tick_size": "0.01",
        })
        executor.client.futures_create_order.return_value = {"orderId": "E1"}
        executor._place_algo_order = MagicMock(
            side_effect=[{"orderId": "SL1"}, {"orderId": "TP1"}]
        )

        with patch("src.api.binance.binance_executor.time.sleep"):
            result = executor.execute_futures_trade("BTC_USDT", "BUY", 58000.0, 65000.0)

        assert result is not None
        assert "entry" in result
        assert "stop_loss" in result
        assert "take_profit" in result

    def test_returns_none_when_position_already_open(self) -> None:
        """Already open position returns None."""
        executor = _build_executor()
        executor._has_open_position = MagicMock(return_value=True)

        result = executor.execute_futures_trade("BTC_USDT", "BUY", 58000.0, 65000.0)

        assert result is None

    def test_returns_none_when_configure_fails(self) -> None:
        """Configuration failure returns None."""
        executor = _build_executor()
        executor._has_open_position = MagicMock(return_value=False)
        executor._configure_symbol = MagicMock(return_value=False)

        result = executor.execute_futures_trade("BTC_USDT", "BUY", 58000.0, 65000.0)

        assert result is None

    def test_returns_none_when_quantity_is_none(self) -> None:
        """None quantity (insufficient balance) returns None."""
        executor = _build_executor()
        executor._has_open_position = MagicMock(return_value=False)
        executor._configure_symbol = MagicMock(return_value=True)
        executor._calculate_quantity = MagicMock(return_value=None)

        result = executor.execute_futures_trade("BTC_USDT", "BUY", 58000.0, 65000.0)

        assert result is None

    def test_returns_none_on_market_order_api_error(self) -> None:
        """API error on MARKET order returns None."""
        executor = _build_executor()
        executor._has_open_position = MagicMock(return_value=False)
        executor._configure_symbol = MagicMock(return_value=True)
        executor._calculate_quantity = MagicMock(return_value=0.01)
        executor._get_symbol_filters = MagicMock(return_value={
            "step_size": "0.001", "min_qty": "0.001",
            "min_notional": 5.0, "tick_size": "0.01",
        })
        executor.client.futures_create_order.side_effect = _make_api_exception(
            -2010, "insufficient balance"
        )

        result = executor.execute_futures_trade("BTC_USDT", "BUY", 58000.0, 65000.0)

        assert result is None

    def test_normalizes_symbol_format(self) -> None:
        """Normalizes symbols with /, _, : to separator-free format."""
        executor = _build_executor()
        executor._has_open_position = MagicMock(return_value=True)

        executor.execute_futures_trade("BTC/USDT:USDT", "BUY", 58000.0, 65000.0)

        executor._has_open_position.assert_called_once_with("BTCUSDT")


class TestGetFuturesBalancePublic:
    """Tests for the public ``get_futures_balance()`` wrapper."""

    def test_delegates_to_private_method(self) -> None:
        """``get_futures_balance()`` returns the value from ``_get_futures_balance()``."""
        executor = _build_executor()
        executor.client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "999.99"},
        ]

        result = executor.get_futures_balance()

        assert result == 999.99


class TestGetOpenPositions:
    """Tests for ``get_open_positions()``.

    The method must call ``futures_position_information`` without filtering by
    symbol and return **only** positions whose ``positionAmt`` is not zero.
    """

    def test_filters_only_nonzero_positions(self) -> None:
        """Given a mixed list of positions (zero, positive, negative),
        only those with ``positionAmt != 0.0`` are returned.
        """
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT",  "positionAmt": "0.000"},
            {"symbol": "ETHUSDT",  "positionAmt": "1.5"},
            {"symbol": "BNBUSDT",  "positionAmt": "0.0"},
            {"symbol": "SOLUSDT",  "positionAmt": "-0.5"},
            {"symbol": "DOGEUSDT", "positionAmt": "0"},
        ]

        result = executor.get_open_positions()

        assert len(result) == 2
        symbols = [p["symbol"] for p in result]
        assert "ETHUSDT" in symbols
        assert "SOLUSDT" in symbols

    def test_returns_empty_list_when_no_open_positions(self) -> None:
        """If all positions have ``positionAmt`` of zero, an empty list is returned."""
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.000"},
            {"symbol": "ETHUSDT", "positionAmt": "0"},
        ]

        result = executor.get_open_positions()

        assert result == []

    def test_returns_empty_list_when_no_positions_at_all(self) -> None:
        """If the API returns an empty list, the method returns an empty list."""
        executor = _build_executor()
        executor.client.futures_position_information.return_value = []

        result = executor.get_open_positions()

        assert result == []


class TestCloseAllPositions:
    """Tests for ``close_all_positions()``.

    The method iterates over open positions, sends inverse MARKET orders to
    close each one, and then cancels all open orders for the affected symbols.
    It captures ``BinanceAPIException`` per symbol so a single failure does
    not abort the loop.
    """

    def test_happy_path_long_and_short(self) -> None:
        """Verify that a Long position triggers a SELL close order and a Short
        position triggers a BUY close order.  Also verify that
        ``futures_cancel_all_open_orders`` is called for each affected symbol.
        """
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "1.5"},
            {"symbol": "ETHUSDT", "positionAmt": "-0.5"},
            {"symbol": "BNBUSDT", "positionAmt": "0.0"},
        ]
        executor.client.futures_create_order.return_value = {"orderId": "OK"}

        closed = executor.close_all_positions()

        assert closed == 2

        calls = executor.client.futures_create_order.call_args_list
        assert len(calls) == 2

        _, kwargs_btc = calls[0]
        assert kwargs_btc["symbol"] == "BTCUSDT"
        assert kwargs_btc["side"] == "SELL"
        assert kwargs_btc["type"] == "MARKET"
        assert kwargs_btc["quantity"] == 1.5

        _, kwargs_eth = calls[1]
        assert kwargs_eth["symbol"] == "ETHUSDT"
        assert kwargs_eth["side"] == "BUY"
        assert kwargs_eth["type"] == "MARKET"
        assert kwargs_eth["quantity"] == 0.5

        cancel_calls = executor.client.futures_cancel_all_open_orders.call_args_list
        cancelled_symbols = {c.kwargs["symbol"] for c in cancel_calls}
        assert cancelled_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_with_exception_continues_and_cancels_both(self) -> None:
        """If ``futures_create_order`` raises ``BinanceAPIException`` for the
        first symbol, the loop must continue, successfully close the second
        symbol, and still attempt to cancel open orders for **both** symbols.
        """
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "2.0"},
            {"symbol": "ETHUSDT", "positionAmt": "-1.0"},
        ]

        executor.client.futures_create_order.side_effect = [
            _make_api_exception(-2010, "Insufficient margin"),
            {"orderId": "OK"},
        ]

        closed = executor.close_all_positions()

        assert closed == 1
        assert executor.client.futures_create_order.call_count == 2

        cancel_calls = executor.client.futures_cancel_all_open_orders.call_args_list
        cancelled_symbols = {c.kwargs["symbol"] for c in cancel_calls}
        assert cancelled_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_returns_zero_when_no_open_positions(self) -> None:
        """If there are no open positions, ``close_all_positions`` returns 0
        and no orders or cancellations are attempted.
        """
        executor = _build_executor()
        executor.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.0"},
        ]

        closed = executor.close_all_positions()

        assert closed == 0
        executor.client.futures_create_order.assert_not_called()
        executor.client.futures_cancel_all_open_orders.assert_not_called()
