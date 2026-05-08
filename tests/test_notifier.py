"""Tests for src.api.telegram.notifier — send_trade_signal, send_execution_result, send_execution_error."""

from unittest.mock import AsyncMock

import pytest
import telegram.error

from src.api.telegram.notifier import (
    send_execution_error,
    send_execution_result,
    send_trade_signal,
)


class TestSendTradeSignal:
    @pytest.mark.asyncio
    async def test_sends_html_formatted_message(self, mock_telegram_app) -> None:
        """Sends message with parse_mode='HTML'."""
        app = mock_telegram_app
        signal = {
            "symbol": "BTC/USDT",
            "strategy": "Momentum",
            "price": 60000.0,
            "tp": 65000.0,
            "sl": 58000.0,
        }

        await send_trade_signal(app, 123, signal)

        app.bot.send_message.assert_awaited_once()
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["parse_mode"] == "HTML"
        assert call_kwargs["chat_id"] == 123
        assert "BTC/USDT" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_escapes_html_in_symbol_and_strategy(self, mock_telegram_app) -> None:
        """HTML characters in symbol/strategy are escaped."""
        app = mock_telegram_app
        signal = {"symbol": "<script>alert(1)</script>", "strategy": "a&b"}

        await send_trade_signal(app, 123, signal)

        text = app.bot.send_message.call_args[1]["text"]
        assert "<script>" not in text
        assert "&lt;script&gt;" in text
        assert "&amp;b" in text

    @pytest.mark.asyncio
    async def test_uses_defaults_for_missing_keys(self, mock_telegram_app) -> None:
        """Missing keys in signal_data use default values."""
        app = mock_telegram_app

        await send_trade_signal(app, 123, {})

        text = app.bot.send_message.call_args[1]["text"]
        assert "UNKNOWN" in text

    @pytest.mark.asyncio
    async def test_handles_telegram_error_gracefully(self, mock_telegram_app) -> None:
        """TelegramError does not propagate."""
        app = mock_telegram_app
        app.bot.send_message = AsyncMock(
            side_effect=telegram.error.TelegramError("blocked")
        )
        signal = {
            "symbol": "BTC/USDT",
            "strategy": "Test",
            "price": 100.0,
            "tp": 110.0,
            "sl": 90.0,
        }

        await send_trade_signal(app, 123, signal)

    @pytest.mark.asyncio
    async def test_handles_network_error_gracefully(self, mock_telegram_app) -> None:
        """NetworkError does not propagate."""
        app = mock_telegram_app
        app.bot.send_message = AsyncMock(
            side_effect=telegram.error.NetworkError("timeout")
        )
        signal = {
            "symbol": "BTC/USDT",
            "strategy": "Test",
            "price": 100.0,
            "tp": 110.0,
            "sl": 90.0,
        }

        await send_trade_signal(app, 123, signal)


class TestSendExecutionResult:
    @pytest.mark.asyncio
    async def test_sends_order_ids_when_result_present(self, mock_telegram_app) -> None:
        """Full result dict → message contains orderId, SL/TP algo IDs, symbol, parse_mode HTML."""
        app = mock_telegram_app
        result = {
            "entry": {"orderId": 111},
            "stop_loss": {"clientAlgoId": "SL-1"},
            "take_profit": {"clientAlgoId": "TP-1"},
        }

        await send_execution_result(app, 123, result, "BTC_USDT")

        app.bot.send_message.assert_awaited_once()
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["parse_mode"] == "HTML"
        assert call_kwargs["chat_id"] == 123
        text = call_kwargs["text"]
        assert "111" in text
        assert "SL-1" in text
        assert "TP-1" in text
        assert "BTC/USDT" in text

    @pytest.mark.asyncio
    async def test_sends_skipped_message_when_result_is_none(
        self, mock_telegram_app
    ) -> None:
        """None result (skipped trade) produces a 'no executed' warning message."""
        app = mock_telegram_app

        await send_execution_result(app, 123, None, "ETH_USDT")

        text = app.bot.send_message.call_args[1]["text"]
        assert "no executed" in text.lower()
        assert "ETH/USDT" in text

    @pytest.mark.asyncio
    async def test_falls_back_to_na_when_order_keys_missing(
        self, mock_telegram_app
    ) -> None:
        """Missing orderId / clientAlgoId keys default to 'N/A'."""
        app = mock_telegram_app
        result = {"entry": {}, "stop_loss": None, "take_profit": None}

        await send_execution_result(app, 123, result, "SOL_USDT")

        text = app.bot.send_message.call_args[1]["text"]
        assert text.count("N/A") >= 2

    @pytest.mark.asyncio
    async def test_escapes_html_in_symbol(self, mock_telegram_app) -> None:
        """Symbol with HTML characters is escaped before sending."""
        app = mock_telegram_app
        result = {"entry": {"orderId": 1}, "stop_loss": None, "take_profit": None}

        await send_execution_result(app, 123, result, "<XSS>_USDT")

        text = app.bot.send_message.call_args[1]["text"]
        assert "<XSS>" not in text

    @pytest.mark.asyncio
    async def test_handles_telegram_error_gracefully(self, mock_telegram_app) -> None:
        """TelegramError does not propagate."""
        app = mock_telegram_app
        app.bot.send_message = AsyncMock(
            side_effect=telegram.error.TelegramError("blocked")
        )

        await send_execution_result(app, 123, None, "BTC_USDT")


class TestSendExecutionError:
    @pytest.mark.asyncio
    async def test_sends_error_message_with_symbol_and_exception(
        self, mock_telegram_app
    ) -> None:
        """Message contains the symbol and the exception text, with parse_mode HTML."""
        app = mock_telegram_app

        await send_execution_error(
            app, 123, "BTC_USDT", ValueError("insufficient balance")
        )

        app.bot.send_message.assert_awaited_once()
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["parse_mode"] == "HTML"
        assert call_kwargs["chat_id"] == 123
        text = call_kwargs["text"]
        assert "BTC/USDT" in text
        assert "insufficient balance" in text

    @pytest.mark.asyncio
    async def test_escapes_html_in_exception_message(self, mock_telegram_app) -> None:
        """HTML in the exception message is escaped."""
        app = mock_telegram_app

        await send_execution_error(app, 123, "ETH_USDT", RuntimeError("<bad>&value"))

        text = app.bot.send_message.call_args[1]["text"]
        assert "<bad>" not in text
        assert "&lt;bad&gt;" in text
        assert "&amp;value" in text

    @pytest.mark.asyncio
    async def test_escapes_html_in_symbol(self, mock_telegram_app) -> None:
        """Symbol with HTML characters is escaped in the error alert."""
        app = mock_telegram_app

        await send_execution_error(app, 123, "<XSS>_USDT", Exception("err"))

        text = app.bot.send_message.call_args[1]["text"]
        assert "<XSS>" not in text

    @pytest.mark.asyncio
    async def test_handles_telegram_error_gracefully(self, mock_telegram_app) -> None:
        """TelegramError does not propagate."""
        app = mock_telegram_app
        app.bot.send_message = AsyncMock(
            side_effect=telegram.error.TelegramError("blocked")
        )

        await send_execution_error(app, 123, "BTC_USDT", Exception("err"))
