"""Tests for src.api.telegram.handlers — Interactive Telegram menu handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.ext import ConversationHandler

from src.api.telegram.handlers import (
    NAVIGATING,
    WAITING_ADD_SYMBOL,
    WAITING_LEVERAGE,
    WAITING_REMOVE_SYMBOL,
    WAITING_RISK,
    _check_credentials,
    _is_authorized,
    build_conversation_handler,
    handle_callback,
    receive_add_symbol,
    receive_leverage,
    receive_remove_symbol,
    receive_risk,
    start_command,
)

# --- Helpers ---


def _make_update(chat_id: int = 123, message_id: int = 100) -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.message_id = message_id
    update.message.reply_text = AsyncMock()
    update.message.text = ""
    return update


def _make_callback_update(chat_id: int = 123, data: str = "menu:main") -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message.chat_id = chat_id
    update.callback_query.message.message_id = 100
    update.callback_query.message.reply_text = AsyncMock()
    return update


def _make_context(evaluate_fn=None, paused: bool = False, executor=None) -> MagicMock:
    context = MagicMock()
    # Provide a default mock executor so _on_balance/_on_positions/_on_panic
    # don't hit the early "no executor" guard.  Tests that need it absent can
    # pass executor=None explicitly.
    _executor = executor if executor is not None else MagicMock()
    context.application.bot_data = {"paused": paused, "executor": _executor}
    if evaluate_fn is not None:
        context.application.bot_data["evaluate_fn"] = evaluate_fn
    context.bot.send_message = AsyncMock()
    context.bot.delete_message = AsyncMock()
    return context


class TestCheckCredentials:
    @patch.dict(
        "os.environ",
        {
            "AUTHORIZED_CHAT_ID": "123",
            "BINANCE_API_KEY": "key",
            "BINANCE_API_SECRET": "secret",
        },
    )
    def test_returns_ok_when_all_present(self) -> None:
        ok, cid = _check_credentials()
        assert ok is True
        assert cid == 123

    @patch.dict(
        "os.environ",
        {
            "AUTHORIZED_CHAT_ID": "123",
            "BINANCE_API_KEY": "",
            "BINANCE_API_SECRET": "secret",
        },
        clear=True,
    )
    def test_returns_false_when_api_key_missing(self) -> None:
        ok, _ = _check_credentials()
        assert ok is False

    @patch.dict(
        "os.environ",
        {
            "AUTHORIZED_CHAT_ID": "",
            "BINANCE_API_KEY": "key",
            "BINANCE_API_SECRET": "secret",
        },
        clear=True,
    )
    def test_returns_false_when_chat_id_missing(self) -> None:
        ok, _ = _check_credentials()
        assert ok is False

    @patch.dict(
        "os.environ",
        {
            "AUTHORIZED_CHAT_ID": "abc",
            "BINANCE_API_KEY": "key",
            "BINANCE_API_SECRET": "secret",
        },
    )
    def test_returns_false_on_invalid_chat_id(self) -> None:
        ok, cid = _check_credentials()
        assert ok is False
        assert cid is None


class TestIsAuthorized:
    @patch("src.api.telegram.handlers._check_credentials", return_value=(True, 123))
    def test_returns_true_when_matching(self, _m: MagicMock) -> None:
        assert _is_authorized(123) is True

    @patch("src.api.telegram.handlers._check_credentials", return_value=(True, 123))
    def test_returns_false_on_mismatch(self, _m: MagicMock) -> None:
        assert _is_authorized(999) is False

    @patch("src.api.telegram.handlers._check_credentials", return_value=(False, None))
    def test_returns_false_when_credentials_missing(self, _m: MagicMock) -> None:
        assert _is_authorized(123) is False


class TestStartCommand:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._check_credentials", return_value=(True, 123))
    async def test_sends_main_menu_when_authorized(self, _m: MagicMock) -> None:
        update = _make_update(chat_id=123)
        context = _make_context()
        result = await start_command(update, context)
        assert result == NAVIGATING
        update.message.reply_text.assert_awaited_once()
        call_kwargs = update.message.reply_text.call_args
        assert "MAIN MENU" in call_kwargs[0][0] or "MAIN MENU" in str(call_kwargs)

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._check_credentials", return_value=(True, 123))
    async def test_denies_wrong_chat_id(self, _m: MagicMock) -> None:
        update = _make_update(chat_id=999)
        context = _make_context()
        result = await start_command(update, context)
        assert result == ConversationHandler.END
        text = update.message.reply_text.call_args[0][0]
        assert "Access Denied" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._check_credentials", return_value=(False, None))
    async def test_denies_missing_credentials(self, _m: MagicMock) -> None:
        update = _make_update(chat_id=123)
        context = _make_context()
        result = await start_command(update, context)
        assert result == ConversationHandler.END


class TestNavigationCallbacks:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_menu_main(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="menu:main")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        update.callback_query.edit_message_text.assert_awaited_once()
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "MAIN MENU" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_menu_bot(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="menu:bot")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "BOT MENU" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_menu_bot_shows_resume_label_when_paused(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="menu:bot")
        await handle_callback(update, _make_context(paused=True))
        markup = update.callback_query.edit_message_text.call_args[1]["reply_markup"]
        button_texts = [btn.text for row in markup.inline_keyboard for btn in row]
        assert any("Resume" in t for t in button_texts)
        assert not any("Pause" in t for t in button_texts)

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_menu_exchange(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="menu:exchange")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "EXCHANGE BOT" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._ui.load_settings",
        return_value={
            "futures": {
                "margin_type": "ISOLATED",
                "default_leverage": 5,
                "symbols": ["BTC_USDT"],
            },
            "global": {"risk_per_trade_pct": 1.0},
        },
    )
    async def test_menu_futures(self, _ms: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="menu:futures")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "FUTURES" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_menu_spot_shows_alert(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="menu:spot")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        update.callback_query.answer.assert_awaited_once()
        call_args = update.callback_query.answer.call_args
        assert "Coming Soon" in call_args[0][0]
        assert call_args[1]["show_alert"] is True

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=False)
    async def test_unauthorized_callback_returns_end(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="menu:main")
        result = await handle_callback(update, _make_context())
        assert result == ConversationHandler.END


class TestBotActions:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.load_bot_state",
        return_value={
            "symbols": {"futures": ["BTC_USDT", "ETH_USDT"]},
        },
    )
    async def test_status_shows_symbol_count(
        self, _ms: MagicMock, _ma: MagicMock
    ) -> None:
        update = _make_callback_update(data="action:status")
        context = _make_context(paused=False)
        result = await handle_callback(update, context)
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Active" in text
        assert "2" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_pause_sets_flag(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="action:pause")
        context = _make_context()
        await handle_callback(update, context)
        assert context.application.bot_data["paused"] is True

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_resume_clears_flag(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="action:resume")
        context = _make_context(paused=True)
        await handle_callback(update, context)
        assert context.application.bot_data["paused"] is False

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_add_symbol_enters_state(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="action:add_symbol")
        result = await handle_callback(update, _make_context())
        assert result == WAITING_ADD_SYMBOL

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "symbols": {"futures": ["BTC_USDT"]},
        },
    )
    async def test_remove_symbol_enters_state(
        self, _ms: MagicMock, _ma: MagicMock
    ) -> None:
        update = _make_callback_update(data="action:remove_symbol")
        result = await handle_callback(update, _make_context())
        assert result == WAITING_REMOVE_SYMBOL

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "user_preferences": {"default_leverage": 10},
        },
    )
    async def test_leverage_enters_state(self, _ms: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:leverage")
        result = await handle_callback(update, _make_context())
        assert result == WAITING_LEVERAGE

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "user_preferences": {"risk_per_trade_pct": 1.0},
        },
    )
    async def test_risk_enters_state(self, _ms: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:risk")
        result = await handle_callback(update, _make_context())
        assert result == WAITING_RISK

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_train_shows_error_when_pipeline_not_configured(
        self, _ma: MagicMock
    ) -> None:
        update = _make_callback_update(data="action:train")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        update.callback_query.edit_message_text.assert_awaited_once()
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Internal Error" in text


class TestFuturesActions:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.load_bot_state",
        return_value={
            "margin_type": "ISOLATED",
        },
    )
    @patch("src.api.telegram._actions.save_bot_state")
    async def test_margin_toggle_isolated_to_crossed(
        self,
        mock_save: MagicMock,
        _ms: MagicMock,
        _ma: MagicMock,
    ) -> None:
        update = _make_callback_update(data="action:margin_toggle")
        await handle_callback(update, _make_context())
        saved = mock_save.call_args[0][0]
        assert saved["margin_type"] == "CROSSED"

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.load_bot_state",
        return_value={
            "margin_type": "CROSSED",
        },
    )
    @patch("src.api.telegram._actions.save_bot_state")
    async def test_margin_toggle_crossed_to_isolated(
        self,
        mock_save: MagicMock,
        _ms: MagicMock,
        _ma: MagicMock,
    ) -> None:
        update = _make_callback_update(data="action:margin_toggle")
        await handle_callback(update, _make_context())
        saved = mock_save.call_args[0][0]
        assert saved["margin_type"] == "ISOLATED"

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_panic_shows_confirmation(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="action:panic")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "PANIC" in text or "sure" in text.lower()

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._ui.load_settings",
        return_value={
            "futures": {"margin_type": "ISOLATED"},
        },
    )
    async def test_panic_cancel_returns_futures(
        self, _ms: MagicMock, _ma: MagicMock
    ) -> None:
        update = _make_callback_update(data="action:panic_cancel")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "FUTURES" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_clear_deletes_messages(self, _m: MagicMock) -> None:
        update = _make_callback_update(data="action:clear")
        context = _make_context()
        await handle_callback(update, context)
        assert context.bot.delete_message.await_count == 50

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_scan_without_evaluate_fn(self, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:scan")
        context = _make_context()  # no evaluate_fn
        result = await handle_callback(update, context)
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Error" in text or "error" in text.lower()

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._ui.load_settings",
        return_value={
            "futures": {"margin_type": "ISOLATED"},
        },
    )
    async def test_scan_with_signals(self, _ms: MagicMock, _ma: MagicMock) -> None:
        mock_eval = AsyncMock(return_value=2)
        update = _make_callback_update(data="action:scan")
        context = _make_context(evaluate_fn=mock_eval)
        result = await handle_callback(update, context)
        assert result == NAVIGATING
        mock_eval.assert_awaited_once()


class TestBinanceActions:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=1234.5678,
    )
    async def test_balance_success(self, _mt: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:balance")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "1234.5678" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    )
    async def test_balance_error(self, _mt: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:balance")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Error" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=[],
    )
    async def test_positions_empty(self, _mt: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:positions")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No open positions" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=[
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "entryPrice": "65000",
                "unRealizedProfit": "50.5",
            },
        ],
    )
    async def test_positions_with_data(self, _mt: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:positions")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "BTCUSDT" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        side_effect=Exception("err"),
    )
    async def test_positions_error(self, _mt: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:positions")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Error" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._ui.load_settings",
        return_value={"futures": {"margin_type": "ISOLATED"}},
    )
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=3,
    )
    async def test_panic_confirm_success(
        self, _mt: MagicMock, _ms: MagicMock, _ma: MagicMock
    ) -> None:
        update = _make_callback_update(data="action:panic_confirm")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Panic executed" in text or "3" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    @patch(
        "src.api.telegram._actions.asyncio.to_thread",
        new_callable=AsyncMock,
        side_effect=Exception("err"),
    )
    async def test_panic_confirm_error(self, _mt: MagicMock, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:panic_confirm")
        result = await handle_callback(update, _make_context())
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Error" in text

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_balance_no_executor(self, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:balance")
        context = _make_context()
        context.application.bot_data["executor"] = None
        result = await handle_callback(update, context)
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "not available" in text.lower()

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_positions_no_executor(self, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:positions")
        context = _make_context()
        context.application.bot_data["executor"] = None
        result = await handle_callback(update, context)
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "not available" in text.lower()

    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers._is_authorized", return_value=True)
    async def test_panic_confirm_no_executor(self, _ma: MagicMock) -> None:
        update = _make_callback_update(data="action:panic_confirm")
        context = _make_context()
        context.application.bot_data["executor"] = None
        result = await handle_callback(update, context)
        assert result == NAVIGATING
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "not available" in text.lower()


class TestReceiveAddSymbol:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers.save_bot_state")
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "symbols": {"futures": ["BTC_USDT"]},
        },
    )
    async def test_adds_valid_symbol(
        self, _ms: MagicMock, mock_save: MagicMock
    ) -> None:
        update = _make_update()
        update.message.text = "DOT_USDT"
        result = await receive_add_symbol(update, _make_context())
        assert result == NAVIGATING
        saved = mock_save.call_args[0][0]
        assert "DOT_USDT" in saved["symbols"]["futures"]

    @pytest.mark.asyncio
    async def test_rejects_invalid_format(self) -> None:
        update = _make_update()
        update.message.text = "btc-usdt"
        result = await receive_add_symbol(update, _make_context())
        assert result == WAITING_ADD_SYMBOL

    @pytest.mark.asyncio
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "symbols": {"futures": ["BTC_USDT"]},
        },
    )
    async def test_rejects_duplicate(self, _ms: MagicMock) -> None:
        update = _make_update()
        update.message.text = "BTC_USDT"
        result = await receive_add_symbol(update, _make_context())
        assert result == WAITING_ADD_SYMBOL


class TestReceiveRemoveSymbol:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers.save_bot_state")
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "symbols": {"futures": ["BTC_USDT", "ETH_USDT"]},
        },
    )
    async def test_removes_existing_symbol(
        self, _ms: MagicMock, mock_save: MagicMock
    ) -> None:
        update = _make_update()
        update.message.text = "ETH_USDT"
        result = await receive_remove_symbol(update, _make_context())
        assert result == NAVIGATING
        saved = mock_save.call_args[0][0]
        assert "ETH_USDT" not in saved["symbols"]["futures"]

    @pytest.mark.asyncio
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "symbols": {"futures": ["BTC_USDT"]},
        },
    )
    async def test_rejects_nonexistent_symbol(self, _ms: MagicMock) -> None:
        update = _make_update()
        update.message.text = "XYZ_USDT"
        result = await receive_remove_symbol(update, _make_context())
        assert result == WAITING_REMOVE_SYMBOL


class TestReceiveLeverage:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers.save_bot_state")
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "user_preferences": {"default_leverage": 5},
            "margin_type": "ISOLATED",
        },
    )
    async def test_valid_leverage(self, _ms: MagicMock, mock_save: MagicMock) -> None:
        update = _make_update()
        update.message.text = "20"
        result = await receive_leverage(update, _make_context())
        assert result == NAVIGATING
        saved = mock_save.call_args[0][0]
        assert saved["user_preferences"]["default_leverage"] == 20

    @pytest.mark.asyncio
    async def test_non_integer_rejected(self) -> None:
        update = _make_update()
        update.message.text = "abc"
        result = await receive_leverage(update, _make_context())
        assert result == WAITING_LEVERAGE

    @pytest.mark.asyncio
    async def test_out_of_range_rejected(self) -> None:
        update = _make_update()
        update.message.text = "200"
        result = await receive_leverage(update, _make_context())
        assert result == WAITING_LEVERAGE


class TestReceiveRisk:
    @pytest.mark.asyncio
    @patch("src.api.telegram.handlers.save_bot_state")
    @patch(
        "src.api.telegram.handlers.load_bot_state",
        return_value={
            "user_preferences": {"risk_per_trade_pct": 1.0},
            "margin_type": "ISOLATED",
        },
    )
    async def test_valid_risk(self, _ms: MagicMock, mock_save: MagicMock) -> None:
        update = _make_update()
        update.message.text = "2.5"
        result = await receive_risk(update, _make_context())
        assert result == NAVIGATING
        saved = mock_save.call_args[0][0]
        assert saved["user_preferences"]["risk_per_trade_pct"] == 2.5

    @pytest.mark.asyncio
    async def test_non_numeric_rejected(self) -> None:
        update = _make_update()
        update.message.text = "abc"
        result = await receive_risk(update, _make_context())
        assert result == WAITING_RISK

    @pytest.mark.asyncio
    async def test_zero_rejected(self) -> None:
        update = _make_update()
        update.message.text = "0"
        result = await receive_risk(update, _make_context())
        assert result == WAITING_RISK

    @pytest.mark.asyncio
    async def test_negative_rejected(self) -> None:
        update = _make_update()
        update.message.text = "-5"
        result = await receive_risk(update, _make_context())
        assert result == WAITING_RISK


class TestBuildConversationHandler:
    def test_returns_conversation_handler(self) -> None:
        handler = build_conversation_handler()
        assert isinstance(handler, ConversationHandler)
