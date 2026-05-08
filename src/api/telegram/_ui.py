"""_ui.py — Pure UI building blocks: state constants, keyboards, and text builders.

These functions carry no ``telegram.ext`` dependency, so they can be imported
by both ``handlers`` and ``_actions`` without risk of circular imports.
The only side-effectful functions are ``_txt_futures`` and ``_current_margin``,
which perform a single synchronous read of ``settings.yaml``.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from src.config.settings_loader import load_settings

# --- ConversationHandler states ---
NAVIGATING = 0
WAITING_ADD_SYMBOL = 1
WAITING_REMOVE_SYMBOL = 2
WAITING_LEVERAGE = 3
WAITING_RISK = 4


# --- Keyboard builders ---


def _kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🤖 Bot Menu", callback_data="menu:bot")],
            [InlineKeyboardButton("💱 Exchange Bot", callback_data="menu:exchange")],
            [InlineKeyboardButton("📊 Status", callback_data="action:status")],
            [InlineKeyboardButton("🧹 Clear Chat", callback_data="action:clear")],
        ]
    )


def _kb_bot(paused: bool) -> InlineKeyboardMarkup:
    toggle_text = "▶️ Resume Bot" if paused else "⏸️ Pause Bot"
    toggle_data = "action:resume" if paused else "action:pause"
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("➕ Add Symbol", callback_data="action:add_symbol")],
            [
                InlineKeyboardButton(
                    "🗑️ Remove Symbol", callback_data="action:remove_symbol"
                )
            ],
            [InlineKeyboardButton("🧠 Train AI", callback_data="action:train")],
            [InlineKeyboardButton(toggle_text, callback_data=toggle_data)],
            [InlineKeyboardButton("🔙 Volver", callback_data="menu:main")],
        ]
    )


def _kb_exchange() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📈 Futures", callback_data="menu:futures")],
            [InlineKeyboardButton("🪙 Spot", callback_data="menu:spot")],
            [InlineKeyboardButton("🔙 Back", callback_data="menu:main")],
        ]
    )


def _kb_futures(margin_type: str) -> InlineKeyboardMarkup:
    margin_label = f"🔄 Margen: {margin_type}"
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💰 View Balance", callback_data="action:balance")],
            [
                InlineKeyboardButton(
                    "📋 View Positions", callback_data="action:positions"
                )
            ],
            [InlineKeyboardButton("🔎 Scan Market", callback_data="action:scan")],
            [
                InlineKeyboardButton(
                    "⚙️ Modify Leverage", callback_data="action:leverage"
                )
            ],
            [InlineKeyboardButton("⚖️ Modify Risk (%)", callback_data="action:risk")],
            [InlineKeyboardButton(margin_label, callback_data="action:margin_toggle")],
            [
                InlineKeyboardButton(
                    "🚨 Panic Button (Close All)", callback_data="action:panic"
                )
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="menu:exchange")],
        ]
    )


def _kb_back(target: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔙 Back", callback_data=target)],
        ]
    )


def _kb_panic_confirm() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ YES", callback_data="action:panic_confirm"),
                InlineKeyboardButton("❌ NO", callback_data="action:panic_cancel"),
            ]
        ]
    )


# --- Text builders ---


def _txt_main() -> str:
    return "📋 <b>MAIN MENU</b>\n\nSelect an option:"


def _txt_bot() -> str:
    return "🤖 <b>BOT MENU</b>\n\nSystem management:"


def _txt_exchange() -> str:
    return "💱 <b>EXCHANGE BOT</b>\n\nSelect a market:"


def _txt_futures() -> str:
    s = load_settings()
    fut = s.get("futures", {})
    glb = s.get("global", {})
    return (
        f"📈 <b>FUTURES</b>\n\n"
        f"⚙️ Leverage: <b>{fut.get('default_leverage', 1)}x</b>\n"
        f"🔄 Margin: <b>{fut.get('margin_type', 'ISOLATED')}</b>\n"
        f"⚖️ Risk: <b>{glb.get('risk_per_trade_pct', 1.0)}%</b>\n"
        f"📊 Symbols: <b>{len(fut.get('symbols', []))}</b>\n\n"
        f"Select an option:"
    )


def _current_margin() -> str:
    return load_settings().get("futures", {}).get("margin_type", "ISOLATED")
