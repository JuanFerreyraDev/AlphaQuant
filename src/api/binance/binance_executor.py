"""binance_executor.py — Execution module for Binance Futures.

Single responsibility: execute trades on Binance Futures robustly, with
isolated margin management, dynamic leverage, position sizing based on
the 1% balance rule, and automatic protection via STOP_MARKET /
TAKE_PROFIT_MARKET orders using the Algo Orders API.

DECOUPLING RULE: This module does NOT import anything from ``src.api.telegram``.
"""

import logging
import math
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

from src.config.settings_loader import get_market_config, load_settings

logger = logging.getLogger(__name__)


class BinanceExecutor:
    """Order executor for Binance USDT-M Futures.

    Business rules:
      - 1 position at a time per symbol (no averaging).
      - ISOLATED margin + leverage read from settings.yaml.
      - Size = 1% of available USDT balance x leverage.
      - Respects Binance LOT_SIZE and MIN_NOTIONAL filters.
      - Places SL and TP using STOP_MARKET and TAKE_PROFIT_MARKET with closePosition=True.
    """

    def __init__(self) -> None:
        load_dotenv()

        api_key: str = os.getenv("BINANCE_API_KEY", "")
        api_secret: str = os.getenv("BINANCE_API_SECRET", "")
        use_testnet: bool = os.getenv("USE_TESTNET", "True").lower() in (
            "true",
            "1",
            "yes",
        )

        if not api_key or not api_secret:
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be defined in .env"
            )

        self.client: Client = Client(api_key, api_secret, testnet=use_testnet)

        env_label = "TESTNET" if use_testnet else "MAINNET"
        logger.info("Connected to Binance Futures (%s).", env_label)

        futures_cfg = get_market_config("futures")
        self.leverage: int = int(futures_cfg.get("default_leverage", 1))
        logger.info("Global leverage configured: %dx", self.leverage)

        global_cfg = load_settings().get("global", {})
        self.risk_pct: float = global_cfg.get("risk_per_trade_pct", 1.0) / 100
        logger.info("Risk per trade: %.2f%%", self.risk_pct * 100)

    def _has_open_position(self, symbol: str) -> bool:
        """Return True if an open position already exists for the symbol.

        Args:
            symbol: Normalized symbol (e.g. ``'BTCUSDT'``).

        Returns:
            ``True`` if there is an open position or if the query fails.
        """
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                amt = float(pos.get("positionAmt", 0))
                if amt != 0.0:
                    logger.warning(
                        "Open position detected for %s (amount: %s). "
                        "New order canceled to avoid averaging.",
                        symbol,
                        amt,
                    )
                    return True
            return False
        except BinanceAPIException as exc:
            logger.error("Error querying positions for %s: %s", symbol, exc)
            return True

    def _configure_symbol(self, symbol: str) -> bool:
        """Configure the symbol in ISOLATED mode and set leverage.

        Args:
            symbol: Normalized symbol.

        Returns:
            ``True`` if the configuration was successful.
        """
        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            logger.info("Margin type set to ISOLATED for %s.", symbol)
        except BinanceAPIException as exc:
            if exc.code == -4046:
                logger.info("%s already in ISOLATED mode.", symbol)
            else:
                logger.error("Error setting margin type for %s: %s", symbol, exc)
                return False

        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
            logger.info("Leverage set to %dx for %s.", self.leverage, symbol)
        except BinanceAPIException as exc:
            logger.error("Error setting leverage for %s: %s", symbol, exc)
            return False

        return True

    def _get_futures_balance(self) -> float:
        """Query the available USDT balance in the Futures account.

        Returns:
            Available balance in USDT.
        """
        try:
            balances = self.client.futures_account_balance()
            for asset in balances:
                if asset["asset"] == "USDT":
                    available = float(asset["availableBalance"])
                    logger.info("USDT balance available in Futures: %.4f", available)
                    return available
            logger.warning("No USDT balance found in Futures.")
            return 0.0
        except BinanceAPIException as exc:
            logger.error("Error querying Futures balance: %s", exc)
            return 0.0

    def _get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Extract LOT_SIZE, MIN_NOTIONAL, and PRICE_FILTER filters from the exchange.

        Args:
            symbol: Normalized symbol.

        Returns:
            Dictionary with ``step_size``, ``min_qty``, ``min_notional``, ``tick_size``.
        """
        result: dict[str, Any] = {
            "step_size": "1",
            "min_qty": "1",
            "min_notional": 0.0,
            "tick_size": "0.01",
        }
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            result["step_size"] = f["stepSize"]
                            result["min_qty"] = f["minQty"]
                        if f["filterType"] == "MIN_NOTIONAL":
                            result["min_notional"] = float(f.get("notional", 0))
                        if f["filterType"] == "PRICE_FILTER":
                            result["tick_size"] = f["tickSize"]
                    break
        except BinanceAPIException as exc:
            logger.error("Error getting exchange filters for %s: %s", symbol, exc)
        return result

    def _round_step_size(self, quantity: float, step_size: str) -> float:
        """Round the quantity down respecting the step_size.

        Args:
            quantity: Unrounded quantity.
            step_size: Step size from the LOT_SIZE filter.

        Returns:
            Rounded quantity.
        """
        d_qty = Decimal(str(quantity))
        d_step = Decimal(step_size)
        rounded = (d_qty / d_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * d_step
        return float(rounded)

    def _round_tick_size(self, price: float, tick_size: str) -> str:
        """Round a price down respecting the PRICE_FILTER tick_size.

        Args:
            price: Unrounded price.
            tick_size: Tick size from the PRICE_FILTER.

        Returns:
            Rounded price as a string ready for the API.
        """
        d_price = Decimal(str(price))
        d_tick = Decimal(tick_size)
        rounded = (d_price / d_tick).quantize(
            Decimal("1"), rounding=ROUND_DOWN
        ) * d_tick
        return str(rounded)

    def _calculate_quantity(self, symbol: str) -> Optional[float]:
        """Calculate the exact quantity to trade using the 1% rule.

        Args:
            symbol: Normalized symbol.

        Returns:
            Calculated quantity or ``None`` if trading is not possible.
        """
        balance = self._get_futures_balance()
        if balance <= 0:
            logger.error("Insufficient balance (%.4f USDT). Aborting.", balance)
            return None

        margin = balance * self.risk_pct
        notional_size = margin * self.leverage
        logger.info(
            "Risk rule: Margin=%.4f USDT | Notional=%.4f USDT (leverage %dx)",
            margin,
            notional_size,
            self.leverage,
        )

        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker["price"])
            logger.info("Current price of %s: %.8f", symbol, current_price)
        except BinanceAPIException as exc:
            logger.error("Error getting price for %s: %s", symbol, exc)
            return None

        if current_price <= 0:
            logger.error("Invalid price for %s: %.8f", symbol, current_price)
            return None

        raw_qty = notional_size / current_price
        filters = self._get_symbol_filters(symbol)
        quantity = self._round_step_size(raw_qty, filters["step_size"])

        if quantity < float(filters["min_qty"]):
            logger.error(
                "Calculated quantity (%.8f) below minQty (%.8f) for %s.",
                quantity,
                float(filters["min_qty"]),
                symbol,
            )
            return None

        actual_notional = quantity * current_price
        if filters["min_notional"] > 0 and actual_notional < filters["min_notional"]:
            logger.error(
                "Actual notional (%.4f USDT) below MIN_NOTIONAL (%.4f USDT) for %s.",
                actual_notional,
                filters["min_notional"],
                symbol,
            )
            return None

        logger.info("Final calculated quantity for %s: %.8f", symbol, quantity)
        return quantity

    def _place_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        stop_price: str,
    ) -> Optional[dict[str, Any]]:
        """Place an algorithmic order (STOP_MARKET or TAKE_PROFIT_MARKET).

        Args:
            symbol: Normalized symbol.
            side: ``'BUY'`` or ``'SELL'``.
            order_type: ``'STOP'`` or ``'TAKE_PROFIT'``.
            stop_price: Trigger price as a string.

        Returns:
            API response or ``None`` on failure.
        """
        try:
            if order_type == "STOP":
                b_type = "STOP_MARKET"
            elif order_type == "TAKE_PROFIT":
                b_type = "TAKE_PROFIT_MARKET"
            else:
                logger.error("Invalid order type: %s", order_type)
                return None

            params = {
                "symbol": symbol,
                "side": side,
                "algoType": "CONDITIONAL",
                "type": b_type,
                "triggerPrice": stop_price,
                "closePosition": "TRUE",
                "workingType": "MARK_PRICE",
            }

            response = self.client._request_futures_api(
                "post", "algoOrder", signed=True, data=params
            )

            order_id = response.get("orderId") or response.get("clientAlgoId", "N/A")
            logger.info(
                "Order %s placed successfully: %s @ %s", b_type, order_id, stop_price
            )
            return response

        except BinanceAPIException as exc:
            logger.error("Error placing %s order for %s: %s", order_type, symbol, exc)
            return None

    def get_futures_balance(self) -> float:
        """Return the available USDT balance in the Futures account."""
        return self._get_futures_balance()

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return all currently open Futures positions.

        Returns:
            List of position dicts from the Binance API that have a
            non-zero ``positionAmt``.
        """
        positions = self.client.futures_position_information()
        return [p for p in positions if float(p.get("positionAmt", 0)) != 0.0]

    def close_all_positions(self) -> int:
        """Close all open Futures positions and cancel all pending orders.

        Returns:
            Number of positions successfully closed.
        """
        positions = self.client.futures_position_information()
        open_pos = [p for p in positions if float(p.get("positionAmt", 0)) != 0.0]
        closed = 0
        symbols_touched: set[str] = set()

        for p in open_pos:
            sym = p["symbol"]
            amt = float(p["positionAmt"])
            side = "SELL" if amt > 0 else "BUY"
            try:
                self.client.futures_create_order(
                    symbol=sym,
                    side=side,
                    type="MARKET",
                    quantity=abs(amt),
                )
                closed += 1
                symbols_touched.add(sym)
            except BinanceAPIException as exc:
                logger.error("Error closing position %s: %s", sym, exc)
                symbols_touched.add(sym)

        for sym in symbols_touched:
            try:
                self.client.futures_cancel_all_open_orders(symbol=sym)
            except BinanceAPIException as exc:
                logger.error("Error cancelling open orders for %s: %s", sym, exc)

        return closed

    def execute_futures_trade(
        self,
        symbol: str,
        side: str,
        sl_price: float,
        tp_price: float,
    ) -> Optional[dict[str, Any]]:
        """Execute a complete trade on Binance Futures.

        Args:
            symbol: Trading pair (e.g. ``'BTC_USDT'``).
            side: ``'BUY'`` or ``'SELL'``.
            sl_price: Stop Loss price.
            tp_price: Take Profit price.

        Returns:
            Dictionary with ``entry``, ``stop_loss``, and ``take_profit`` or ``None``.
        """
        symbol = symbol.replace("_", "").replace("/", "").split(":")[0]
        side = side.upper()

        logger.info(
            "Starting trade: %s %s | SL=%.4f | TP=%.4f",
            side,
            symbol,
            sl_price,
            tp_price,
        )

        if self._has_open_position(symbol):
            return None

        if not self._configure_symbol(symbol):
            return None

        quantity = self._calculate_quantity(symbol)
        if quantity is None:
            return None

        filters = self._get_symbol_filters(symbol)
        tick_size = filters["tick_size"]
        sl_price_str = self._round_tick_size(sl_price, tick_size)
        tp_price_str = self._round_tick_size(tp_price, tick_size)

        entry_order: Optional[dict[str, Any]] = None
        try:
            entry_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            logger.info("MARKET order executed: %s", entry_order.get("orderId"))
        except BinanceAPIException as exc:
            logger.error("Error executing MARKET order for %s: %s", symbol, exc)
            return None

        time.sleep(1)

        close_side = "SELL" if side == "BUY" else "BUY"

        sl_order = self._place_algo_order(
            symbol=symbol,
            side=close_side,
            order_type="STOP",
            stop_price=sl_price_str,
        )

        tp_order = self._place_algo_order(
            symbol=symbol,
            side=close_side,
            order_type="TAKE_PROFIT",
            stop_price=tp_price_str,
        )

        logger.info("Trade completed for %s.", symbol)

        return {
            "entry": entry_order,
            "stop_loss": sl_order,
            "take_profit": tp_order,
        }
