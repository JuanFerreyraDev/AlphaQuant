"""settings_loader.py — Configuration loading and access for AlphaQuant.

Architecture
~~~~~~~~~~~~
* ``settings.yaml``   → **read-only** factory defaults.
* ``data/bot_state.json`` → **read/write** runtime state managed by
  the Telegram UI and system processes.

The public ``load_settings()`` / ``get_config()`` function merges the
two sources (bot_state overrides YAML defaults) so that every consumer
transparently gets the correct value without knowing where it lives.
"""

import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

_DEFAULT_BOT_STATE: dict[str, Any] = {
    "active_market": "futures",
    "bot_active": True,
    "symbols": {
        "futures": ["BTC_USDT"],
        "spot": ["BTC_USDT"],
    },
    "user_preferences": {
        "risk_per_trade_pct": 1.0,
        "default_leverage": 2,
    },
    "margin_type": "ISOLATED",
}


def _settings_path() -> Path:
    """Return the absolute path to ``settings.yaml``."""
    return _PROJECT_ROOT / "settings.yaml"


def _bot_state_path() -> Path:
    """Return the absolute path to ``data/bot_state.json``."""
    return _PROJECT_ROOT / "data" / "bot_state.json"


def _load_yaml_defaults() -> dict[str, Any]:
    """Read ``settings.yaml`` and return the complete dictionary.

    Returns:
        Configuration dictionary. Empty if the file does not exist or
        contains invalid YAML.
    """
    filepath = _settings_path()

    if not filepath.exists():
        logger.error("settings.yaml not found at %s", filepath)
        return {}

    try:
        with filepath.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        logger.error("Error parsing settings.yaml: %s", exc)
        return {}
    except OSError as exc:
        logger.error("Error reading settings.yaml: %s", exc)
        return {}


def _ensure_bot_state() -> Path:
    """Create ``data/bot_state.json`` with the default schema if missing.

    Seeds initial values from ``settings.yaml`` when available so that
    existing YAML customizations are preserved on first startup.

    Returns:
        The absolute path to the bot state file.
    """
    path = _bot_state_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    yaml_cfg = _load_yaml_defaults()
    initial: dict[str, Any] = copy.deepcopy(_DEFAULT_BOT_STATE)

    if yaml_cfg:
        initial["active_market"] = yaml_cfg.get("global", {}).get(
            "active_market", initial["active_market"]
        )
        for market in ("futures", "spot"):
            syms = yaml_cfg.get(market, {}).get("symbols")
            if syms is not None:
                initial["symbols"][market] = syms
        prefs = initial["user_preferences"]
        prefs["risk_per_trade_pct"] = yaml_cfg.get("global", {}).get(
            "risk_per_trade_pct", prefs["risk_per_trade_pct"]
        )
        prefs["default_leverage"] = yaml_cfg.get("futures", {}).get(
            "default_leverage", prefs["default_leverage"]
        )
        initial["margin_type"] = yaml_cfg.get("futures", {}).get(
            "margin_type", initial["margin_type"]
        )
        initial["bot_active"] = yaml_cfg.get("global", {}).get(
            "default_bot_active", initial["bot_active"]
        )

    save_bot_state(initial)
    logger.info("Initialized bot_state.json at %s", path)
    return path


def load_bot_state() -> dict[str, Any]:
    """Read ``data/bot_state.json`` and return the state dictionary.

    If the file is missing it will be created with the default schema.
    If the file contains invalid JSON the default schema is returned.

    Returns:
        Bot-state dictionary.
    """
    path = _bot_state_path()
    if not path.exists():
        _ensure_bot_state()

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else copy.deepcopy(_DEFAULT_BOT_STATE)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error reading bot_state.json: %s", exc)
        return copy.deepcopy(_DEFAULT_BOT_STATE)


def save_bot_state(data: dict[str, Any]) -> None:
    """Write *data* to ``data/bot_state.json`` atomically.

    Writes to a temporary file in the same directory first, then uses
    ``os.replace`` (an atomic operation on POSIX) to swap it in.

    Args:
        data: State dictionary to persist.
    """
    path = _bot_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".bot_state_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4, ensure_ascii=False)
            os.replace(tmp_path, path)
            logger.info("bot_state.json saved successfully.")
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.error("Error writing bot_state.json: %s", exc)


def get_config() -> dict[str, Any]:
    """Return the merged configuration (bot_state over YAML defaults).

    Values present in ``bot_state.json`` take priority.  Anything
    missing falls back to ``settings.yaml``.

    Returns:
        Complete configuration dictionary.
    """
    merged = _load_yaml_defaults()
    if not merged:
        merged = {}
    merged = copy.deepcopy(merged)

    state = load_bot_state()

    if "active_market" in state:
        merged.setdefault("global", {})["active_market"] = state["active_market"]

    for market in ("futures", "spot"):
        if market in state.get("symbols", {}):
            merged.setdefault(market, {})["symbols"] = state["symbols"][market]

    prefs = state.get("user_preferences", {})
    if "risk_per_trade_pct" in prefs:
        merged.setdefault("global", {})["risk_per_trade_pct"] = prefs[
            "risk_per_trade_pct"
        ]
    if "default_leverage" in prefs:
        merged.setdefault("futures", {})["default_leverage"] = prefs["default_leverage"]

    if "margin_type" in state:
        merged.setdefault("futures", {})["margin_type"] = state["margin_type"]

    if "bot_active" in state:
        merged.setdefault("global", {})["bot_active"] = state["bot_active"]

    return merged


# Backward-compatible alias — every consumer that imports load_settings
# transparently gets the merged view.
load_settings = get_config


def get_bot_active() -> bool:
    """Return whether the bot is active (not paused).

    Reads from ``bot_state.json`` first; falls back to
    ``settings.yaml``'s ``global.default_bot_active``; defaults to ``True``.

    Returns:
        ``True`` if the bot should execute trading/evaluation logic.
    """
    return bool(load_settings().get("global", {}).get("bot_active", True))


def get_active_market() -> str:
    """Return the active market (``futures``, ``spot``, or ``both``).

    Returns:
        String with the active market. Defaults to ``'futures'``.
    """
    settings = load_settings()
    return settings.get("global", {}).get("active_market", "futures")


def get_active_symbols() -> list[str]:
    """Return the list of symbols for the current active market.

    Returns:
        List of symbol strings (e.g. ``['BTC_USDT', 'ETH_USDT']``).
    """
    settings = load_settings()
    active = get_active_market()

    if active == "both":
        f_syms: list[str] = settings.get("futures", {}).get("symbols", [])
        s_syms: list[str] = settings.get("spot", {}).get("symbols", [])
        return list(set(f_syms + s_syms))

    return settings.get(active, {}).get("symbols", [])


def get_market_config(market: str = "futures") -> dict[str, Any]:
    """Return the configuration for a specific market.

    Args:
        market: ``'futures'`` or ``'spot'``.

    Returns:
        Dictionary with the requested market configuration.
    """
    settings = load_settings()
    return settings.get(market, {})


def get_trading_settings() -> dict[str, float]:
    """Return trading execution settings (fee_rate, slippage) from settings.yaml.

    Reads directly from the YAML defaults so that values are available
    before ``bot_state.json`` is initialised.  Falls back to safe defaults
    if the ``trading`` section is absent.

    Returns:
        Dictionary with ``fee_rate`` and ``slippage`` keys.
    """
    data = _load_yaml_defaults()
    trading = data.get("trading", {})
    return {
        "fee_rate": float(trading.get("fee_rate", 0.001)),
        "slippage": float(trading.get("slippage", 0.0005)),
    }


def get_project_root() -> Path:
    """Return the absolute path to the AlphaQuant project root.

    Returns:
        ``pathlib.Path`` pointing to ``AlphaQuant/``.
    """
    return _PROJECT_ROOT
