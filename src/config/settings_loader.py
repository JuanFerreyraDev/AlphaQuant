"""settings_loader.py — Configuration loading and access for AlphaQuant.

Reads ``settings.yaml`` from the project root deterministically
using pathlib instead of os.getcwd().
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _settings_path() -> Path:
    """Return the absolute path to ``settings.yaml``."""
    return _PROJECT_ROOT / "settings.yaml"


def load_settings() -> dict[str, Any]:
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


def save_settings(data: dict[str, Any]) -> None:
    """Write the complete dictionary back to ``settings.yaml``.

    Args:
        data: Configuration dictionary to persist.
    """
    filepath = _settings_path()
    try:
        with filepath.open("w", encoding="utf-8") as fh:
            yaml.dump(
                data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        logger.info("settings.yaml saved successfully.")
    except OSError as exc:
        logger.error("Error writing settings.yaml: %s", exc)


def get_project_root() -> Path:
    """Return the absolute path to the AlphaQuant project root.

    Returns:
        ``pathlib.Path`` pointing to ``AlphaQuant/``.
    """
    return _PROJECT_ROOT
