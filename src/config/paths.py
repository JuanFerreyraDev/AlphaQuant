"""paths.py — Centralized path-building helpers for AlphaQuant.

Every module that needs to locate data files (raw CSVs, models, plots)
should import from here instead of hand-formatting paths.  This keeps
the directory layout defined in one place.

Storage layout (raw_csv)::

    data/raw_csv/
    ├── BTC_USDT/
    │   ├── 1h.csv
    │   ├── 4h.csv
    │   └── 1d.csv
    └── ETH_USDT/
        └── 4h.csv
"""

from pathlib import Path

from src.config.settings_loader import get_project_root


def get_raw_csv_path(symbol: str, timeframe: str) -> Path:
    """Return the canonical path for a symbol's raw CSV at a given timeframe.

    The path follows the per-symbol directory layout::

        data/raw_csv/{safe_symbol}/{timeframe}.csv

    Args:
        symbol: Trading pair in safe format (e.g. ``'BTC_USDT'``).
            Slash/colon variants are auto-normalized.
        timeframe: Candle interval string (e.g. ``'1d'``, ``'4h'``).

    Returns:
        ``pathlib.Path`` to the CSV file.  The file and parent
        directory may or may not exist yet — callers are responsible
        for creating them when writing.
    """
    safe_symbol = _sanitize_symbol(symbol)
    return get_project_root() / "data" / "raw_csv" / safe_symbol / f"{timeframe}.csv"


def _sanitize_symbol(symbol: str) -> str:
    """Normalize any symbol format to the file-safe ``XXX_USDT`` form.

    Args:
        symbol: Symbol in any format (e.g. ``'BTC/USDT:USDT'``).

    Returns:
        Normalized format (e.g. ``'BTC_USDT'``).
    """
    return symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"
