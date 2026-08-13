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

import pandas as pd

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


def get_funding_rate_path(symbol: str) -> Path:
    """Return the canonical path for a symbol's Futures funding rate history.

    Layout::

        data/raw_csv/{safe_symbol}/funding_rate.csv

    Args:
        symbol: Trading pair in safe format (e.g. ``'BTC_USDT'``).

    Returns:
        ``pathlib.Path`` to the funding rate CSV.
    """
    safe_symbol = _sanitize_symbol(symbol)
    return get_project_root() / "data" / "raw_csv" / safe_symbol / "funding_rate.csv"


def load_funding_rate_csv(symbol: str) -> pd.DataFrame:
    """Load a previously downloaded funding rate CSV.

    The CSV is expected to have a ``funding_rate`` column indexed by
    ``timestamp`` (datetime of the funding *settlement* — i.e. the
    exact moment the rate was applied, e.g. 00:00, 08:00, 16:00 UTC
    every day on Binance USD-M Futures).

    Args:
        symbol: Trading pair in safe format (e.g. ``'BTC_USDT'``).

    Returns:
        DataFrame with DatetimeIndex and a ``funding_rate`` column.
        Empty DataFrame if the file does not exist.
    """
    path = get_funding_rate_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def _sanitize_symbol(symbol: str) -> str:
    """Normalize any symbol format to the file-safe ``XXX_USDT`` form.

    Args:
        symbol: Symbol in any format (e.g. ``'BTC/USDT:USDT'``).

    Returns:
        Normalized format (e.g. ``'BTC_USDT'``).
    """
    return symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"


def get_report_dir(symbol: str) -> Path:
    """Return the report output directory for a symbol.

    Layout::

        reports/{safe_symbol}/

    Args:
        symbol: Trading pair in any accepted format.

    Returns:
        Path to the symbol's report directory (created on write, not here).
    """
    safe_symbol = _sanitize_symbol(symbol)
    return get_project_root() / "reports" / safe_symbol


def get_report_path(symbol: str, experiment_name: str, timestamp: str | None = None) -> Path:
    """Build a timestamped report file path under ``reports/{symbol}/``.

    Args:
        symbol: Trading pair.
        experiment_name: Short experiment slug (e.g. ``'baseline'``).
        timestamp: Optional ``YYYYMMDD_HHMMSS`` stamp; auto-generated when ``None``.

    Returns:
        Full path to the JSON report file.
    """
    from datetime import datetime

    safe_symbol = _sanitize_symbol(symbol)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_report_dir(safe_symbol) / f"{experiment_name}_{stamp}.json"

