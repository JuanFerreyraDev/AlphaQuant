"""visualize_val_signals.py — Diagnostic tool to inspect model signals on the Out-of-Sample Validation period.

Usage:
    python -m tools.visualize_val_signals BTC_USDT

Loads the saved config.json and .pkl model for a symbol, runs inference
strictly on the Out-of-Sample Validation slice (the last 20% of the
indicator-clean dataset — identical to the slice the optimizer evaluated),
and produces an interactive Plotly HTML chart overlaying buy signals on the
OHLCV candlestick. Detects regime overfitting by making temporal clustering
of signals visually obvious on unseen data.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.brain.data_fetcher import get_fear_and_greed

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path when run as a module
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.brain.features import add_sentiment, compute_all_technicals  # noqa: E402
from src.config.settings_loader import get_project_root  # noqa: E402
from src.utils.helpers import load_csv_data  # noqa: E402

# Mirrors the train/val+test split used by temporal_split_with_embargo
_VAL_START_PCT: float = 0.8


def _normalize_symbol(symbol: str) -> str:
    """Return a safe filesystem-friendly symbol string (e.g. 'BTC_USDT')."""
    return symbol.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"


def _load_config(model_dir: Path, safe_symbol: str) -> dict[str, Any]:
    """Load and validate config.json from the model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(
            f"[ERROR] config.json not found at {config_path}\n"
            f"Run strategy_optimizer.py for {safe_symbol} first."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_bundle(model_dir: Path, safe_symbol: str) -> dict[str, Any]:
    """Glob for the production .pkl bundle and load it with joblib.

    If multiple .pkl files exist, the most recently modified one is used.
    """
    pkl_files = sorted(model_dir.glob("*.pkl"))
    if not pkl_files:
        raise SystemExit(
            f"[ERROR] No .pkl model file found in {model_dir}\n"
            f"Run train.py for {safe_symbol} first."
        )
    bundle_path = max(pkl_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading model bundle: {bundle_path.name}")
    return joblib.load(bundle_path)


def _prepare_dataframe(safe_symbol: str) -> pd.DataFrame:
    """Load raw OHLCV CSV, compute all technical indicators, and attach sentiment.

    Does NOT call cleanup_columns — OHLCV columns are required for the chart.
    Attempts a live fetch of the Fear & Greed index; falls back to an empty
    DataFrame (technical-only mode) if the request fails.
    """
    filename = f"{safe_symbol}_1d.csv"
    try:
        df = load_csv_data(filename)
    except FileNotFoundError:
        raise SystemExit(
            f"[ERROR] Historical CSV not found: data/raw_csv/{filename}\n"
            "Run the data fetcher to download price data first."
        )
    compute_all_technicals(df)
    # TODO: Evaluate whether sentiment actually contributes to alpha in the future
    try:
        sentiment_df = get_fear_and_greed()
        df, _ = add_sentiment(df, sentiment_df)
    except Exception as exc:
        print(f"[WARN] Could not load Fear & Greed data ({exc}). Falling back to technical-only.")
        df, _ = add_sentiment(df, pd.DataFrame())
    return df


def _generate_signals(
    df: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Run model inference on the Out-of-Sample Validation slice only.

    The validation slice is defined as the last ``(1 - _VAL_START_PCT)``
    rows of the NaN-clean dataset — the same split the optimizer used, so
    no in-sample data contaminates the chart.

    Args:
        df: Full OHLCV + indicator DataFrame (OHLCV columns intact).
        bundle: Loaded .pkl dict with keys: model, features, threshold.

    Returns:
        ``df_val`` — the out-of-sample validation slice with OHLCV intact.
        ``signal_mask`` — boolean Series indexed by ``df_val.index``.
    """
    model = bundle["model"]
    features: list[str] = bundle["features"]
    threshold: float = bundle["threshold"]

    available_features = [f for f in features if f in df.columns]
    missing = set(features) - set(available_features)
    if missing:
        print(f"[WARN] Feature(s) not found in data, skipping: {sorted(missing)}")

    if not available_features:
        raise SystemExit("[ERROR] None of the model features are present in the data.")

    # Drop NaNs only on required columns so the candle chart stays complete
    required_cols = available_features + ["open", "high", "low", "close"]
    df_valid = df.dropna(subset=required_cols).copy()

    # --- Chronological 80/20 split: keep only the validation portion ---
    split_idx = int(len(df_valid) * _VAL_START_PCT)
    df_val = df_valid.iloc[split_idx:].copy()

    val_start_date = df_val.index[0].strftime("%Y-%m-%d") if len(df_val) else "N/A"
    val_end_date = df_val.index[-1].strftime("%Y-%m-%d") if len(df_val) else "N/A"
    print(
        f"Out-of-Sample window : {val_start_date} → {val_end_date} "
        f"({len(df_val)} bars, last {100 * (1 - _VAL_START_PCT):.0f}% of clean data)"
    )

    X_val = df_val[available_features].copy()
    for f in missing:        # pad absent features with 0 (neutral / no signal)
        X_val[f] = 0.0
    X_val = X_val[features]  # restore original training column order
    probas = model.predict_proba(X_val)[:, 1]
    signal_mask = pd.Series(probas >= threshold, index=df_val.index, name="signal")

    n_signals = int(signal_mask.sum())
    print(
        f"Threshold : {threshold:.3f}\n"
        f"Signals   : {n_signals} ({100 * n_signals / max(len(df_val), 1):.1f}%)"
    )
    return df_val, signal_mask


def _build_chart(
    df_val: pd.DataFrame,
    signal_mask: pd.Series,
    config: dict[str, Any],
    bundle: dict[str, Any],
    safe_symbol: str,
) -> Any:
    """Build and return a Plotly Figure with candlestick + signal markers."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit(
            "[ERROR] plotly is not installed.\n"
        )

    signal_dates = df_val.index[signal_mask]
    # Place markers just below the low of each signal candle to avoid overlap
    signal_lows = df_val.loc[signal_dates, "low"] * 0.992

    # ---------- Metadata for subtitle row ----------
    pf = config.get("test_profit_factor", "N/A")
    mdd = config.get("test_max_drawdown", "N/A")
    trade_count = config.get("test_trade_count", "N/A")
    strategy = bundle.get("strategy_name", config.get("strategy_name", "Unknown"))
    trained_date = config.get("last_trained", "unknown")[:10]
    atr_tp = config.get("atr_tp_multi", bundle.get("atr_tp_multi", "?"))
    atr_sl = config.get("atr_sl_multi", bundle.get("atr_sl_multi", "?"))
    swing = config.get("swing_period", "?")

    val_start = df_val.index[0].strftime("%Y-%m-%d") if len(df_val) else "N/A"
    val_end = df_val.index[-1].strftime("%Y-%m-%d") if len(df_val) else "N/A"

    subtitle = (
        f"Strategy: <b>{strategy}</b> | "
        f"TP: {atr_tp}\u00d7ATR  SL: {atr_sl}\u00d7ATR  Swing: {swing}d | "
        f"PF: {pf}  MDD: {mdd}  Val Trades: {trade_count} | "
        f"Trained: {trained_date} | "
        f"Val window: {val_start} \u2192 {val_end}"
    )

    fig = go.Figure()

    # --- Candlestick base (validation period only) ---
    fig.add_trace(
        go.Candlestick(
            x=df_val.index,
            open=df_val["open"],
            high=df_val["high"],
            low=df_val["low"],
            close=df_val["close"],
            name="Price (OOS)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            whiskerwidth=0.6,
        )
    )

    # --- Signal markers ---
    if len(signal_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=signal_dates,
                y=signal_lows,
                mode="markers",
                name=f"Signal (\u2265{bundle['threshold']:.2f})",
                marker=dict(
                    symbol="triangle-up",
                    size=14,
                    color="#00e676",
                    line=dict(color="#007a3d", width=1.5),
                ),
                hovertemplate=(
                    "<b>BUY Signal</b><br>"
                    "Date: %{x}<br>"
                    "Low: %{y:.4f}<extra></extra>"
                ),
            )
        )
    else:
        print("[WARN] No signals generated — chart will show price data only.")

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>{safe_symbol}</b> — Out-of-Sample Validation Signals",
            font=dict(size=20),
            x=0.5,
            xanchor="center",
        ),
        annotations=[
            dict(
                text=subtitle,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.055,
                showarrow=False,
                font=dict(size=11, color="#aaaaaa"),
                xanchor="center",
            )
        ],
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
            gridcolor="#2a2a4a",
        ),
        yaxis=dict(
            title="Price (USDT)",
            side="right",
            gridcolor="#2a2a4a",
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(l=40, r=60, t=110, b=40),
        hovermode="x unified",
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an Out-of-Sample Validation signal diagnostic chart "
            "for a trained symbol model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python -m tools.visualize_val_signals BTC_USDT",
    )
    parser.add_argument(
        "symbol",
        type=str,
        help="Trading pair symbol, e.g. BTC_USDT or BTC/USDT",
    )
    args = parser.parse_args()

    safe_symbol = _normalize_symbol(args.symbol)
    base_dir = get_project_root()
    model_dir = base_dir / "data" / "models" / safe_symbol

    print(f"\n{'=' * 60}")
    print(f"  OOS Validation Signal Diagnostic — {safe_symbol}")
    print(f"{'=' * 60}\n")

    config = _load_config(model_dir, safe_symbol)
    bundle = _load_bundle(model_dir, safe_symbol)

    print("\nPreparing data…")
    df = _prepare_dataframe(safe_symbol)

    print("Running OOS inference…")
    df_val, signal_mask = _generate_signals(df, bundle)

    print("\nBuilding chart…")
    fig = _build_chart(df_val, signal_mask, config, bundle, safe_symbol)

    plots_dir = base_dir / "data" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{safe_symbol}_val_signals_chart.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")

    print(f"\n{'=' * 60}")
    print("[OK] Chart saved successfully.")
    print(f"     {out_path.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
