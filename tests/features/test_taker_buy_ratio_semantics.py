"""Semantic & temporal correctness tests for add_taker_buy_ratio.

``taker_buy_ratio`` is a NATIVE bar-level feature — it uses only fields
stored *within* the same candle (``taker_buy_base_vol`` and ``volume``).
There is no cross-timeframe merge, no index shift, and no way for
future-bar values to leak through a temporal join.

Three classes of verification, same rigour standard as the leakage tests:
  1. Pointwise semantics: ratio[i] == taker_buy[i] / volume[i], row-isolated.
  2. Bounds & edge cases: values in [0,1], zero-volume → NaN.
  3. Real-data sanity: taker_buy ≤ total volume in every row of the
     downloaded BTC_USDT 4h CSV.

Each test_ function is independently collectable by pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.brain.features import add_taker_buy_ratio
from src.utils.helpers import load_csv_data


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

def _build_synthetic_df() -> pd.DataFrame:
    """Synthetic 1h candles with known per-row taker_buy_ratio values."""
    dts = pd.date_range("2026-01-01 00:00", periods=8, freq="1h")
    volume    = np.array([100.0, 200.0,  50.0, 0.0, 1000.0, 500.0, 250.0,   0.0])
    taker_buy = np.array([ 80.0,  50.0,  50.0, 0.0,   10.0, 250.0,   0.0, 123.0])
    # expected ratios per row:
    #   0: 80/100=0.80  1: 50/200=0.25  2: 50/50=1.00  3: vol=0→NaN
    #   4: 10/1000=0.01 5: 250/500=0.50 6: 0/250=0.00  7: vol=0→NaN
    expected = np.array([0.80, 0.25, 1.00, np.nan, 0.01, 0.50, 0.00, np.nan])
    df = pd.DataFrame(
        {
            "open":  np.linspace(40000, 41000, 8),
            "close": np.linspace(40050, 41050, 8),
            "volume":              volume,
            "taker_buy_base_vol":  taker_buy,
        },
        index=pd.DatetimeIndex(dts, name="timestamp"),
    )
    df.attrs["expected_ratio"] = expected
    return df


# ---------------------------------------------------------------------------
# Test 1 — add_taker_buy_ratio succeeds and produces the output column
# ---------------------------------------------------------------------------

def test_add_taker_buy_ratio_returns_true_and_column() -> None:
    df = _build_synthetic_df()
    df_out, ok = add_taker_buy_ratio(df)
    assert ok is True
    assert "taker_buy_ratio" in df_out.columns


# ---------------------------------------------------------------------------
# Test 2 — Pointwise correctness: every row equals taker/vol exactly
# ---------------------------------------------------------------------------

def test_pointwise_ratio_equals_taker_over_volume() -> None:
    df = _build_synthetic_df()
    expected: np.ndarray = df.attrs["expected_ratio"]
    df_out, _ = add_taker_buy_ratio(df)

    for i, ts in enumerate(df.index):
        exp = expected[i]
        got = df_out.loc[ts, "taker_buy_ratio"]
        if np.isnan(exp):
            assert np.isnan(got), (
                f"Row {i} ({ts}): expected NaN (zero volume), got {got}"
            )
        else:
            assert abs(float(got) - float(exp)) < 1e-12, (
                f"Row {i} ({ts}): expected {exp:.6f}, got {got:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Zero-volume rows produce NaN (no division-by-zero artifacts)
# ---------------------------------------------------------------------------

def test_zero_volume_rows_produce_nan() -> None:
    df = _build_synthetic_df()
    df_out, _ = add_taker_buy_ratio(df)
    nan_rows = int(df_out["taker_buy_ratio"].isna().sum())
    zero_vol_rows = int((df["volume"] == 0).sum())
    assert nan_rows == zero_vol_rows, (
        f"NaN count {nan_rows} != zero-volume count {zero_vol_rows}"
    )


# ---------------------------------------------------------------------------
# Test 4 — All finite values live in [0, 1]
# ---------------------------------------------------------------------------

def test_finite_values_in_unit_interval() -> None:
    df = _build_synthetic_df()
    df_out, _ = add_taker_buy_ratio(df)
    finite = df_out["taker_buy_ratio"].dropna()
    assert float(finite.min()) >= 0.0 - 1e-9, f"ratio below 0: {finite.min()}"
    assert float(finite.max()) <= 1.0 + 1e-9, f"ratio above 1: {finite.max()}"


# ---------------------------------------------------------------------------
# Test 5 — Harness sensitivity: a shift(-1) contamination is detectable
#
# This verifies that the synthetic fixture would catch a look-ahead bug
# implemented as .shift(-1) inside add_taker_buy_ratio.  It is a
# meta-test of the test harness itself.
# ---------------------------------------------------------------------------

def test_harness_detects_shift_leakage() -> None:
    df = _build_synthetic_df()
    expected: np.ndarray = df.attrs["expected_ratio"]

    df_shifted = df.copy()
    df_shifted["taker_buy_base_vol"] = df_shifted["taker_buy_base_vol"].shift(-1)
    df_shifted_out, _ = add_taker_buy_ratio(df_shifted)

    n_contaminated = 0
    for i, ts in enumerate(df.index[:-2]):
        exp = expected[i]
        got = df_shifted_out.loc[ts, "taker_buy_ratio"]
        if np.isfinite(exp) and np.isfinite(got) and abs(got - exp) > 1e-9:
            n_contaminated += 1

    assert n_contaminated >= 4, (
        f"Harness too weak: only {n_contaminated}/6 rows detected shift(-1) leakage"
    )


# ---------------------------------------------------------------------------
# Test 6 — Missing columns return (df, False) without raising
# ---------------------------------------------------------------------------

def test_missing_columns_return_false() -> None:
    df_no_taker = pd.DataFrame(
        {"close": [42000.0], "volume": [100.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01")]),
    )
    df_out, ok = add_taker_buy_ratio(df_no_taker)
    assert ok is False
    assert "taker_buy_ratio" not in df_out.columns


# ---------------------------------------------------------------------------
# Test 7 — Real-data sanity: taker_buy_base_vol <= volume in every row
# ---------------------------------------------------------------------------

def test_real_btc_taker_buy_does_not_exceed_total_volume() -> None:
    """Integrity check against the downloaded BTC_USDT 4h REST CSV.

    Skipped automatically if the CSV was not fetched with --binance-rest
    (i.e. taker_buy_base_vol column is absent).
    """
    try:
        df_real = load_csv_data("BTC_USDT", "4h")
    except Exception as exc:
        pytest.skip(f"BTC_USDT 4h CSV not loadable: {exc}")

    if "taker_buy_base_vol" not in df_real.columns:
        pytest.skip(
            "taker_buy_base_vol column absent — re-download with --binance-rest"
        )

    violations = df_real[
        df_real["taker_buy_base_vol"] > df_real["volume"] + 1e-9
    ]
    assert len(violations) == 0, (
        f"{len(violations)} rows have taker_buy_base_vol > volume — "
        "corrupted download or field off-by-one"
    )


# ---------------------------------------------------------------------------
# Test 8 — Real-data sanity: ratio is finite on >99% of rows
# ---------------------------------------------------------------------------

def test_real_btc_taker_ratio_finite_coverage() -> None:
    """Ratio computation yields finite values on the vast majority of rows."""
    try:
        df_real = load_csv_data("BTC_USDT", "4h")
    except Exception as exc:
        pytest.skip(f"BTC_USDT 4h CSV not loadable: {exc}")

    if "taker_buy_base_vol" not in df_real.columns:
        pytest.skip(
            "taker_buy_base_vol column absent — re-download with --binance-rest"
        )

    df_out, ok = add_taker_buy_ratio(df_real)
    assert ok is True
    finite = df_out["taker_buy_ratio"].dropna()
    coverage = len(finite) / len(df_out)
    assert coverage >= 0.99, (
        f"taker_buy_ratio finite coverage {coverage:.2%} < 99%"
    )
    assert float(finite.min()) >= 0.0, f"ratio below 0: {finite.min()}"
    assert float(finite.max()) <= 1.0, f"ratio above 1: {finite.max()}"
