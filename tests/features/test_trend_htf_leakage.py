"""Leakage verification tests for add_trend_htf.

Verifies that every sub-daily bar inherits the trend_htf value from the
MOST RECENT FULLY CLOSED daily bar — never the daily bar that is still
in progress on the same calendar day.

Each test_ function is independently collectable by pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from src.brain.features import add_trend_htf


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _build_daily_df(n_days: int = 300) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """Return (df_1d, dates_1d, close_1d) with monotonic closes."""
    dates_1d = pd.date_range("2023-01-01", periods=n_days, freq="D")
    close_1d = 100.0 + np.arange(n_days)
    df_1d = pd.DataFrame(
        {
            "open":   close_1d - 0.5,
            "high":   close_1d + 2.0,
            "low":    close_1d - 2.0,
            "close":  close_1d,
            "volume": np.ones(n_days) * 1000.0,
        },
        index=dates_1d,
    )
    return df_1d, dates_1d, close_1d


def _precompute_expected(df_1d: pd.DataFrame) -> pd.Series:
    """Compute trend_htf WITHOUT the index shift (reference values only)."""
    df_check = df_1d.copy()
    df_check["ema_200_1d"] = ta.ema(df_check["close"], length=200)
    df_check["trend_htf_expected"] = (
        df_check["close"] - df_check["ema_200_1d"]
    ) / df_check["ema_200_1d"]
    return df_check["trend_htf_expected"]


# ---------------------------------------------------------------------------
# Test 1 — add_trend_htf succeeds on valid input
# ---------------------------------------------------------------------------

def test_add_trend_htf_returns_true() -> None:
    """add_trend_htf must return (df, True) when 1d data is valid."""
    df_1d, dates_1d, close_1d = _build_daily_df()
    n_sub = 3
    df_sub = pd.DataFrame(
        {"close": [350.0, 351.0, 352.0]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2023-09-09 00:00"),
            pd.Timestamp("2023-09-09 04:00"),
            pd.Timestamp("2023-09-10 00:00"),
        ]),
    )
    _, ok = add_trend_htf(df_sub, df_1d)
    assert ok is True, "add_trend_htf returned False on valid input"


# ---------------------------------------------------------------------------
# Test 2 — Per-day values are distinct and EMA200 is warmed up
# ---------------------------------------------------------------------------

def test_anchor_values_distinct_and_non_nan() -> None:
    """Verify the test fixture itself: EMA200 warmed up, values distinct."""
    df_1d, dates_1d, _ = _build_daily_df()
    expected = _precompute_expected(df_1d)

    val_A = expected.iloc[250]
    val_B = expected.iloc[251]
    val_C = expected.iloc[252]

    assert not np.isnan(val_A), "val_A is NaN — EMA200 not warmed up at day 250"
    assert not np.isnan(val_B), "val_B is NaN"
    assert not np.isnan(val_C), "val_C is NaN"
    assert abs(val_A - val_B) > 1e-8, "val_A and val_B are not distinct"
    assert abs(val_B - val_C) > 1e-8, "val_B and val_C are not distinct"


# ---------------------------------------------------------------------------
# Test 3 — Bars DURING a day inherit the PRIOR closed day's value (no leak)
# ---------------------------------------------------------------------------

def test_bars_during_day_inherit_prior_closed_bar() -> None:
    """Bars opening during day B must see val_A, never the open bar B."""
    df_1d, dates_1d, _ = _build_daily_df()
    expected = _precompute_expected(df_1d)
    val_A = expected.iloc[250]

    # Day B = 2023-09-09 (index 251). Bar B is still open during this day.
    # All 4h bars from 00:00 through 20:00 must inherit val_A (day 250).
    sub_ts = [
        pd.Timestamp("2023-09-09 00:00:00"),
        pd.Timestamp("2023-09-09 04:00:00"),
        pd.Timestamp("2023-09-09 12:00:00"),
        pd.Timestamp("2023-09-09 20:00:00"),
    ]
    n = len(sub_ts)
    df_sub = pd.DataFrame(
        {"close": np.linspace(350.0, 351.0, n)},
        index=pd.DatetimeIndex(sub_ts),
    )
    df_merged, _ = add_trend_htf(df_sub, df_1d)

    for ts in sub_ts:
        got = df_merged.loc[ts, "trend_htf"]
        assert abs(got - val_A) < 1e-10, (
            f"LEAK at {ts}: bar during day B saw {got:+.8f} "
            f"(expected val_A={val_A:+.8f} from prior closed bar)"
        )


# ---------------------------------------------------------------------------
# Test 4 — Bar AT midnight (day close boundary) sees the just-closed bar
# ---------------------------------------------------------------------------

def test_midnight_bar_sees_just_closed_daily_bar() -> None:
    """Bar at 2023-09-10 00:00 must see val_B (bar B just closed)."""
    df_1d, _, _ = _build_daily_df()
    expected = _precompute_expected(df_1d)
    val_B = expected.iloc[251]  # day index 251 = 2023-09-09

    midnight_ts = pd.Timestamp("2023-09-10 00:00:00")
    df_sub = pd.DataFrame({"close": [351.0]}, index=pd.DatetimeIndex([midnight_ts]))
    df_merged, _ = add_trend_htf(df_sub, df_1d)

    got = df_merged.loc[midnight_ts, "trend_htf"]
    assert abs(got - val_B) < 1e-10, (
        f"Midnight bar at {midnight_ts} saw {got:+.8f}; "
        f"expected val_B={val_B:+.8f} (bar B just closed)"
    )


# ---------------------------------------------------------------------------
# Test 5 — Full boundary sequence: day B slots → val_A, then day C → val_B
# ---------------------------------------------------------------------------

def test_full_boundary_sequence() -> None:
    """End-to-end: 9 timestamps across two day boundaries, all correct."""
    df_1d, dates_1d, _ = _build_daily_df()
    expected = _precompute_expected(df_1d)
    val_A = expected.iloc[250]
    val_B = expected.iloc[251]
    val_C = expected.iloc[252]

    sub_ts = [
        # During day B (Sep 9) — must see val_A
        pd.Timestamp("2023-09-09 00:00:00"),
        pd.Timestamp("2023-09-09 04:00:00"),
        pd.Timestamp("2023-09-09 12:00:00"),
        pd.Timestamp("2023-09-09 20:00:00"),
        # Midnight: bar B just closed — must see val_B
        pd.Timestamp("2023-09-10 00:00:00"),
        # During day C (Sep 10) — must see val_B
        pd.Timestamp("2023-09-10 04:00:00"),
        pd.Timestamp("2023-09-10 12:00:00"),
        pd.Timestamp("2023-09-10 20:00:00"),
        # Midnight: bar C just closed — must see val_C
        pd.Timestamp("2023-09-11 00:00:00"),
    ]
    expected_per_ts = [val_A, val_A, val_A, val_A, val_B, val_B, val_B, val_B, val_C]

    n = len(sub_ts)
    df_sub = pd.DataFrame(
        {"close": np.linspace(350.0, 353.0, n)},
        index=pd.DatetimeIndex(sub_ts),
    )
    df_merged, _ = add_trend_htf(df_sub, df_1d)

    for ts, exp in zip(sub_ts, expected_per_ts):
        got = df_merged.loc[ts, "trend_htf"]
        assert abs(got - exp) < 1e-10, (
            f"LEAK at {ts}: expected {exp:+.8f}, got {got:+.8f}"
        )


# ---------------------------------------------------------------------------
# Test 6 — First bar (before any daily close + shift) must be NaN
# ---------------------------------------------------------------------------

def test_first_bar_is_nan_before_shift_window() -> None:
    """Bar at 2023-01-01 00:00 must be NaN — no closed daily bar yet available."""
    df_1d, _, _ = _build_daily_df()

    first_ts = pd.Timestamp("2023-01-01 00:00:00")
    df_sub = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex([first_ts]))
    df_merged, _ = add_trend_htf(df_sub, df_1d)

    val = df_merged.loc[first_ts, "trend_htf"]
    assert np.isnan(val), (
        f"First bar should be NaN (shift is real), but got {val:+.8f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Empty daily data returns (df, False)
# ---------------------------------------------------------------------------

def test_empty_daily_returns_false() -> None:
    """add_trend_htf must degrade gracefully when daily df is empty."""
    df_sub = pd.DataFrame(
        {"close": [350.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2023-09-09 04:00")]),
    )
    df_out, ok = add_trend_htf(df_sub, pd.DataFrame())
    assert ok is False, "Expected False when daily data is empty"
    assert "trend_htf" not in df_out.columns
