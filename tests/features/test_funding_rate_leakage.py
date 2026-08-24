"""Leakage verification tests for add_funding_rate.

Binance USD-M Futures funding settles every 8 hours: 00:00, 08:00, 16:00 UTC.
Each settlement's timestamp in the CSV is the EXACT moment the rate is
applied / becomes known.  Verifies that every sub-daily bar inherits the
funding rate from the MOST RECENT SETTLEMENT that happened AT OR BEFORE
the bar's open timestamp — never the still-upcoming funding settlement.

Each test_ function is independently collectable by pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.brain.features import add_funding_rate


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _build_funding_df() -> tuple[pd.DataFrame, tuple]:
    """Return (df_funding, (S0, S1, S2, S3, S4)) with distinct values."""
    settlement_dts = [
        pd.Timestamp("2024-01-02 00:00:00"),  # S0
        pd.Timestamp("2024-01-02 08:00:00"),  # S1
        pd.Timestamp("2024-01-02 16:00:00"),  # S2
        pd.Timestamp("2024-01-03 00:00:00"),  # S3
        pd.Timestamp("2024-01-03 08:00:00"),  # S4
    ]
    distinct_vals = np.array([-10e-5, +5e-5, -20e-5, +30e-5, -15e-5])
    df_funding = pd.DataFrame(
        {"funding_rate": distinct_vals},
        index=pd.DatetimeIndex(settlement_dts),
    )
    return df_funding, tuple(distinct_vals)


def _make_sub_df(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {"close": np.linspace(42000.0, 42500.0, n), "atr_14": np.ones(n) * 800.0},
        index=pd.DatetimeIndex(timestamps),
    )


# ---------------------------------------------------------------------------
# Test 1 — add_funding_rate succeeds and produces the output column
# ---------------------------------------------------------------------------

def test_add_funding_rate_returns_true_and_column() -> None:
    df_funding, (S0, *_) = _build_funding_df()
    df_sub = _make_sub_df([pd.Timestamp("2024-01-02 04:00:00")])
    df_out, ok = add_funding_rate(df_sub, df_funding)
    assert ok is True
    assert "funding_rate_current" in df_out.columns


# ---------------------------------------------------------------------------
# Test 2 — Bars before and at S0 (00:00 settlement)
# ---------------------------------------------------------------------------

def test_bars_at_and_before_first_settlement() -> None:
    df_funding, (S0, S1, S2, S3, S4) = _build_funding_df()

    timestamps = [
        pd.Timestamp("2024-01-02 00:00:00"),  # exactly at S0 → S0
        pd.Timestamp("2024-01-02 04:00:00"),  # 4h after S0, before S1 → S0
        pd.Timestamp("2024-01-02 07:00:00"),  # 1h before S1 → S0
        pd.Timestamp("2024-01-02 07:59:59"),  # 1s before S1 → S0 (critical)
    ]
    df_sub = _make_sub_df(timestamps)
    df_out, _ = add_funding_rate(df_sub, df_funding)

    for ts in timestamps:
        got = df_out.loc[ts, "funding_rate_current"]
        assert abs(got - S0) < 1e-12, (
            f"Bar at {ts}: expected S0={S0:+.6f}, got {got:+.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3 — CRITICAL boundary: 1 second before S1 must NOT see S1
# ---------------------------------------------------------------------------

def test_one_second_before_settlement_does_not_leak_next() -> None:
    """The single most important boundary check: T-1s before settlement."""
    df_funding, (S0, S1, *_) = _build_funding_df()

    critical_ts = pd.Timestamp("2024-01-02 07:59:59")
    df_sub = _make_sub_df([critical_ts])
    df_out, _ = add_funding_rate(df_sub, df_funding)

    got = df_out.loc[critical_ts, "funding_rate_current"]
    assert abs(got - S0) < 1e-12, (
        f"CRITICAL BOUNDARY FAILURE: 1s before 08:00 settlement "
        f"got {got:+.6f} (S1={S1:+.6f}), expected S0={S0:+.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Bars at and after each settlement pick up the new value
# ---------------------------------------------------------------------------

def test_bars_at_each_settlement_see_new_value() -> None:
    df_funding, (S0, S1, S2, S3, S4) = _build_funding_df()

    timestamps = [
        pd.Timestamp("2024-01-02 08:00:00"),   # exactly at S1 → S1
        pd.Timestamp("2024-01-02 08:00:01"),   # 1s after S1 → S1
        pd.Timestamp("2024-01-02 12:00:00"),   # 4h after S1 → S1
        pd.Timestamp("2024-01-02 15:59:59"),   # 1s before S2 → S1
        pd.Timestamp("2024-01-02 16:00:00"),   # exactly at S2 → S2
        pd.Timestamp("2024-01-02 20:00:00"),   # after S2 → S2
        pd.Timestamp("2024-01-02 23:59:59"),   # 1s before S3 → S2
        pd.Timestamp("2024-01-03 00:00:00"),   # exactly at S3 → S3
        pd.Timestamp("2024-01-03 04:00:00"),   # after S3 → S3
        pd.Timestamp("2024-01-03 07:59:59"),   # 1s before S4 → S3
        pd.Timestamp("2024-01-03 08:00:00"),   # exactly at S4 → S4
        pd.Timestamp("2024-01-03 12:00:00"),   # after S4 → S4
    ]
    expected = [S1, S1, S1, S1, S2, S2, S2, S3, S3, S3, S4, S4]

    df_sub = _make_sub_df(timestamps)
    df_out, _ = add_funding_rate(df_sub, df_funding)

    for ts, exp in zip(timestamps, expected):
        got = df_out.loc[ts, "funding_rate_current"]
        assert abs(got - exp) < 1e-12, (
            f"Bar at {ts}: expected {exp:+.6f}, got {got:+.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Bar BEFORE any settlement must be NaN
# ---------------------------------------------------------------------------

def test_bar_before_any_settlement_is_nan() -> None:
    df_funding, _ = _build_funding_df()
    pre_ts = pd.Timestamp("2024-01-01 12:00:00")  # 12h before first settlement
    df_sub = _make_sub_df([pre_ts])
    df_out, _ = add_funding_rate(df_sub, df_funding)
    val = df_out.loc[pre_ts, "funding_rate_current"]
    assert pd.isna(val), (
        f"Bar before any settlement should be NaN, got {val:+.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Empty funding data returns (df, False)
# ---------------------------------------------------------------------------

def test_empty_funding_returns_false() -> None:
    df_sub = _make_sub_df([pd.Timestamp("2024-01-02 04:00")])
    df_out, ok = add_funding_rate(df_sub, pd.DataFrame())
    assert ok is False
    assert "funding_rate_current" not in df_out.columns
