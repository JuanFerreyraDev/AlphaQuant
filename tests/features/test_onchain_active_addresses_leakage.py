"""Leakage verification tests for add_onchain_active_addresses.

Verifies that every sub-daily bar inherits the onchain_active_addresses
value from a FULLY CLOSED daily bar that is at least 2 calendar days old —
never from the current or immediately preceding day.

Conservative shift decision (+2 days, not +1)
---------------------------------------------
Blockchain.com does not publish a latency SLA for the n-unique-addresses
endpoint.  Without a documented guarantee, a +1 day shift (the minimum
for any closed-day aggregate) cannot be relied upon: the aggregation
pipeline may not complete by D+1 00:00 UTC.  A conservative +2 day shift
is applied instead, assuming up to 24 hours of aggregation delay beyond
day close with no empirical verification of the actual publish time.
This accepts one extra day of signal lag in exchange for a hard
leakage-free guarantee.  Revisit if Blockchain.com publishes an official
SLA in future.

Boundary semantics under +2d shift
------------------------------------
  * Sub-daily bars during day D       → value from day D-2
  * Bar at D 00:00 UTC (midnight)     → value from day D-2  (NOT D-1)
  * Bar at (D+1) 00:00 UTC            → value from day D-1
  * First 2 calendar days of bars     → NaN  (shift is confirmed real)
  * Bar at dates[0]+2d 00:00          → non-NaN (no off-by-one)

Synthetic dataset granularity note
------------------------------------
The NaN early-bar checks use a small set of hand-picked timestamps
(not a complete 4h or 1h grid) to keep the fixture minimal and focused.
This is intentional: the leakage property depends only on the shift
arithmetic, not on the density of sub-daily bars.  A full grid would add
no additional coverage for the temporal invariant being tested.

Each test_ function is independently collectable by pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.brain.features import add_onchain_active_addresses


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _build_onchain_df(n_days: int = 120) -> pd.DataFrame:
    """120-day daily onchain series with distinct integer values per day."""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    values = 100_000.0 + np.arange(n_days)
    return pd.DataFrame({"onchain_active_addresses": values}, index=dates)


def _make_sub_df(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "open":   np.linspace(28000.0, 28100.0, n),
            "close":  np.linspace(28020.0, 28120.0, n),
            "volume": np.ones(n) * 500.0,
        },
        index=pd.DatetimeIndex(timestamps),
    )


# ---------------------------------------------------------------------------
# Anchor values (reused across tests)
# Day indices into the 120-day series:
#   50 → 2023-02-20 → val = 100_050  (D-2)
#   51 → 2023-02-21 → val = 100_051  (D-1)
#   52 → 2023-02-22 → val = 100_052  (D)
#   53 → 2023-02-23 → val = 100_053  (D+1)
# ---------------------------------------------------------------------------
_DAY_D       = pd.Timestamp("2023-02-22")
_DAY_D_PLUS1 = pd.Timestamp("2023-02-23")
_VAL_D_MINUS2 = 100_050.0
_VAL_D_MINUS1 = 100_051.0
_VAL_D        = 100_052.0


# ---------------------------------------------------------------------------
# Test 1 — add_onchain_active_addresses succeeds on valid input
# ---------------------------------------------------------------------------

def test_add_onchain_returns_true_and_column() -> None:
    df_onchain = _build_onchain_df()
    df_sub = _make_sub_df([_DAY_D + pd.Timedelta(hours=4)])
    df_out, ok = add_onchain_active_addresses(df_sub, df_onchain)
    assert ok is True
    assert "onchain_active_addresses" in df_out.columns


# ---------------------------------------------------------------------------
# Test 2 — All bars during day D inherit val[D-2], not D-1 or D
# ---------------------------------------------------------------------------

def test_bars_during_day_d_inherit_d_minus2() -> None:
    df_onchain = _build_onchain_df()
    timestamps = [
        _DAY_D + pd.Timedelta(hours=0),   # midnight of day D
        _DAY_D + pd.Timedelta(hours=4),
        _DAY_D + pd.Timedelta(hours=8),
        _DAY_D + pd.Timedelta(hours=12),
        _DAY_D + pd.Timedelta(hours=16),
        _DAY_D + pd.Timedelta(hours=20),
    ]
    df_sub = _make_sub_df(timestamps)
    df_out, _ = add_onchain_active_addresses(df_sub, df_onchain)

    for ts in timestamps:
        got = df_out.loc[ts, "onchain_active_addresses"]
        assert abs(got - _VAL_D_MINUS2) < 1e-6, (
            f"Bar at {ts}: expected val[D-2]={_VAL_D_MINUS2:.0f}, got {got:.0f}"
        )


# ---------------------------------------------------------------------------
# Test 3 — CRITICAL: bar at day D midnight must NOT see val[D-1]
#
# This is the core assertion that distinguishes +2d from +1d shift.
# Under a +1d shift, the midnight bar at day D would see val[D-1].
# Under the correct +2d shift, it must see val[D-2].
# ---------------------------------------------------------------------------

def test_midnight_bar_does_not_leak_d_minus1() -> None:
    """Bar at day D 00:00 must see val[D-2], NOT val[D-1] (+2d vs +1d)."""
    df_onchain = _build_onchain_df()
    midnight_ts = _DAY_D  # 2023-02-22 00:00
    df_sub = _make_sub_df([midnight_ts])
    df_out, _ = add_onchain_active_addresses(df_sub, df_onchain)

    got = df_out.loc[midnight_ts, "onchain_active_addresses"]
    assert abs(got - _VAL_D_MINUS2) < 1e-6, (
        f"SHIFT TOO SMALL: bar at day D midnight saw val[D-1]={_VAL_D_MINUS1:.0f} "
        f"or val[D]={_VAL_D:.0f}. "
        f"+2d shift must deliver val[D-2]={_VAL_D_MINUS2:.0f}, got {got:.0f}"
    )
    # Also confirm it did NOT receive val[D-1] (the +1d-shift value)
    assert abs(got - _VAL_D_MINUS1) > 1e-6, (
        f"Received val[D-1] — this is what +1d shift would give. "
        f"+2d shift required."
    )


# ---------------------------------------------------------------------------
# Test 4 — Bar at D+1 midnight sees val[D-1]  (shift boundary crosses)
# ---------------------------------------------------------------------------

def test_d_plus1_midnight_sees_d_minus1() -> None:
    df_onchain = _build_onchain_df()
    ts = _DAY_D_PLUS1  # 2023-02-23 00:00
    df_sub = _make_sub_df([ts])
    df_out, _ = add_onchain_active_addresses(df_sub, df_onchain)

    got = df_out.loc[ts, "onchain_active_addresses"]
    assert abs(got - _VAL_D_MINUS1) < 1e-6, (
        f"Bar at D+1 midnight: expected val[D-1]={_VAL_D_MINUS1:.0f}, got {got:.0f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — First 2 calendar days of bars are NaN (shift is real)
# ---------------------------------------------------------------------------

def test_first_two_days_are_nan() -> None:
    """Bars before dates[0]+2d must be NaN — confirms the shift exists."""
    df_onchain = _build_onchain_df()

    # These timestamps all fall within the first 2 calendar days (before 2023-01-03)
    early_timestamps = [
        pd.Timestamp("2023-01-01 00:00:00"),
        pd.Timestamp("2023-01-01 04:00:00"),
        pd.Timestamp("2023-01-01 20:00:00"),
        pd.Timestamp("2023-01-02 12:00:00"),
    ]
    df_early = _make_sub_df(early_timestamps)
    df_out, _ = add_onchain_active_addresses(df_early, df_onchain)

    for ts in early_timestamps:
        val = df_out.loc[ts, "onchain_active_addresses"]
        assert np.isnan(val), (
            f"Bar at {ts} should be NaN (within +2d shift window), got {val:.0f}"
        )


# ---------------------------------------------------------------------------
# Test 6 — First bar at dates[0]+2d is non-NaN (no off-by-one in shift)
# ---------------------------------------------------------------------------

def test_first_valid_bar_at_shift_boundary_is_non_nan() -> None:
    """Bar at 2023-01-03 00:00 (dates[0] + 2d) must be the first non-NaN."""
    df_onchain = _build_onchain_df()
    first_valid_ts = pd.Timestamp("2023-01-03 00:00:00")
    df_sub = _make_sub_df([first_valid_ts])
    df_out, _ = add_onchain_active_addresses(df_sub, df_onchain)

    val = df_out.loc[first_valid_ts, "onchain_active_addresses"]
    assert not np.isnan(val), (
        f"Bar at {first_valid_ts} (dates[0]+2d) should be non-NaN, got NaN. "
        "Off-by-one in shift?"
    )
    # Value must be the first daily entry (100_000.0)
    assert abs(val - 100_000.0) < 1e-6, (
        f"Expected 100000.0 (first daily value), got {val:.0f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Empty onchain data returns (df, False)
# ---------------------------------------------------------------------------

def test_empty_onchain_returns_false() -> None:
    df_sub = _make_sub_df([_DAY_D + pd.Timedelta(hours=4)])
    df_out, ok = add_onchain_active_addresses(df_sub, pd.DataFrame())
    assert ok is False
    assert "onchain_active_addresses" not in df_out.columns


# ---------------------------------------------------------------------------
# Test 8 — Missing column in onchain df returns (df, False)
# ---------------------------------------------------------------------------

def test_wrong_column_name_returns_false() -> None:
    df_wrong = pd.DataFrame(
        {"wrong_col": [100_000.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2023-01-01")]),
    )
    df_sub = _make_sub_df([_DAY_D + pd.Timedelta(hours=4)])
    df_out, ok = add_onchain_active_addresses(df_sub, df_wrong)
    assert ok is False
    assert "onchain_active_addresses" not in df_out.columns
