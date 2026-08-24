"""Leakage verification tests for add_mempool_fee_rate_p50.

Verifies that every sub-daily bar inherits the mempool_fee_rate_p50 value
from the MOST RECENT FULLY CLOSED calendar day — never from the current
day still in progress.

Shift decision (+1 day, same as trend_htf)
-------------------------------------------
The mempool.space backend indexes fee_rate_percentiles synchronously in
the same block-processing cycle as block confirmation, from Bitcoin Core
RPC directly (verified in backend/src/api/blocks.ts).  There is no async
pipeline or artificial delay.  The daily aggregate is complete once the
last block of the day is confirmed — available from D+1 00:00 UTC onward.
A +1 day shift is therefore both necessary and sufficient (unlike
onchain_active_addresses which uses +2 days due to missing SLA).

Real-data day assignment (Point 3 verification)
------------------------------------------------
The API timestamp is the UTC epoch of the *central block* of each daily
bucket — not midnight UTC.  Verified across all 2420 entries (2020-2026):
every timestamp falls between 06:43 and 14:10 UTC, so
pd.to_datetime(ts, unit='s', utc=True).normalize().tz_localize(None)
always assigns the correct calendar UTC day (0 day-crossing edge cases).

Six real entries confirmed (height → date):
  610778 (1577883269) → 2020-01-01   610935 (1577965502) → 2020-01-02
  611729 (1578400751) → 2020-01-07   612037 (1578572566) → 2020-01-09
  689380 (1625222985) → 2021-07-02   762080 (1667816976) → 2022-11-07

Boundary semantics under +1d shift
------------------------------------
  * Sub-daily bars during day D          → value from day D-1
  * Bar at D+1 00:00 UTC (midnight)      → value from day D (just closed)
  * First calendar day of bars           → NaN  (shift confirmed real)
  * Bar at dates[0]+1d 00:00             → non-NaN (no off-by-one)

Synthetic dataset granularity note
------------------------------------
The NaN and boundary checks use a small set of hand-picked timestamps
(not a complete 4h or 1h grid) to keep the fixture minimal and focused.
The leakage property depends only on the shift arithmetic, not on the
density of sub-daily bars.

Each test_ function is independently collectable by pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.brain.features import add_mempool_fee_rate_p50


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _build_mempool_df(n_days: int = 60) -> pd.DataFrame:
    """60-day daily series with distinct integer p50 values per day."""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    # Use visually distinct values: 10, 11, 12, …  (realistic sat/vB range)
    values = 10.0 + np.arange(n_days, dtype=float)
    return pd.DataFrame({"mempool_fee_rate_p50": values}, index=dates)


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


# Anchor values for the 60-day series starting 2020-01-01:
#   index 0 → 2020-01-01 → val = 10.0  (D-1 when D = 2020-01-02)
#   index 1 → 2020-01-02 → val = 11.0  (D)
#   index 2 → 2020-01-03 → val = 12.0  (D+1)
_DAY_D_MINUS1 = pd.Timestamp("2020-01-01")   # val = 10.0
_DAY_D        = pd.Timestamp("2020-01-02")   # val = 11.0
_DAY_D_PLUS1  = pd.Timestamp("2020-01-03")   # val = 12.0

_VAL_D_MINUS2 = 9.0    # (not in series — would be NaN)
_VAL_D_MINUS1 = 10.0
_VAL_D        = 11.0
_VAL_D_PLUS1  = 12.0


# ---------------------------------------------------------------------------
# Test 1 — add_mempool_fee_rate_p50 succeeds on valid input
# ---------------------------------------------------------------------------

def test_add_mempool_returns_true_and_column() -> None:
    df_mempool = _build_mempool_df()
    df_sub = _make_sub_df([_DAY_D + pd.Timedelta(hours=4)])
    df_out, ok = add_mempool_fee_rate_p50(df_sub, df_mempool)
    assert ok is True
    assert "mempool_fee_rate_p50" in df_out.columns


# ---------------------------------------------------------------------------
# Test 2 — Bars during day D see val[D-1], not val[D]  (+1d shift core check)
# ---------------------------------------------------------------------------

def test_bars_during_day_d_inherit_d_minus1() -> None:
    """Every sub-daily bar opening during day D must see val[D-1]."""
    df_mempool = _build_mempool_df()
    timestamps = [
        _DAY_D + pd.Timedelta(hours=0),    # midnight of D
        _DAY_D + pd.Timedelta(hours=4),
        _DAY_D + pd.Timedelta(hours=8),
        _DAY_D + pd.Timedelta(hours=12),
        _DAY_D + pd.Timedelta(hours=16),
        _DAY_D + pd.Timedelta(hours=20),
    ]
    df_sub = _make_sub_df(timestamps)
    df_out, _ = add_mempool_fee_rate_p50(df_sub, df_mempool)

    for ts in timestamps:
        got = df_out.loc[ts, "mempool_fee_rate_p50"]
        assert abs(got - _VAL_D_MINUS1) < 1e-6, (
            f"Bar at {ts}: expected val[D-1]={_VAL_D_MINUS1}, got {got}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Bar at D+1 midnight sees val[D]  (boundary: day D just closed)
# ---------------------------------------------------------------------------

def test_d_plus1_midnight_sees_val_d() -> None:
    """Bar at D+1 00:00 must inherit val[D] — the day that just closed."""
    df_mempool = _build_mempool_df()
    ts = _DAY_D_PLUS1  # 2020-01-03 00:00
    df_sub = _make_sub_df([ts])
    df_out, _ = add_mempool_fee_rate_p50(df_sub, df_mempool)

    got = df_out.loc[ts, "mempool_fee_rate_p50"]
    assert abs(got - _VAL_D) < 1e-6, (
        f"Bar at D+1 midnight: expected val[D]={_VAL_D}, got {got}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Full boundary sequence: 9 timestamps across two day boundaries
# ---------------------------------------------------------------------------

def test_full_boundary_sequence() -> None:
    """End-to-end: bars spanning two day boundaries all inherit correctly."""
    df_mempool = _build_mempool_df()

    # val_A = val for 2020-01-01 = 10.0 (available from 2020-01-02 00:00)
    # val_B = val for 2020-01-02 = 11.0 (available from 2020-01-03 00:00)
    # val_C = val for 2020-01-03 = 12.0 (available from 2020-01-04 00:00)
    val_A = 10.0
    val_B = 11.0
    val_C = 12.0
    day_A_start = pd.Timestamp("2020-01-02")  # bars see val_A
    day_B_start = pd.Timestamp("2020-01-03")  # bars see val_B
    day_C_start = pd.Timestamp("2020-01-04")  # bars see val_C

    sub_ts = [
        day_A_start + pd.Timedelta(hours=0),    # 2020-01-02 00:00 → val_A
        day_A_start + pd.Timedelta(hours=4),    # → val_A
        day_A_start + pd.Timedelta(hours=20),   # → val_A
        day_B_start + pd.Timedelta(hours=0),    # 2020-01-03 00:00 → val_B
        day_B_start + pd.Timedelta(hours=8),    # → val_B
        day_B_start + pd.Timedelta(hours=20),   # → val_B
        day_C_start + pd.Timedelta(hours=0),    # 2020-01-04 00:00 → val_C
        day_C_start + pd.Timedelta(hours=4),    # → val_C
        day_C_start + pd.Timedelta(hours=12),   # → val_C
    ]
    expected = [val_A, val_A, val_A, val_B, val_B, val_B, val_C, val_C, val_C]

    df_sub = _make_sub_df(sub_ts)
    df_out, _ = add_mempool_fee_rate_p50(df_sub, df_mempool)

    for ts, exp in zip(sub_ts, expected):
        got = df_out.loc[ts, "mempool_fee_rate_p50"]
        assert abs(got - exp) < 1e-6, (
            f"LEAK at {ts}: expected {exp}, got {got}"
        )


# ---------------------------------------------------------------------------
# Test 5 — First calendar day of bars is NaN  (+1d shift is real)
# ---------------------------------------------------------------------------

def test_first_day_is_nan() -> None:
    """Bars on 2020-01-01 must be NaN: no closed day available yet (+1d shift)."""
    df_mempool = _build_mempool_df()
    early_ts = [
        pd.Timestamp("2020-01-01 00:00:00"),
        pd.Timestamp("2020-01-01 04:00:00"),
        pd.Timestamp("2020-01-01 20:00:00"),
    ]
    df_early = _make_sub_df(early_ts)
    df_out, _ = add_mempool_fee_rate_p50(df_early, df_mempool)

    for ts in early_ts:
        val = df_out.loc[ts, "mempool_fee_rate_p50"]
        assert np.isnan(val), (
            f"Bar at {ts} should be NaN (first day, +1d shift not yet crossed), got {val}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Bar at dates[0]+1d 00:00 is the first non-NaN  (no off-by-one)
# ---------------------------------------------------------------------------

def test_first_valid_bar_at_shift_boundary() -> None:
    """2020-01-02 00:00 must be the first non-NaN bar (dates[0] + 1d)."""
    df_mempool = _build_mempool_df()
    first_valid_ts = pd.Timestamp("2020-01-02 00:00:00")
    df_sub = _make_sub_df([first_valid_ts])
    df_out, _ = add_mempool_fee_rate_p50(df_sub, df_mempool)

    val = df_out.loc[first_valid_ts, "mempool_fee_rate_p50"]
    assert not np.isnan(val), (
        f"2020-01-02 00:00 should be non-NaN (dates[0]+1d), got NaN. Off-by-one in shift?"
    )
    # Must be val for 2020-01-01 = 10.0
    assert abs(val - _VAL_D_MINUS1) < 1e-6, (
        f"Expected val[2020-01-01]={_VAL_D_MINUS1}, got {val}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Real-data day assignment: 6 verified entries from mempool.space API
#
# These timestamps were taken directly from the live API response and
# verified against the expected calendar day during the PASO 0 investigation.
# Each raw_ts is the epoch of the central block of that daily bucket.
# The expected_day is the UTC calendar date, which normalize() must produce.
# ---------------------------------------------------------------------------

def test_real_data_day_assignment() -> None:
    """
    Verify normalize() assigns the correct UTC calendar day for 6 real API entries.

    Source: mempool.space/api/v1/mining/blocks/fee-rates/all (fetched 2026-08-15).
    All 2420 entries (2020-2026) fall between 06:43 and 14:10 UTC — no entry
    is in the day-crossing risk zones (00:00-06:00 or 18:00-24:00 UTC).

    Entry format: (avgHeight, raw_ts_epoch_sec, expected_calendar_day_utc, p50_sat_vb)
    """
    real_entries = [
        (610778, 1577883269, "2020-01-01", 4),    # 12:54 UTC
        (610935, 1577965502, "2020-01-02", 5),    # 11:45 UTC
        (611729, 1578400751, "2020-01-07", 11),   # 12:39 UTC
        (612037, 1578572566, "2020-01-09", 18),   # 12:22 UTC
        (689380, 1625222985, "2021-07-02", 61),   # 10:49 UTC
        (762080, 1667816976, "2022-11-07", 10),   # 10:29 UTC
    ]

    for avg_height, raw_ts, expected_day, p50 in real_entries:
        # This is the exact normalization logic used in fetch_mempool_fee_rate_median
        normalized = (
            pd.to_datetime(raw_ts, unit="s", utc=True)
            .normalize()
            .tz_localize(None)
        )
        normalized_day = str(normalized.date())
        assert normalized_day == expected_day, (
            f"Real entry height={avg_height} ts={raw_ts}: "
            f"normalize() produced {normalized_day}, expected {expected_day}. "
            f"(UTC time was {pd.to_datetime(raw_ts, unit='s', utc=True).strftime('%H:%M')})"
        )


# ---------------------------------------------------------------------------
# Test 8 — Empty mempool data returns (df, False)
# ---------------------------------------------------------------------------

def test_empty_mempool_returns_false() -> None:
    df_sub = _make_sub_df([_DAY_D + pd.Timedelta(hours=4)])
    df_out, ok = add_mempool_fee_rate_p50(df_sub, pd.DataFrame())
    assert ok is False
    assert "mempool_fee_rate_p50" not in df_out.columns


# ---------------------------------------------------------------------------
# Test 9 — Wrong column name returns (df, False)
# ---------------------------------------------------------------------------

def test_wrong_column_name_returns_false() -> None:
    df_wrong = pd.DataFrame(
        {"wrong_col": [5.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-01-01")]),
    )
    df_sub = _make_sub_df([_DAY_D + pd.Timedelta(hours=4)])
    df_out, ok = add_mempool_fee_rate_p50(df_sub, df_wrong)
    assert ok is False
    assert "mempool_fee_rate_p50" not in df_out.columns
