"""Leakage verification test for add_trend_htf.

Verifies that every sub-daily bar inherits the trend_htf value from the
MOST RECENT FULLY CLOSED daily bar — never the daily bar that is still
in progress on the same calendar day.

Run: python3 tools/diagnostics/test_trend_htf_leakage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.brain.features import add_trend_htf


def main() -> None:
    print("=" * 70)
    print("LEAKAGE VERIFICATION: add_trend_htf merge_asof temporal ordering")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1 — Build mock daily data with DISTINCT close values per day.
    # Each day gets a unique close (100 + day_number) so after EMA200
    # warms up, trend_htf will be distinct and traceable per day.
    # ------------------------------------------------------------------
    n_days = 300
    dates_1d = pd.date_range("2023-01-01", periods=n_days, freq="D")
    close_1d = 100.0 + np.arange(n_days)  # monotonic 100, 101, 102, ...
    df_1d = pd.DataFrame(
        {
            "open": close_1d - 0.5,
            "high": close_1d + 2.0,
            "low": close_1d - 2.0,
            "close": close_1d,
            "volume": np.ones(n_days) * 1000,
        },
        index=dates_1d,
    )

    # Pre-compute the expected per-day trend_htf (using the same formula
    # as add_trend_htf, WITHOUT the index shift) so we can check which
    # value ends up on which sub-daily bar.
    import pandas_ta as ta

    df_1d_check = df_1d.copy()
    df_1d_check["ema_200_1d"] = ta.ema(df_1d_check["close"], length=200)
    df_1d_check["trend_htf_expected"] = (
        df_1d_check["close"] - df_1d_check["ema_200_1d"]
    ) / df_1d_check["ema_200_1d"]

    # Pick 3 days where EMA200 is warmed up and values are well separated.
    # Day 250 → index 250 → close=350 → date 2023-09-08
    # Day 251 → index 251 → close=351 → date 2023-09-09
    # Day 252 → index 252 → close=352 → date 2023-09-10
    day_idx_A = 250  # Closed daily bar A (becomes available at midnight 9/9)
    day_idx_B = 251  # Closed daily bar B (becomes available at midnight 9/10)
    day_idx_C = 252  # Closed daily bar C (becomes available at midnight 9/11)

    val_A = df_1d_check["trend_htf_expected"].iloc[day_idx_A]
    val_B = df_1d_check["trend_htf_expected"].iloc[day_idx_B]
    val_C = df_1d_check["trend_htf_expected"].iloc[day_idx_C]

    print(f"\nDistinct per-day trend_htf values (after EMA200 warmup):")
    print(f"  Bar A (day {day_idx_A}, {dates_1d[day_idx_A].date()}, close={close_1d[day_idx_A]:.0f}): {val_A:+.8f}")
    print(f"  Bar B (day {day_idx_B}, {dates_1d[day_idx_B].date()}, close={close_1d[day_idx_B]:.0f}): {val_B:+.8f}")
    print(f"  Bar C (day {day_idx_C}, {dates_1d[day_idx_C].date()}, close={close_1d[day_idx_C]:.0f}): {val_C:+.8f}")

    distinct_ok = (
        not np.isnan(val_A) and not np.isnan(val_B) and not np.isnan(val_C)
        and abs(val_A - val_B) > 1e-8
        and abs(val_B - val_C) > 1e-8
    )
    assert distinct_ok, "Test setup error: expected distinct per-day values"
    print("  ✓ Per-day values are distinct and non-NaN\n")

    # ------------------------------------------------------------------
    # Step 2 — Build 4h sub-daily candles at KEY TIMESTAMPS around the
    # daily boundaries. We want to test:
    #   - Bars *during* day B (Sep 9): must see bar A only (bar B open!)
    #   - Midnight between B/C (Sep 10 00:00): bar B has just closed
    #   - Bars *during* day C (Sep 10): must see bar B only (bar C open!)
    # ------------------------------------------------------------------
    # Day B = 2023-09-09  (index 251, daily OPEN at 00:00, CLOSE at 2023-09-10 00:00)
    # Day C = 2023-09-10  (index 252, daily OPEN at 00:00, CLOSE at 2023-09-11 00:00)
    sub_ts = [
        # --- During day B (September 9) — bar B is OPEN, must NOT leak val_B
        pd.Timestamp("2023-09-09 00:00:00"),  # 4h bar 0-4h: bar B *just opened*
        pd.Timestamp("2023-09-09 04:00:00"),  # 4h bar 4-8h: bar B still open
        pd.Timestamp("2023-09-09 12:00:00"),  # 4h bar 12-16h: bar B still open
        pd.Timestamp("2023-09-09 20:00:00"),  # 4h bar 20-24h: last 4h of bar B
        # --- Midnight boundary: bar B CLOSES, val_B becomes available
        pd.Timestamp("2023-09-10 00:00:00"),  # 4h bar at midnight: bar B just closed
        # --- During day C (September 10) — bar C is OPEN, must NOT leak val_C
        pd.Timestamp("2023-09-10 04:00:00"),
        pd.Timestamp("2023-09-10 12:00:00"),
        pd.Timestamp("2023-09-10 20:00:00"),
        # --- Next midnight: bar C just closed, val_C available
        pd.Timestamp("2023-09-11 00:00:00"),
    ]
    n_sub = len(sub_ts)
    df_4h = pd.DataFrame(
        {
            "open": np.linspace(350.0, 353.0, n_sub),
            "high": np.linspace(351.0, 354.0, n_sub),
            "low": np.linspace(349.0, 352.0, n_sub),
            "close": np.linspace(350.5, 353.5, n_sub),
            "volume": np.ones(n_sub) * 500,
        },
        index=pd.DatetimeIndex(sub_ts),
    )

    # ------------------------------------------------------------------
    # Step 3 — Run add_trend_htf and inspect the resulting assignments
    # ------------------------------------------------------------------
    df_merged, ok = add_trend_htf(df_4h, df_1d)
    assert ok is True, "add_trend_htf returned False unexpectedly"

    print(f"{'Sub-daily timestamp':<25} {'Expected':>12} {'Got':>12} {'Match':>7}")
    print("-" * 60)

    errors: list[str] = []

    # Expected mapping:
    # 2023-09-09 00:00 → val_A (Bar A: closed at 2023-09-09 00:00; Bar B just opened)
    # 2023-09-09 04:00 → val_A
    # 2023-09-09 12:00 → val_A
    # 2023-09-09 20:00 → val_A
    # 2023-09-10 00:00 → val_B (Bar B: just closed at 2023-09-10 00:00)
    # 2023-09-10 04:00 → val_B
    # 2023-09-10 12:00 → val_B
    # 2023-09-10 20:00 → val_B
    # 2023-09-11 00:00 → val_C (Bar C: just closed at 2023-09-11 00:00)
    expected_per_ts = [val_A, val_A, val_A, val_A, val_B, val_B, val_B, val_B, val_C]

    for ts, expected in zip(sub_ts, expected_per_ts):
        got = df_merged.loc[ts, "trend_htf"]
        match = abs(got - expected) < 1e-10
        status = "✓" if match else "✗ LEAK!"
        print(
            f"{str(ts):<25} {expected:>+12.8f} {got:>+12.8f} {status:>7}"
        )
        if not match:
            errors.append(
                f"LEAK at {ts}: expected {expected:+.8f} (prior closed bar), "
                f"got {got:+.8f} — a still-open daily bar leaked into the merge!"
            )

    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 4 — Also verify the FIRST bar in the range (before any daily
    # bar has closed + shifted) is NaN — this confirms the shift is real
    # and we're not accidentally using pre-close data.
    # ------------------------------------------------------------------
    first_4h = pd.Timestamp("2023-01-01 00:00:00")
    df_4h_first = pd.DataFrame(
        {"close": [100.0]}, index=pd.DatetimeIndex([first_4h])
    )
    df_merged_first, _ = add_trend_htf(df_4h_first, df_1d)
    first_trend = df_merged_first.loc[first_4h, "trend_htf"]
    print(f"\nFirst 4h bar ({first_4h.date()}) trend_htf is NaN:", np.isnan(first_trend))
    if not np.isnan(first_trend):
        errors.append(
            f"First bar should be NaN (no closed daily bar available yet), "
            f"but got {first_trend:+.8f}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if errors:
        print("\n❌ FAILURE — leakage detected!")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("\n✅ PASS — zero leakage. Every sub-daily bar inherits ONLY from "
              "fully closed daily bars (closed at or before the bar's open time).")
        print("   • Bars during day X → inherit day (X-1) trend_htf")
        print("   • Bars at midnight → inherit the just-closed day's trend_htf")
        print("   • No bar ever sees the same-calendar-day still-open daily bar")


if __name__ == "__main__":
    main()
