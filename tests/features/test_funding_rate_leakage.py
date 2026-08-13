"""Leakage verification test for add_funding_rate.

Binance USD-M Futures funding settles every 8 hours: 00:00, 08:00, 16:00 UTC.
Each settlement's timestamp in the CSV is the EXACT moment the rate is
applied / becomes known.  Verifies that every sub-daily bar inherits the
funding rate from the MOST RECENT SETTLEMENT that happened AT OR BEFORE
the bar's open timestamp — never the still-upcoming funding settlement.

Run: python3 tools/diagnostics/test_funding_rate_leakage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.brain.features import add_funding_rate


def main() -> None:
    print("=" * 70)
    print("LEAKAGE VERIFICATION: add_funding_rate merge_asof temporal ordering")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1 — Build synthetic funding rate history with DISTINCT values
    #   at every 8-hour settlement mark so we can trace exactly which
    #   settlement was joined to each sub-daily bar.
    #
    #   Funding liquidation times (Binance convention):
    #     2024-01-02 00:00  → funding_val_0 (val = -0.00010)
    #     2024-01-02 08:00  → funding_val_8 (val = +0.00005)
    #     2024-01-02 16:00  → funding_val_16 (val = -0.00020)
    #     2024-01-03 00:00  → funding_val_24 (val = +0.00030)
    #     2024-01-03 08:00  → funding_val_32 (val = -0.00015)
    # ------------------------------------------------------------------
    settlement_dts = [
        pd.Timestamp("2024-01-02 00:00:00"),  # S0
        pd.Timestamp("2024-01-02 08:00:00"),  # S1 — 8h later
        pd.Timestamp("2024-01-02 16:00:00"),  # S2 — 16h
        pd.Timestamp("2024-01-03 00:00:00"),  # S3 — 24h (next midnight)
        pd.Timestamp("2024-01-03 08:00:00"),  # S4 — 32h
    ]
    # Values intentionally well-separated so any mis-match is obvious
    distinct_vals = np.array([-10e-5, +5e-5, -20e-5, +30e-5, -15e-5])
    S0, S1, S2, S3, S4 = distinct_vals

    df_funding = pd.DataFrame(
        {"funding_rate": distinct_vals},
        index=pd.DatetimeIndex(settlement_dts),
    )
    print(f"\nSynthetic funding settlements (5 rows, 8-hour cadence):")
    for i, (ts, v) in enumerate(zip(settlement_dts, distinct_vals)):
        print(f"  S{i}: {ts}  →  funding_rate_current = {v:+.6f}")

    # ------------------------------------------------------------------
    # Step 2 — Build sub-daily (4h + 1h) candles placed at the CRITICAL
    # timestamps around each of the 5 settlement boundaries.  For each
    # candle we state the EXPECTED funding value.
    #
    # KEY: the funding timestamp = SETTLEMENT (availability) time, so:
    #   * A bar at time T sees the most recent settlement ≤ T.
    #   * A bar at exactly the settlement time (e.g. 08:00:00) MAY see
    #     that settlement (it settles at 08:00:00 exactly — bar opens at
    #     08:00:00, so value is available at bar open → correct).
    # ------------------------------------------------------------------
    sub_dts = [
        # --------- BEFORE S1 (the 2024-01-02 08:00 settlement) ----------
        # 4h candles opening before 08:00 on Jan 2: S0 is the latest known
        pd.Timestamp("2024-01-02 00:00:00"),   # exactly at S0 → S0
        pd.Timestamp("2024-01-02 04:00:00"),   # 4h after S0, S1 in 4h → S0
        pd.Timestamp("2024-01-02 07:00:00"),   # 1h bar at 07:00, S1 in 1h → S0
        pd.Timestamp("2024-01-02 07:59:59"),   # 1s BEFORE S1 → S0 (critical!)
        # ----- AT & AFTER S1 (2024-01-02 08:00:00 settlement) -----
        pd.Timestamp("2024-01-02 08:00:00"),   # exactly at S1 → S1
        pd.Timestamp("2024-01-02 08:00:01"),   # 1s AFTER S1 → S1
        pd.Timestamp("2024-01-02 12:00:00"),   # 4h bar at 12:00 → S1
        pd.Timestamp("2024-01-02 15:00:00"),   # 1h at 15:00, S2 in 1h → S1
        pd.Timestamp("2024-01-02 15:59:59"),   # 1s BEFORE S2 → S1
        # ----- AT & AFTER S2 (2024-01-02 16:00:00 settlement) -----
        pd.Timestamp("2024-01-02 16:00:00"),   # exactly at S2 → S2
        pd.Timestamp("2024-01-02 20:00:00"),   # 4h bar 20:00 → S2
        pd.Timestamp("2024-01-02 23:59:59"),   # 1s BEFORE S3 (midnight) → S2
        # ----- AT & AFTER S3 (2024-01-03 00:00:00 settlement) -----
        pd.Timestamp("2024-01-03 00:00:00"),   # exactly at S3 → S3
        pd.Timestamp("2024-01-03 04:00:00"),   # 4h bar 04:00 → S3
        pd.Timestamp("2024-01-03 07:59:59"),   # 1s BEFORE S4 → S3
        # ----- AT & AFTER S4 (2024-01-03 08:00:00 settlement) -----
        pd.Timestamp("2024-01-03 08:00:00"),   # exactly at S4 → S4
        pd.Timestamp("2024-01-03 12:00:00"),   # 4h bar 12:00 → S4
    ]
    expected_vals = [
        S0, S0, S0, S0,   # before / at S0 → S0
        S1, S1, S1, S1, S1,  # at / after S1 → S1
        S2, S2, S2,       # at / after S2 → S2
        S3, S3, S3,       # at / after S3 → S3
        S4, S4,           # at / after S4 → S4
    ]
    assert len(sub_dts) == len(expected_vals), "test setup error"

    n_sub = len(sub_dts)
    df_sub = pd.DataFrame(
        {
            "close": np.linspace(42000.0, 42500.0, n_sub),
            "atr_14": np.ones(n_sub) * 800.0,
        },
        index=pd.DatetimeIndex(sub_dts),
    )

    # ------------------------------------------------------------------
    # Step 3 — Run add_funding_rate and verify every single assignment
    # ------------------------------------------------------------------
    df_merged, ok = add_funding_rate(df_sub, df_funding)
    assert ok is True, "add_funding_rate returned False unexpectedly"
    assert "funding_rate_current" in df_merged.columns, "output column missing"

    print(f"\n{'Sub-daily timestamp':<25} {'Expected':>10} {'Got':>10} {'Match':>7}")
    print("-" * 58)

    errors: list[str] = []

    for ts, expected in zip(sub_dts, expected_vals):
        got = df_merged.loc[ts, "funding_rate_current"]
        match = abs(got - expected) < 1e-12
        status = "✓" if match else "✗ LEAK!"
        print(
            f"{str(ts):<25} {expected:>+10.6f} {got:>+10.6f} {status:>7}"
        )
        if not match:
            errors.append(
                f"LEAK at {ts}: expected {expected:+.6f} "
                f"(latest settled funding ≤ bar open), got {got:+.6f} — "
                f"a FUTURE (not-yet-settled) funding rate leaked!"
            )

    print("-" * 58)

    # ------------------------------------------------------------------
    # Step 4 — Edge case: bar BEFORE first settlement → must be NaN
    # ------------------------------------------------------------------
    pre_dts = [pd.Timestamp("2024-01-01 12:00:00")]  # 12h before S0
    df_pre = pd.DataFrame(
        {"close": [42000.0]}, index=pd.DatetimeIndex(pre_dts)
    )
    df_merged_pre, _ = add_funding_rate(df_pre, df_funding)
    pre_val = df_merged_pre.loc[pre_dts[0], "funding_rate_current"]
    print(f"\nBar *before first settlement* funding_rate_current is NaN:",
          pd.isna(pre_val))
    if not pd.isna(pre_val):
        errors.append(
            f"Bar before any settlement should be NaN, got {pre_val:+.6f}"
        )

    # ------------------------------------------------------------------
    # Step 5 — Critical boundary: bar at 07:59:59 (1s before S1) must
    # NOT see S1.  This is the single most important assertion in the
    # whole test — it catches the classic off-by-one in merge_asof when
    # timestamps are "near" each other.
    # ------------------------------------------------------------------
    critical_ts = pd.Timestamp("2024-01-02 07:59:59")
    crit_got = df_merged.loc[critical_ts, "funding_rate_current"]
    crit_ok = abs(crit_got - S0) < 1e-12  # must be S0, not S1
    print(f"\n[CRITICAL] Bar at {critical_ts} (1s before 08:00 settlement) "
          f"sees funding={crit_got:+.6f} → expected S0={S0:+.6f}: "
          + ("✓ CORRECT" if crit_ok else "✗ WRONG — leaks future S1!"))
    if not crit_ok:
        errors.append(
            f"CRITICAL BOUNDARY FAILURE at {critical_ts}: 1s before 08:00 "
            f"settlement, bar sees S1={crit_got:+.6f} instead of "
            f"S0={S0:+.6f}. merge_asof semantics broken!"
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
        print("\n✅ PASS — zero leakage. add_funding_rate temporal ordering is correct.")
        print("   • Bars BEFORE a settlement → prior settlement value (never peek future)")
        print("   • Bar AT EXACT settlement time → the just-settled value (available at open)")
        print("   • Bar AFTER settlement → correct latest known value")
        print("   • Bar before any settlement → NaN (no phantom data)")


if __name__ == "__main__":
    main()
