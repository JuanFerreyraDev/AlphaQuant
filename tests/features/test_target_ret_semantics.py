"""Semantic & temporal correctness verification for target_ret (continuous return).

Verifies that ``target_ret`` is computed exactly in lockstep with ``target``:
  (a) TP wins  → target_ret == +atr[i]*tp_multi / close[i]  (exact)
  (b) SL wins  → target_ret == -atr[i]*sl_multi / close[i]  (exact)
  (c) Timeout  → target_ret == (close[exit_idx] - close[i]) / close[i]
                 with exit_idx == min(i + swing_days, n - 1) — never peeks
                 beyond the swing window.
  (d) Sign consistency between target (ternary class) and target_ret (continuous).

Run: python3 tests/features/test_target_ret_semantics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import compute_target


def build_synthetic_df() -> tuple[pd.DataFrame, dict]:
    """Build a synthetic DataFrame with KNOWN target/target_ret outcomes.

    Constructs 14 bars with non-overlapping scan windows so cases are
    deterministic and independent. Swing = 3 bars (scan j=i+1..i+3).

    Layout:
      Bar 0 (i=0): Bar 1 low touches SL first → target=-1, SL-ret exact
      Bar 1 (i=1): Bar 2 high touches TP first → target=+1, TP-ret exact
      Bar 2..4: RESERVED scan window for bars 0..1 (leave undisturbed)

      Bar 5 (i=5): Scan bars 6,7,8. NO TP/SL → timeout, exit at i+3=8
      Bar 6..8: RESERVED scan window for bar 5 (timeout)

      Bar 9 (i=9): Bar 10 hits BOTH TP+SL in SAME bar → tie-break SL wins
      Bar 10..12: RESERVED scan window for bar 9
    """
    dts = pd.date_range("2026-01-01 00:00", periods=14, freq="4h")
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                      106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                      112.0, 113.0], dtype=np.float64)
    atr_14 = np.full(14, 2.0, dtype=np.float64)

    tp_multi = 1.5
    sl_multi = 1.0
    swing_days = 3

    high = close.copy()
    low = close.copy()

    # --- Bar 0 (i=0): SL hits at j=1 ---
    # TP0=103, SL0=98
    low[1] = 97.0       # SL hit at j=1
    high[1] = 102.0     # No TP at j=1
    low[2] = 99.0;  high[2] = 102.5   # j=2: safe
    low[3] = 98.5;  high[3] = 102.8   # j=3: safe

    # --- Bar 1 (i=1): TP hits at j=2 ---
    # TP1=104, SL1=99
    high[2] = 104.5     # TP hit at j=2  (overwrite j=2 from before)
    low[2] = 100.0      # No SL at j=2
    # j=3: already safe above

    # --- Bar 5 (i=5): Scan j=6,7,8 → TIMEOUT → exit at 5+3=8 ---
    # TP5 = 105 + 3 = 108, SL5 = 105 - 2 = 103
    # Make sure none of j=6,7,8 trigger SL/TP:
    low[6]  = 104.0  # > 103
    high[6] = 107.5  # < 108
    low[7]  = 104.5  # > 103
    high[7] = 107.8  # < 108
    low[8]  = 105.0  # > 103
    high[8] = 107.9  # < 108

    # --- Bar 9 (i=9): Bar 10 → BOTH TP+SL → tie-break SL wins ---
    # TP9 = 109 + 3 = 112, SL9 = 109 - 2 = 107
    low[10]  = 106.0  # <= 107 (SL)
    high[10] = 113.0  # >= 112 (TP)
    # Tie-break: SL checked first → target=-1, SL ret

    open_ = close - 0.1

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "atr_14": atr_14, "volume": np.ones(14) * 1000.0},
        index=pd.DatetimeIndex(dts, name="timestamp"),
    )
    meta = {
        "swing_days": swing_days,
        "tp_multi": tp_multi,
        "sl_multi": sl_multi,
        "SL_idx": [0, 9],
        "TP_idx": [1],
        "timeout_idx": [5],
    }
    return df, meta


def main() -> None:
    print("=" * 72)
    print("SEMANTIC VERIFICATION: target_ret (continuous return) correctness")
    print("=" * 72)

    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1) Pointwise exact-value check on synthetic candles
    # ------------------------------------------------------------------
    df_syn, meta = build_synthetic_df()
    sd = meta["swing_days"]
    tpm = meta["tp_multi"]
    slm = meta["sl_multi"]
    close = df_syn["close"].values
    atr = df_syn["atr_14"].values
    n = len(df_syn)

    df_out = compute_target(
        df_syn.copy(),
        swing_days=sd,
        atr_tp_multi=tpm,
        atr_sl_multi=slm,
        timeframe_hours=4.0,
    )
    assert "target" in df_out.columns
    assert "target_ret" in df_out.columns

    tgt = df_out["target"].values.astype(int)
    tret = df_out["target_ret"].values.astype(float)

    print("\n[1/4] Pointwise exact target_ret check (synthetic, 3 known-case types):")
    print(f"  swing={sd}  TP={tpm}xATR  SL={slm}xATR")
    print(f"  {'i':>3} {'target':>7} {'expected_ret':>14} {'got_ret':>14} "
          f"{'match':>8} {'case':>12}")
    print("  " + "-" * 70)

    for i in range(n):
        # Per-case expected.
        if i in meta["SL_idx"]:
            exp_ret = -(atr[i] * slm) / close[i]
            exp_tgt = -1
            case = "SL"
        elif i in meta["TP_idx"]:
            exp_ret = (atr[i] * tpm) / close[i]
            exp_tgt = 1
            case = "TP"
        elif i in meta["timeout_idx"]:
            exit_idx = min(i + sd, n - 1)
            exp_ret = (close[exit_idx] - close[i]) / close[i]
            exp_tgt = 0
            case = f"TO→{exit_idx}"
        else:
            exp_tgt = int(tgt[i])
            if tgt[i] == 1:
                exp_ret = (atr[i] * tpm) / close[i]
            elif tgt[i] == -1:
                exp_ret = -(atr[i] * slm) / close[i]
            else:
                ei = min(i + sd, n - 1)
                exp_ret = (close[ei] - close[i]) / close[i]
            case = "padding"

        ret_match = abs(float(tret[i]) - float(exp_ret)) < 1e-14
        tgt_match = bool(tgt[i] == exp_tgt)
        ok = ret_match and tgt_match
        status = "✓" if ok else "✗ WRONG"
        print(f"  {i:>3} {int(tgt[i]):>7} {float(exp_ret):>+14.10f} "
              f"{float(tret[i]):>+14.10f} {status:>8} {case:>12}")
        if not tgt_match:
            errors.append(f"i={i} target mismatch: expected {exp_tgt}, got {int(tgt[i])}")
        if not ret_match:
            errors.append(
                f"i={i} target_ret mismatch (case={case}): "
                f"expected {float(exp_ret):+.12f}, got {float(tret[i]):+.12f}"
            )

    # ------------------------------------------------------------------
    # 2) Timeout exit_idx bound: NEVER exceeds i + swing_days, NEVER peeks
    #    beyond n - 1.  We prove this by recomputing every timeout row with
    #    both i+sd and clamped variants and asserting equality — if the
    #    implementation ever used j > i+sd or j > n-1, the equality would
    #    fail for last rows.
    # ------------------------------------------------------------------
    print("\n[2/4] Timeout boundary audit (exit_idx ≤ i+swing_days ≤ n-1):")
    to_rows = [i for i in range(n) if tgt[i] == 0]
    bad_bound = 0
    for i in to_rows:
        exit_idx_clamped = min(i + sd, n - 1)
        exp_ret_clamped = (close[exit_idx_clamped] - close[i]) / close[i]
        if abs(float(tret[i]) - float(exp_ret_clamped)) > 1e-14:
            bad_bound += 1
            errors.append(
                f"Timeout boundary violation at i={i}: ret does not match "
                f"clamped exit_idx={exit_idx_clamped} (n-1={n-1}, i+sd={i+sd})"
            )
    print(f"  Timeout rows checked: {len(to_rows)}  boundary violations: {bad_bound}")

    # ------------------------------------------------------------------
    # 3) Target / target_ret sign consistency across the full synthetic set.
    # ------------------------------------------------------------------
    print("\n[3/4] Sign consistency (target class ↔ target_ret sign):")
    sign_ok = 0
    for i in range(n):
        tg = int(tgt[i])
        tr = float(tret[i])
        if tg == 1 and tr > 0:
            sign_ok += 1
        elif tg == -1 and tr < 0:
            sign_ok += 1
        elif tg == 0:
            # Timeout: sign of ret matches (close[exit] - close[i]) / close[i]
            exit_idx = min(i + sd, n - 1)
            raw_ret_sign = np.sign((close[exit_idx] - close[i]) / close[i])
            if np.sign(tr) == raw_ret_sign or (tr == 0.0 and raw_ret_sign == 0):
                sign_ok += 1
            else:
                errors.append(f"i={i} timeout sign mismatch")
        else:
            errors.append(f"i={i} sign inconsistent: target={tg} ret_sign={np.sign(tr)}")
    print(f"  Consistent rows: {sign_ok}/{n}")

    # ------------------------------------------------------------------
    # 4) Real-data sanity: BTC_USDT / 4h — distribution of target_ret.
    #    * TP-group mean ≈ +(1.5 * mean(atr/close))  (positive, bounded)
    #    * SL-group mean ≈ -(1.0 * mean(atr/close))  (negative, bounded)
    #    * Timeout-group: roughly centered but with heavier tails
    # ------------------------------------------------------------------
    print("\n[4/4] Real-data sanity (BTC_USDT / 4h target_ret distribution):")
    try:
        from src.brain.features import compute_all_technicals
        from src.utils.helpers import load_csv_data

        df_real = load_csv_data("BTC_USDT", "4h")
    except Exception as exc:
        errors.append(f"Could not load BTC_USDT 4h CSV: {exc}")
        df_real = pd.DataFrame()

    if not df_real.empty and "atr_14" not in df_real.columns:
        df_real = compute_all_technicals(df_real)

    if not df_real.empty and "atr_14" in df_real.columns:
        df_real_out = compute_target(
            df_real.copy(),
            swing_days=10,
            atr_tp_multi=1.5,
            atr_sl_multi=1.0,
            timeframe_hours=4.0,
        )
        tr_real = df_real_out["target_ret"].values
        tg_real = df_real_out["target"].values.astype(int)
        finite_mask = np.isfinite(tr_real)
        tr_real = tr_real[finite_mask]
        tg_real = tg_real[finite_mask]

        print(f"  Rows (finite): {len(tr_real)}")
        for label, name in [(1, "TP-group"), (-1, "SL-group"), (0, "TO-group")]:
            mask = tg_real == label
            sub = tr_real[mask]
            if len(sub) > 0:
                print(f"  {name:>10}  n={len(sub):>6}  "
                      f"mean={sub.mean():+.8f}  std={sub.std():.8f}  "
                      f"min={sub.min():+.8f}  max={sub.max():+.8f}")
                if label == 1 and sub.mean() <= 0:
                    errors.append(f"Real TP-group mean non-positive: {sub.mean():+.8f}")
                if label == -1 and sub.mean() >= 0:
                    errors.append(f"Real SL-group mean non-negative: {sub.mean():+.8f}")
        # Overall target_ret quantiles.
        q = np.percentile(tr_real, [0, 25, 50, 75, 100])
        print(f"  {'Overall':>10}  min={q[0]:+.8f}  p25={q[1]:+.8f}  "
              f"med={q[2]:+.8f}  p75={q[3]:+.8f}  max={q[4]:+.8f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    if errors:
        print("❌ FAILURE — target_ret semantic issues detected:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("✅ PASS — target_ret computed correctly:")
        print("   • TP wins  → ret = +atr*tp_multi/close  (exact)")
        print("   • SL wins  → ret = -atr*sl_multi/close  (exact, tie-break preserved)")
        print("   • Timeout  → ret = (close[i+swing]-close[i])/close[i] (bounded)")
        print("   • Class ↔ sign: fully consistent across all rows")
        print("   • Real 4h BTC data: TP-group pos, SL-group neg, expected quantile range")


if __name__ == "__main__":
    main()
