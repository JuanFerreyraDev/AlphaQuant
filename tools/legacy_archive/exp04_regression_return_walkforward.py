"""Experiment 04: regression_return pivot — continuous-return regression formulation.

Summary:
  - Formulation: `regression_return` (predict continuous `target_ret`)
  - Status: DISCARDED after full protocol + bugfix verification (no exploitable signal)

Description:
  This script records the experimental design and final consolidated results
  from the regression-return pivot. The experiment evaluated 6 configurations
  (3 symbols × 2 timeframes) using the baseline 14 control features.

Key configuration (shared across runs):
  - swing_period = 10 bars
  - tp_multi = 1.5 × ATR
  - sl_multi = 1.0 × ATR
  - window_months = 6, step_months = 6
  - threshold_grid = (-0.0035, 0.0070, 0.0003)
  - n_bootstrap = 1000, n_blocks = 8, random_state = 42

Final numeric results (post-fix, cleaned):
  - BTC_USDT 4h: windows_used=12/12, pooled_trades=1210, ΔPF_p5=-0.0338, ΔPF_p95=+0.1491, passes_gate=FAIL
  - BTC_USDT 1h: windows_used=12/12, pooled_trades=4798, ΔPF_p5=-0.0002, ΔPF_p95=+0.0752, passes_gate=FAIL
  - ETH_USDT 4h: windows_used=12/12, pooled_trades=1223, ΔPF_p5=-0.0278, ΔPF_p95=+0.0634, passes_gate=FAIL
  - ETH_USDT 1h: windows_used=12/12, pooled_trades=5190, ΔPF_p5=-0.0399, ΔPF_p95=+0.0120, passes_gate=FAIL
  - SOL_USDT 4h: windows_used=10/10, pooled_trades=832, ΔPF_p5=-0.1612, ΔPF_p95=-0.0219, passes_gate=FAIL
  - SOL_USDT 1h: windows_used=10/10, pooled_trades=2522, ΔPF_p5=-0.0494, ΔPF_p95=+0.0433, passes_gate=FAIL

Lessons (separate):

1) Statistical result
   - After the bugfix (threshold sentinel handling), NONE of the 6
     `regression_return` configurations pass the pre-registered gate
     (ΔPF_p5 > 0.0).
   - CI widths post-fix are reasonably narrow (all < 0.20), so the
     post-fix numbers are stable.
   - A cheap diagnostic (distribution compression check: compare
     predicted distribution vs true `target_ret` variance) showed a
     ~26× compression in many windows (model predictions collapsed around
     a near-constant). This should be run for any future regression pivot
     as a fast signal-quality check.

2) Engineering lesson (bug)
   - Root cause: sentinel `-1.0` used to indicate "no valid threshold" was
     inadvertently tested with `thr < 0`, which also rejected valid
     negative thresholds for regression grids (e.g. -0.0035).
   - Effect: some OOS windows were silently skipped (`threshold_failed`) and
     pooled_trade counts were artificially reduced, producing false
     positives in some aggregated reports (BTC/4h was affected).
   - Fix applied: explicit constant `THRESHOLD_NOT_FOUND = -1.0` and
     comparison `thr == THRESHOLD_NOT_FOUND`.
   - Operational recommendation: always report `windows_used` per config
     and fail the run if any windows are unexpectedly skipped; this is the
     single cheap signal that would have exposed the issue earlier.

3) Multiple comparisons argument
   - The baseline screening evaluated 18 configurations (3 symbols × 2
     TFs × 3 formulations). With a one-sided 95% gate, expected false
     positives ≈ 0.9 under pure noise. Post-fix observed 1/18 nominal
     gate pass (ETH/4h `binary_homerun`), consistent with chance; and
     that lone pass fails deeper rigor checks.

Conclusion:
  - The `regression_return` pivot is NOT supported by the data and fails
    pre-registered rigor checks after correcting the sentinel bug. The
    regressions were integrated into the pipeline as reusable code
    (parametrizable `target_col`, training helpers), but the pivot
    conclusion is to discard `regression_return` as an exploitable
    formulation on the current control feature set.

Reference:
  - This experiment's raw reports and JSON outputs are in `reports/`.
  - See the WORKFLOW documentation §5.4 and the ARCHITECTURE entry for
    cross-reference to this archive note.
"""

# This file is an archival note. The executable experiment runner lives
# in the main diagnostics tooling (tools/diagnostics) and the canonical
# reports are saved under `reports/` during runs.
