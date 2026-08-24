# Archive: Diagnostic Scripts (Productized in aq.py)

## Overview

This directory contains diagnostic scripts that were integral to AlphaQuant research between June-August 2026 but have since been productized as subcommands in the unified [`tools/aq.py`](../../aq.py) CLI.

These scripts are preserved here for **historical reference and reproducibility** — they should not be used for new analysis. Use the `aq` CLI instead.

---

## Archived Scripts & Their Replacements

| Script | Functionality | Replaced By | Usage |
|--------|---|---|---|
| `diagnose_naive_baseline.py` | Model (Volatility Hunter 3-feature) vs naive long-only baseline comparison on val/test split. Reports PF, trade counts, and signal vs ema_50 correlation. | `aq diagnose-naive-baseline` | `python -m tools.aq diagnose-naive-baseline BTC_USDT` |
| `diagnose_swing_and_regimes.py` | Part A: Swing sweep (swing ∈ {2,3,4,5,7,10}) on 4h production val/test. Part B: Cross-regime consistency with best swing on bear 2022 / range 2023 / bull 2024 windows. Train-only-past-data rule enforced. | `aq diagnose-swing-and-regimes` | `python -m tools.aq diagnose-swing-and-regimes ETH_USDT` |
| `diagnose_regimes_rigorous.py` | Cross-regime analysis with rigorous bootstrap (1000x per-trade return resampling). Evaluates 5 windows (3 in-sample, 2 OOS) with p5/p95 CI on delta PF (model - naive). No lookahead. | `aq diagnose-regimes-rigorous` | `python -m tools.aq diagnose-regimes-rigorous BTC_USDT` |
| `diagnose_timeframe_swing_sweep.py` | Swing sweep on alternative timeframe (default 1h, configurable). Validates TP/SL ratio at different holding periods. Reports ATR-as-%price sanity check. | `aq diagnose-timeframe-swing-sweep` | `python -m tools.aq diagnose-timeframe-swing-sweep SOL_USDT --timeframe 1h` |
| `diagnose_timeframe_data.py` | Level-1 data health check: feature NaN/zero %, sentiment merge sanity, target class balance, regime comparison (val vs test), point-biserial correlation on train split. | `aq diagnose-data` | `python -m tools.aq diagnose-data SOL_USDT --timeframe 4h` |

---

## Why Archive These?

1. **Code Centralization**: Combining 5 standalone scripts into a single CLI reduces duplication and maintenance burden.
2. **Consistent Interface**: All diagnostics now share argument parsing, logging, and error handling via argparse + `aq.py`.
3. **No Loss of Functionality**: Every diagnostic check is preserved; users simply call `aq {subcommand}` instead of `python tools/diagnostics/{script}.py`.
4. **Version Control**: Archived scripts remain in git history; no data loss.

---

## When to Use These Original Scripts

**Use cases for the originals (preserved here):**
- Reproducing exact results from a specific date/commit (each script contains hardcoded parameters like `SWING=10`, `TP=1.5`, etc.).
- Auditing the exact logic that ran at a previous point in time.
- Academic/historical reference on the diagnostic philosophy used in August 2026.

**For all new work**: Use `aq` subcommands. They are identical clones of the originals.

---

## Implementation Note

The `aq.py` versions are **exact functional duplicates** of these standalone scripts. Each was ported by:
1. Extracting the main logic from the `.py` file.
2. Wrapping it in an `_cmd_*` function.
3. Adding CLI parser registration in `main()`.
4. Testing output parity with the original.

See [tools/aq.py](../../aq.py) for implementation details.

---

## Historical Context

- **Investigation Scope Closed**: June-August 2026 focused on technical feature engineering (OHLCV-derived + sentiment). Three target formulations (`binary_homerun`, `multiclass_3`, `regression_return`) were exhaustively tested across BTC/ETH/SOL. All failed the gate (ΔPF p5 ≤ 0). These diagnostics were essential to that investigation.
- **Next Phase**: Transitioning to on-chain data domain (ballpark metrics, exchange flow, holder behavior, etc.). The diagnostic pipeline remains in place; only the data sources change.
- **Bot Production**: Independent of this research pipeline. Bot runs using `main.py` + `strategy_optimizer.py`, not walk-forward research code.

---

## References

- **WORKFLOW.md §5.4**: Archive + discard policy.
- **Refactor Plan (plan_fase_dominio_nuevo_y_refactor.md)**: §2.1 specifies these 5 scripts for archiving once `aq diagnose-data` coverage was confirmed.
- **tools/aq.py**: Unified CLI source code.
