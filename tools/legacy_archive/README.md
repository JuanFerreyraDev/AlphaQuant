# Legacy Archive

These scripts were the original walk-forward experiment implementations
(Phase 1 of the AlphaQuant audit). They have been superseded by the
unified pipeline in `src/pipeline/` and the CLI in `tools/aq.py`.

## Replacement mapping

| Legacy script                      | New equivalent                                        |
|------------------------------------|-------------------------------------------------------|
| `exp01_trend_htf_walkforward.py`   | `python -m tools.aq ab-test SYMBOL --profile trend_htf` |
| `exp02_funding_rate_walkforward.py`| `python -m tools.aq ab-test SYMBOL --profile funding_rate` |
| `exp03_taker_buy_ratio_walkforward.py` | `python -m tools.aq ab-test SYMBOL --profile taker_buy_ratio` |
| `compare_binary_vs_multiclass.py`  | _N/A (Historical analysis script)_ |
| `reconcile_naive_target_comparison.py` | _N/A (Historical analysis script)_ |

## JSON result files

The `*_results.json` files in this directory are the original outputs
from the legacy scripts and are kept for reference / reproducibility.
New runs produce timestamped reports under `reports/{SYMBOL}/`.
