# AlphaQuant — Predictive Edge Research: Conclusion

> **Status:** Closed (August 2026), not ruled out for revisiting in the future
> with new evidence or new data domains.
> **Scope of this document:** summarizes what was investigated, what was
> found, why it was decided to stop here, and what would be needed to pick
> it back up without having to reconstruct the context from scratch.

---

## 1. Research question

Does a predictive edge exist — statistically robust and economically
executable — for trading BTC_USDT (and secondarily ETH_USDT, SOL_USDT) on
4h/1h timeframes, using an XGBoost classifier/regressor trained on features
derived from public market data (OHLCV, technical indicators, sentiment,
funding rate, volume microstructure, and on-chain data)?

## 2. Executive summary

**No edge was found that survives a rigorous validation protocol.** Three
symbols, three model formulations, and five features/data domains beyond
the base feature set were evaluated — no combination passed the
pre-registered statistical significance criterion (walk-forward OOS +
paired block bootstrap vs. a naive baseline, with cross-formulation
consistency and bootstrap-seed stability checks).

This does **not** mean "the market is perfectly efficient" or "a
profitable bot is impossible" — it means that, with the evidence gathered,
**this specific approach** (publicly-derived features, a 10-40 bar holding
horizon, TP/SL barrier classification/regression) showed no measurable
advantage over simply buying whenever possible.

## 3. What was built (reusable asset, independent of the outcome)

This is the part of the work that does **not** expire even though the
conclusion is negative — it is reusable infrastructure for any future
research, on these or other symbols:

- **Rigorous out-of-sample validation engine** (`src/utils/oos_validation.py`):
  expanding-window walk-forward, temporal embargo, paired block bootstrap
  (not naive resampling) against a naive "buy whenever possible" baseline —
  not against absolute breakeven.
- **Pre-registered statistical gates**: `MIN_BOOTSTRAP_P5 > 0.0` (the
  delta's confidence interval must exclude zero) and
  `MIN_POOLED_TRADES ≥ 300` (statistical power floor), fixed in code, not
  adjustable per run.
- **Three interchangeable model formulations** (factory pattern): binary
  classification (`binary_homerun`), multiclass (`multiclass_3`),
  continuous-return regression (`regression_return`) — any new feature is
  automatically evaluated against all three.
- **Declarative feature pipeline** (`FeatureProfile` +
  `ENRICHMENT_REGISTRY`): adding a new data domain means writing a
  fetcher + an enrichment function + a leakage test, without touching the
  validation engine.
- **Unified CLI** (`tools/aq.py`): `baseline`, `ab-test`, `diagnose-data`,
  and the legacy diagnostics, with a consistent and honest argument
  pattern (no flag that the implementation silently ignores).
- **3-step rigor criterion** for any result that crosses the raw gate:
  confidence-interval width, consistency across sister formulations on the
  same timeframe, stability across different bootstrap seeds. No "PASS" is
  accepted without clearing all three.

## 4. What was investigated and what was found

### 4.1 Target and model formulation

| Investigated                                                       | Result                                                                                                                                                                                              |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Binary labeling with temporal-order resolution ("pessimistic AND") | Emptied the positive class on long windows — fixed with a bar-by-bar resolver with a consistent tie-break                                                                                           |
| Ternary target (TP / Timeout / SL) with realistic timeout payoff   | Fixed the bias of treating every timeout as a full SL loss                                                                                                                                          |
| `binary_homerun` (binary "TP vs rest" classification)              | No confirmed edge after fixing calibration bugs                                                                                                                                                     |
| `multiclass_3` (3-class softmax)                                   | No confirmed edge; requires its own threshold_grid (0.25-0.70, not 0.50-0.85)                                                                                                                       |
| `regression_return` (continuous return)                            | No confirmed edge; required recalibrating the threshold_grid to the model's real prediction range (much narrower than the real target range — model "shrunk" toward the mean due to lack of signal) |

### 4.2 Symbols

| Symbol   | Result                                                                                                                                                                      |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BTC_USDT | Baseline not confirmed in any formulation/timeframe                                                                                                                         |
| ETH_USDT | Baseline had 1 raw PASS (`4h × binary_homerun`) that failed the cross-formulation consistency check (`multiclass_3` on the same timeframe gave the worst result in the set) |
| SOL_USDT | 0/4 configurations pass the raw gate; confirmed that the FTX shock (Nov-Dec 2022) does not explain the result — excluding it makes the picture worse, not better            |

### 4.3 Features / data domains (all evaluated control-vs-treatment, BTC_USDT, 4h+1h, 3 formulations)

| Feature                                     | Domain                            | Result                                                          |
| ------------------------------------------- | --------------------------------- | --------------------------------------------------------------- |
| `trend_htf` (distance to daily EMA200)      | Price, higher timeframe           | 0/8 configurations pass                                         |
| `funding_rate_current`                      | Derivatives/leverage              | 1/8 raw pass, fails cross-formulation consistency               |
| `taker_buy_ratio`                           | Aggressor volume microstructure   | 0/8 pass                                                        |
| `onchain_active_addresses` (Blockchain.com) | On-chain, address activity        | 1/6 raw pass, fails cross-formulation consistency               |
| `mempool_fee_rate_p50` (mempool.space)      | On-chain, fee/congestion pressure | 1/6 raw pass (seed-stable), fails cross-formulation consistency |

### 4.4 Barrier configuration (swing/TP/SL)

Swept 24 valid combinations (swing ∈ {5,7,10,15,20}, TP/SL ∈
{1.0,1.5,2.0}×ATR) over the control feature set, BTC_USDT, 3 formulations,
2 timeframes. **The original configuration (`swing=10, TP=1.5×ATR,
SL=1.0×ATR`) remained the best available** — no alternative candidate
cleared the 3-criterion rigor check, and observed differences were within
expected bootstrap noise.

## 5. Cross-cutting pattern in the results (the most important takeaway)

Across **every** feature discard, the same signature repeats: a single raw
`PASS`, almost always in one formulation only, that does not replicate in
the sister formulations on the same timeframe. Given the accumulated
volume of tests across the investigation (60+ configurations evaluated),
this is exactly what theory predicts under the "no real edge" hypothesis —
it is not evidence of something that almost worked, it is the expected
signature of multiple-comparisons noise.

No individual feature ever had a point-biserial correlation with the
target above ~0.05 at any point in the investigation — a signal that the
problem was not "finding the right model" but that the informational
content available in these sources, for this holding horizon (10-40 bars)
and these symbols, is insufficient.

## 6. Why the decision is to stop here (and not keep adding symbols/features)

- Three genuinely distinct information domains were covered (higher
  timeframe price, derivatives/leverage, volume microstructure) plus two
  independent on-chain sources (network activity, fee pressure) — this is
  not a single failed attempt, it's a consistent pattern across
  heterogeneous domains.
- The barrier configuration (swing/TP/SL) was ruled out as the limiting
  factor — a 24-combination sweep found nothing better than the starting
  point.
- Directional bias / market beta was ruled out as an explanation for
  apparently positive results (systematic comparison against a naive
  baseline, not against absolute breakeven).
- The marginal cost of continuing (new paid on-chain sources via
  CoinMetrics, expanding to ETH/SOL for each domain) grows faster than the
  probability of finding something, given the observed pattern.

## 7. What it would take to pick this back up in the future

If the research is resumed at some point, the most productive entry point
is **not** repeating what's already been done — it's one of these routes,
in priority order:

1. **A genuinely new data domain**: finer order-flow (if a reliable
   historical source for liquidations/order book appears), cross-asset
   data (lead-lag with SPX/DXY/BTC dominance), or fundamental/narrative
   signals (not explored at all).
2. **Cross-symbol pooling**: training a model on combined data from
   several symbols to increase available statistical power — doesn't add
   new information, but directly attacks the wide-confidence-interval
   problem that showed up in nearly every experiment.
3. **A different holding horizon**: everything investigated used 10-40
   bars (hours to ~2 days). Pure daily timeframe (1d) was never explored
   with the current validation engine, nor were multi-week horizons.
4. **Expanding on-chain coverage to ETH**: left pending due to the cost of
   registering with CoinMetrics — if resumed, evaluate Etherscan/Dune
   Analytics first (free, no paid account) before CoinMetrics.

All infrastructure from Section 3 continues to work for any of these four
routes without architectural changes.

## 8. Code status

- **Production** (`main.py`, `src/api/*`, `src/engine/tasks.py`,
  `strategy_optimizer.py`): unchanged, out of scope for this research.
  Still uses its own single-split gate (`_passes_oos_sanity_check`),
  weaker than the research walk-forward — whether to migrate it is a
  pending, separate decision.
- **Research layer** (`src/pipeline/*`, `src/utils/oos_validation.py`,
  `src/utils/data_splits.py`, `tools/aq.py`): consolidated, no duplicated
  split logic, single verified gate, honest CLI (no flag that does
  anything other than what it claims).
- **Discarded features** (`trend_htf`, `funding_rate_current`,
  `taker_buy_ratio`, `onchain_active_addresses`, `mempool_fee_rate_p50`):
  code and leakage tests preserved, not deleted — left out of the
  `control` profile by default, documented as candidates for
  re-evaluation if new evidence appears.
- **Legacy diagnostic scripts**: archived under `tools/legacy_archive/`
  with notes on which `aq.py` command replaces each one.

## 9. Process lessons (beyond the research outcome)

Regardless of the negative conclusion, the research left behind
infrastructure fixes that are valuable on their own:

- Split calibration bug not propagated across multiple standalone
  scripts (fixed, consolidated into shared functions).
- Selection bias from in-sample contamination when evaluating "historical
  regimes" (caught and corrected).
- `baseline` gate comparing against absolute breakeven instead of the
  naive baseline (real bug, fixed — would have produced systematic false
  positives).

---

**This document replaces the need to re-read the full research
conversation.** Any future resumption should start here, not from scratch.
