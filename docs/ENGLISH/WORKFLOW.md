# AlphaQuant — Operational Workflow and Methodology

> Step-by-step guide with technical rationale for each phase. Version 1.0 — August 2026.

---

## Prologue: Non-Negotiable Core Principles

Before initiating any operation in AlphaQuant, internalize the following rules:

1. **One orthogonal feature per experiment.** Never stack multiple new features in the same A/B test. Causality cannot be assigned if more than one independent variable changes between control and treatment.
2. **Pre-registered statistical gates.** The thresholds `pooled_trades ≥ 300` and `ΔPF_p5 > 0.0` (model vs naive_long, for both baseline and A/B tests) are hardcoded and must NOT be modified post-hoc to fit marginal results. Any relaxation constitutes p-hacking by definition. The baseline does NOT use absolute PF > 1.0 as a gate — that reintroduces directional drift of the asset (bullish drift ≠ alpha).
3. **Leakage prevention is critical.** All higher timeframe (HTF) data, future data, or data requiring temporal settlement MUST be verified with integration tests in `tests/features/`. Without a leakage test, a feature cannot enter the pipeline.
4. **Fixed control + variable treatment.** The set of 14 control features is immutable. A treatment NEVER removes control features; it can only add 1 new orthogonal column.

---

## Phase 1 — Data Ingestion and EDA Diagnostics

### Objective

Acquire raw market data of the highest quality and diagnose its validity before feeding any model. Garbage in → garbage out, amplified in crypto by information asymmetry and microstructure dynamics.

### Operational Steps

#### 1.1 Downloading OHLCV with Microstructure Data

Always prefer `--binance-rest` over standard ccxt endpoints. The native endpoint exposes microstructure fields (`taker_buy_base_vol`, `n_trades`, `quote_volume`) required for order flow aggressiveness features.

```bash
# USD-M Futures, 4h + 1h timeframes, with microstructure fields
python -m src.brain.data_fetcher BTC_USDT --timeframe 4h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --timeframe 1h --binance-rest

# Daily timeframe mandatory for HTF features (trend_htf = distance to 1d EMA200)
python -m src.brain.data_fetcher BTC_USDT --timeframe 1d

# Funding rate history (applicable to perpetual futures)
python -m src.brain.data_fetcher BTC_USDT --funding-rate
```

**Expected Output:**

```
data/raw_csv/BTC_USDT/
├── 1h.csv          # open,high,low,close,volume,quote_volume,n_trades,taker_buy_base_vol,taker_buy_quote_vol
├── 4h.csv
├── 1d.csv
└── funding_rate.csv  # timestamp, funding_rate (settlements every 8h UTC)
```

> **⚠️ Note on `--fetch` in Unified CLI:** The `--fetch` flag in `aq baseline` / `aq ab-test` invokes `data_fetcher` **without** `--binance-rest`. For the `taker_buy_ratio` profile (which requires microstructure columns), manually download data with `--binance-rest` **before** running the A/B test. For `control`, `trend_htf`, and `funding_rate` profiles, `--fetch` is sufficient.

#### 1.2 Level-1 Data Health Diagnostics (`aq diagnose-data`)

Before training any model, execute the data health sanity check:

```bash
python -m tools.aq diagnose-data BTC_USDT --timeframe 4h --swing 10
```

This diagnostic reports:

| Panel | What it Detects |
|-------|-----------------|
| **(a) Feature health: % NaN / % == 0** | Broken features (all-NaN) or degenerate features (all-zero). RSI/Stoch/MACD/ADX/ATR must be <1% NaN post-warmup. Sentiment <2% NaN. |
| **(b) Sentiment merge sanity** | `df.index` must be MONOTONICALLY increasing `datetime64[ns]` WITHOUT duplicates. `fng_value` must display a "staircase" pattern: each daily value repeats for ALL intraday candles of that day. If `days_with_multiple > 0`, F&G merging is broken. |
| **(c) Target class balance** | Measured on BTC/4h swing=10 ternary target (TP=1.5xATR, SL=1.0xATR): **33.52% TP / 13.51% timeout / 52.97% SL**. Ranges are NOT universal — they vary with `swing_period`, `tp_multi`, and `sl_multi`: timeout increases if TP/SL are too wide; SL increases if swing is too short or SL is too tight. If `target SL > 55%` persistently, review per-bar setup risk. |
| **(d) Val vs test regime** | Train/val/test cumulative return and volatility comparison. If `test_std > 2 × train_std`, the test period is structurally different: the model will generalize poorly. Reject the asset or extend the training window. |
| **(e) Point-biserial corr(feature, target)** | Top-5 correlations in training data. If ALL correlations are < 0.03, there is no linear signal detectable → abandon the asset, do not waste time on XGBoost. |

#### 1.3 Deep Diagnostics (`tools/diagnostics/`)

If Level-1 checks pass, run rigorous EDA scripts:

```bash
# 1. Decouple "market beta" vs "classifier alpha"
python3 tools/diagnostics/diagnose_naive_baseline.py --symbol BTC_USDT
# Output: If naive_long PF > model PF in val, your model has NO timing edge;
#         performance is driven entirely by asset directional drift.

# 2. Temporal regime comparison
python3 tools/diagnostics/diagnose_regimes_rigorous.py --symbol BTC_USDT
# Output: Does the train-val-test split preserve regime statistics?
#         If significant regime shift is detected (KS-test p < 0.05), baseline will fail gates.

# 3. Swing × return sweep
python3 tools/diagnostics/diagnose_swing_and_regimes.py --symbol BTC_USDT
# Output: Sensitivity of PF to swing_period. Confirms swing=10 is not an accidental overfitting point.

# 4. Timeframe × swing sweep
python3 tools/diagnostics/diagnose_timeframe_swing_sweep.py --symbol BTC_USDT --timeframe 1h
# Output: Confirms model performance across different timeframes and swing periods.
```

### 🔴 Why do we do this?

**Mathematical Rationale:** The No Free Lunch Theorem in ML guarantees no model excels without proper data. In quantitative trading, 80% of "failing models" actually fail due to **data contamination** (NaN propagation, all-zero features, misaligned sentiment, targets with look-ahead leakage), not model architecture.

**Quantitative Rationale:** A misaligned `fng_sma_14` (NaN propagation across intraday timeframes) reduces effective training size from N bars to N - 14 per fold, skewing the entire validation distribution. Using `merge_asof(direction='backward')` is mandatory because a 4h candle at 22:00 UTC DOES NOT have access to 00:00 UTC Fear & Greed data of the next day.

**Engineering Rationale:** Regime diagnostics prevent the "most expensive false positive in quant trading": a baseline that appears performant in validation during a bull trend, but whose Profit Factor collapses to 0.6 in test (bear or sideways market). Panel (d) quantifies this drift BEFORE spending 30 minutes running walk-forward validation.

---

## Phase 2 — New Asset Screening (Baseline)

### Objective

Determine whether a new asset (e.g., `SOL_USDT`, `ETH_USDT`) exhibits sufficient market inefficiency for the 14 control features to generate a Profit Factor **statistically superior to naive_long (always long)** in OOS testing. The gate criterion is `ΔPF p5 > 0.0` (model vs naive), NOT absolute PF > 1.0 — absolute PF confuses directional market drift with classifier alpha. If it fails, abandon the asset before investing effort in orthogonal features.

### Operational Steps

#### 2.1 Running Baseline Screening

```bash
# Automatically fetches data if local CSV is missing
python -m tools.aq baseline ETH_USDT --timeframes 4h 1h --fetch
```

**Internal Pipeline Execution:**

```
run_baseline(ETH_USDT, timeframes=[4h, 1h])
  │
  ├─ For each timeframe:
  │    ├─ build_dataset(profile="control")
  │    │    ├─ load_csv_data → compute_all_technicals → add_sentiment
  │    │    ├─ compute_target(swing=10, TP=1.5×ATR, SL=1.0×ATR)
  │    │    └─ drop(COLS_TO_DROP) + dropna()
  │    │
  │    └─ For each formulation ∈ {binary_homerun, multiclass_3}:
  │         └─ run_walk_forward()
  │              ├─ Dynamic train/val/test splits (with swing-bar embargo)
  │              ├─ For each OOS window (6m train → 6m step):
  │              │    ├─ Train XGB on train split
  │              │    ├─ Grid-search threshold on val split (maximizes net return)
  │              │    └─ Evaluate model + threshold on test split (OOS)
  │              │         → PF_model, PF_naive_long (always long), ΔPF = model − naive
  │              │
  │              ├─ Pooled trade count: Σ test trades across all windows
  │              └─ Paired block bootstrap ΔPF (1000 iterations, 8 blocks/window) → p5, p95
  │                   passes_gate = (trades ≥ 300) AND (ΔPF p5 > 0.0)
  │
  └─ JSON report saved to reports/ETH_USDT/baseline_{timestamp}.json
```

#### 2.2 Interpreting the SUMMARY Table

```
====================================================================================
SUMMARY — ETH_USDT Baseline
====================================================================================
Config                       OOS PF  ΔPF p5   Trades   Gate  #w_used  #w_Δ>0
------------------------------------------------------------------------------------
4h × binary_homerun          1.1234 +0.0312      412   PASS        12        9
4h × multiclass_3            1.0988 -0.0124      389   FAIL        12        5
1h × binary_homerun          1.0411 -0.0377      612   FAIL        12        4
1h × multiclass_3            1.0765 +0.0102      598   PASS        12        8
------------------------------------------------------------------------------------
```

> **⚠️ Column Semantics:** `OOS PF` is the point estimate **absolute** model Profit Factor (visual reference only, NOT used for gating). `ΔPF p5` is the 5th percentile of the `PF_model − PF_naive_long` bootstrap distribution. This is the true statistical gate: if ΔPF p5 > 0.0 (with ≥300 trades), the model outperforms a passive long position with 95% confidence. `#w_Δ>0` = number of individual OOS windows where the model beat naive long.

**Decision Matrix:**

| Outcome | Action |
|---------|--------|
| `0/4` PASS across all combinations | **Reject asset.** No signal detectable with control features. Do not waste time on A/B feature experiments. |
| `1-2/4` PASS (only 1 timeframe or 1 formulation) | **Conditional.** If both PASS results share the same timeframe, that timeframe exhibits structure; experiment exclusively on that timeframe. |
| `2+/4` PASS (multiple combinations PASS) | **Approve asset.** Proceed to Phase 3 (A/B testing orthogonal features). Recommended: select winning formulation for subsequent research. |

#### 2.3 Lessons Learned (BTC vs ETH)

Repository findings (see `tools/legacy_archive/exp_eth_baseline_oos.py`):

```
BTC_USDT is relatively more Efficient than ETH_USDT on 4h when
evaluating DELTA vs naive_long (the correct statistical criterion):
  • BTC 4h: ΔPF p5 typically ~0.00 to +0.03 (marginal, few PASS results)
  • ETH 4h: ΔPF p5 typically ~+0.02 to +0.06 (more consistent PASS results)

Conclusion: High market-cap, deep-liquidity assets exhibit fewer inefficiencies
relative to buy-and-hold. It is easier to extract alpha in ETH, SOL, MATIC
than in BTC using the same feature set. The ΔPF_p5 > 0.0 gate automatically
filters this efficiency differential WITHOUT contamination from market directional drift.
```

> **⚠️ Historical Correction:** Prior to gate remediation (August 2026), the baseline used absolute PF p5 > 1.0. This artificially inflated PASS rates on assets with sustained bullish drift: a model with PF=1.02 might appear as a PASS due to market long bias, while actually underperforming naive_long (ΔPF < 0). Always evaluate ΔPF, never absolute PF.

### 🔴 Why do we do this?

**Mathematical Rationale:** The `ΔPF p5 > 0.0` gate (model − naive_long) isolates classifier alpha from directional market drift. In a sustained bull market, even an untrained random classifier exhibits absolute PF > 1.0 (defaulting to long trades). The ΔPF metric removes this bias: we ask, "does the model IMPROVE upon passive buy-and-hold with 95% confidence?". If `ΔPF p5 = +0.0312`, in 95% of bootstrap scenarios the model outperforms naive long by at least +0.0312.

**Quantitative Rationale:** `pooled_trades ≥ 300` is non-arbitrary. Profit Factor is a ratio whose asymptotic variance is inversely proportional to trade count. With fewer than 300 pooled trades, the bootstrap confidence interval is SO wide that a true ΔPF of +0.05 may yield `p5 < 0.0` (false negative). At 300+ trades, standard error stabilizes and Type-I error rates remain controlled.

**Engineering Rationale:** Baseline screening prevents the "sunk cost fallacy". A `0/4` PASS baseline (ΔPF p5 > 0.0) is an unequivocal signal: the asset is either too efficient for this framework, or naive_long already captures available inefficiency. Any subsequent A/B test would guarantee p-hacking.

---

## Phase 3 — Feature Experimentation (A/B Test)

### Objective

Evaluate whether ONE new orthogonal feature statistically improves baseline Profit Factor under identical conditions (same train/val/test splits, same hyperparameters, same threshold grid).

### Inviolable Golden Rule

> Each experiment modifies EXACTLY one variable: it adds a single column to the control feature set.
>
> ❌ Forbidden: changing swing_period AND adding trend_htf in the same experiment.
> ❌ Forbidden: adding funding_rate AND taker_buy_ratio simultaneously.
> ❌ Forbidden: modifying XGBoost hyperparameters between control and treatment.
> ✅ Allowed: 14 control features → 15 features (14 + 1 new feature).

### Operational Steps

#### 3.1 Registering the Feature Profile

In `src/pipeline/feature_profiles.py`:

```python
# Example: feature "volume_delta_1h" (Template)
FEATURE_PROFILES["volume_delta_1h"] = FeatureProfile(
    name="volume_delta_1h",
    enrichments=("technicals", "sentiment", "volume_delta_1h"),
    treatment_col="volume_delta_1h",
    extra_csv_requirements=("1h.csv",),  # specify if extra timeframe CSV required
)

ENRICHMENT_REGISTRY["volume_delta_1h"] = _apply_volume_delta_1h
```

`treatment_col` is the column that:
- Is INCLUDED in the dataset (`df`)
- Is EXCLUDED from `control_features`
- Is ADDED to `treatment_feats = control_feats + [treatment_col]`

This ensures an exact apples-to-apples comparison on identical DataFrames.

#### 3.2 Preventing Leakage — Creating Integration Tests

Before running a single walk-forward validation fold, create a test in `tests/features/` verifying the feature contains NO future data.

Example (based on `test_funding_rate_leakage.py`):

```python
def test_volume_delta_no_lookahead():
    # Setup: mock dataframe with volume_delta_1h
    # Verify: for bar i, volume_delta_1h[i] DOES NOT depend on close[i+1] or any future bar
    # Verify: merge_asof(direction='backward') NEVER uses 'forward'
    # Verify: Daily HTF data is shifted +1 day BEFORE merge
    ...
```

**If test fails → FIX the feature before continuing. There is no such thing as "minor leakage". Look-ahead leakage completely invalidates PF results.**

#### 3.3 Executing the A/B Test

```bash
# A/B test: trend_htf on BTC_USDT, 4h and 1h timeframes
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# With prior data download (without --binance-rest; see §1.1)
python -m tools.aq ab-test BTC_USDT --profile funding_rate --fetch --timeframes 4h 1h
```

**Internal Pipeline (per timeframe × formulation):**

```
build_dataset(profile=TREATMENT_PROFILE) → df_common (contains treatment column)
  │
  ├─ VARIANT=CONTROL: features = control_feats (14 cols, WITHOUT treatment)
  │    └─ run_walk_forward → PF_control, trades_control
  │
  ├─ VARIANT=TREATMENT: features = control_feats + [treatment_col] (15 cols)
  │    └─ run_walk_forward → PF_treatment, trades_treatment
  │
  └─ Paired Block Bootstrap:
       For each OOS window (same timestamp range):
           ΔPF_w = PF_treatment,w − PF_control,w
       Resample 1000 times using 8 contiguous blocks per window
       → ΔPF_p5, ΔPF_p95
       → passes_gate = (trades ≥ 300) AND (ΔPF_p5 > 0.0)
```

#### 3.4 Interpreting Results

```
====================================================================================
SUMMARY — BTC_USDT A/B  profile=funding_rate
====================================================================================
4h × binary_homerun    CONTROL      p5=-0.0021  trades=412  gate=FAIL
4h × binary_homerun    TREATMENT    p5=+0.0188  trades=418  gate=PASS ← Δp5=+0.0209
4h × multiclass_3      CONTROL      p5=-0.0112  trades=389  gate=FAIL
4h × multiclass_3      TREATMENT    p5=+0.0799  trades=395  gate=PASS ← Δp5=+0.0911 ⭐
1h × binary_homerun    CONTROL      p5=-0.0301  trades=612  gate=FAIL
1h × binary_homerun    TREATMENT    p5=-0.0289  trades=618  gate=FAIL
1h × multiclass_3      CONTROL      p5=-0.0015  trades=598  gate=FAIL
1h × multiclass_3      TREATMENT    p5=-0.0102  trades=604  gate=FAIL
====================================================================================
```

> **⚠️ `p5` Column Semantics:** The `p5` column displays the **paired block bootstrap ΔPF p5** — the 5th percentile of `PF(variant) − PF(naive_long)`. It is **not** the absolute PF of that variant. A `p5=-0.0021` for CONTROL indicates control outperforms naive long by at least -0.0021 with 95% confidence (equal to or slightly worse than naive). A `p5=+0.0188` for TREATMENT indicates treatment outperforms naive long by at least +0.0188 with 95% confidence.

**Decision Matrix:**

| Outcome | Action |
|---------|--------|
| `0/8` combinations PASS | **Failed feature.** Archive script to `tools/legacy_archive/`. Do not re-test. |
| `1-2/8` PASS but limited to ONE specific tf × formulation | **Suspicious result.** Potential accidental overfitting to that specific setup. Replicate with different `random_state` or `n_bootstrap=5000`. If it continues to pass → promote. |
| `3+/8` combinations PASS (cross-formulation consistency) | **Successful feature.** Promote to production pipeline. Do NOT add to control profile for future baselines immediately: maintain 1 feature per experiment protocol. |

#### 3.5 Archiving Failed Experiments

EVERY experiment that fails the statistical gate MUST be archived under `tools/legacy_archive/`. The archived script preserves the exact code used to execute the test.

**Rationale:** Long-term meta-analysis. If 3 volatility-based features failed, do not propose another volatility feature without fundamental modifications. Archiving prevents repeating past mistakes months later.

**Repository History (Lessons Learned):**

- **Exp01 trend_htf:** 0/8 PASS. Distance to Daily EMA200 does NOT improve PF vs naive long.
- **Exp02 funding_rate_current:** 1/8 PASS (4h×multiclass_3 only, Δp5=+0.0799). Result statistically indistinguishable from noise given test volume (~1.2 expected false positives across 24 evaluated configs). Lacks cross-formulation support on the same timeframe (4h×binary degrades with the feature). Classified as DISCARDED.
- **Exp03 taker_buy_ratio:** 0/8 PASS. Point-in-time aggressor buy volume ratio is not an orthogonal signal vs naive long.
- **Exp04 regression_return:** 0/6 PASS. Continuous return regression formulation (`target_ret`). 0/6 configs (3 assets × 2 TFs) pass `ΔPF_p5 > 0.0` after correcting the sentinel bug (`THRESHOLD_NOT_FOUND = -1.0`). Prediction variance displays massive compression (~26× vs real target variance). Discarded.

### 🔴 Why do we do this?

**Mathematical Rationale:** Paired block bootstrapping validates the gate logic. If independent bootstrapping were used for control and treatment (sampling individual trades i.i.d.), we would severely underestimate covariance (both variants run on identical folds!). Pairing isolates: "given the EXACT SAME temporal history, how much better is treatment?". In statistics, this matched-pairs design maximizes power to detect subtle signals.

**Quantitative Rationale:** Testing "one orthogonal feature per experiment" derives from degrees of freedom principles. If 3 features are added simultaneously and ΔPF p5 = +0.08, which feature drove the gain? All 3? 1 feature? A 2-way interaction? Isolating k=1 variable per test ensures direct causal attribution.

**Engineering Rationale:** Leakage testing prevents catastrophic errors: passing OOS gates, deploying to production, and discovering 3 months later that the "successful" feature used next-bar close prices. What appeared to be alpha was pure look-ahead leakage, resulting in live losses (PF=0.5).

---

## Phase 4 — Integration and Monitoring (Production)

### Objective

Deploy validated models (passing baseline and A/B gates) into daily evaluation loops: signal detection → Telegram alert → optional execution on Binance Futures.

### Operational Steps

#### 4.1 Training Final Production Models

```bash
# Execute strategy_optimizer.py for approved symbol
python -m src.brain.strategy_optimizer BTC_USDT --timeframe 4h
```

This writes `data/models/BTC_USDT/config.json`:

```json
{
  "strategy_name": "multiclass_3_control_fundingrate",
  "features": ["rsi_14", "macd", "...", "fng_vol_14", "funding_rate_current"],
  "optimal_threshold": 0.42,
  "swing_period": 10,
  "atr_tp_multi": 1.5,
  "atr_sl_multi": 1.0,
  "n_estimators": 200,
  "max_depth": 3,
  "learning_rate": 0.05,
  "passed_oos_sanity_check": true,
  "last_trained": "2026-08-12T00:00:00Z"
}
```

Then train:

```bash
python -m src.brain.train BTC_USDT --timeframe 4h
```

Generates joblib-serialized `BTC_USDT_1_5_1_0_10_0-42.pkl`.

#### 4.2 Environment Configuration (`.env`)

```bash
cp .env.example .env
# Edit credentials:
TELEGRAM_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
AUTHORIZED_CHAT_ID=987654321
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=True  # Always TEST on Testnet first
```

#### 4.3 Registering Symbols in `bot_state.json` (via Telegram)

```
User: /start → Bot sends main menu
Bot → [Bot] → [➕ Add Symbol] → User inputs: ETH_USDT
    → ✅ Symbol added successfully.
```

This updates `data/bot_state.json`:

```json
{
  "symbols": {
    "futures": ["BTC_USDT", "ETH_USDT"]
  }
}
```

#### 4.4 Starting Bot + Scheduler

```bash
python main.py
```

APScheduler runs `daily_market_evaluation()` daily at **21:00 ART (America/Argentina/Cordoba)**.

**Daily Execution Flow:**

```
For each .pkl in data/models/{SYMBOL}/*.pkl:
  1. fetch_ohlcv_binance(limit=100) → latest 100 timeframe candles
  2. compute_all_technicals + add_sentiment
  3. model.predict_proba(last_candle[features])[0, class_1]
  4. IF proba >= optimal_threshold:
       a. TP = close + (atr_14 × atr_tp_multi)
       b. SL = close − (atr_14 × atr_sl_multi)
       c. send_trade_signal → Telegram (HTML: Pair / Strategy / Entry / TP / SL)
       d. IF executor initialized:
           i.  _has_open_position(symbol) → if open → skip (no averaging)
           ii. _configure_symbol → ISOLATED margin, x2 leverage
           iii. _calculate_quantity → 1% balance, step_size compliance
           iv.  MARKET entry order
           v.   STOP_MARKET + TAKE_PROFIT_MARKET (closePosition=TRUE)
           vi. send_execution_result → Telegram
```

#### 4.5 Bot Monitoring Commands

| Menu → Action | Description |
|---------------|-------------|
| **Bot → Status** | Active/Paused status, N monitored symbols, next scheduled run |
| **Bot → Pause / Resume** | Toggles `bot_active` in `bot_state.json`. Paused = skips signal evaluation. |
| **Bot → Train** | Triggers background `run_full_training_pipeline` for all symbols. Skipped if last_trained < 14 days. |
| **Futures → Balance** | Queries available USDT balance on Futures via API. |
| **Futures → Positions** | Lists active open positions (positionAmt != 0). |
| **Futures → Scan** | On-demand `daily_market_evaluation()` execution. |
| **Futures → Leverage** | Modify `default_leverage` (range: 1–125). Persisted in `bot_state.json`. |
| **Futures → Risk %** | Modify `risk_per_trade_pct` (0.01–100). Default 1% rule. |
| **Futures → Margin Toggle** | ISOLATED ↔ CROSS margin. **Recommended: always ISOLATED.** |
| **Futures → ⚠️ PANIC BUTTON** | Confirmation → `close_all_positions()`: MARKET close for ALL open positions + cancels ALL open orders. |

### 🔴 Why do we do this?

**Mathematical Rationale:** The `1% balance × leverage` sizing rule derives from fractional Kelly Criterion principles. Full Kelly sizing is overly aggressive for models with PF ≤ 1.3 due to return noise. Sizing at 1% limits ruin probability (drawdown > 50%) to < 0.1%, assuming PF remains in the [1.05, 1.3] bootstrap range.

**Quantitative Rationale:** `ISOLATED margin` over `CROSS` is a hard requirement. Under CROSS margin, a single large loss risks liquidating the entire account balance. Under ISOLATED margin, collateral is strictly segregated per trade (`margin = notional / leverage`). Risk remains strictly compartmentalized.

**Engineering Rationale:** The `Training Cooldown = 14 days` rule balances two competing dynamics:
- **Excessive Retraining (daily):** Risk of "data drift overfitting" — frequent retraining introduces weight volatility without structural changes in underlying market distributions.
- **Insufficient Retraining (>90 days):** Model becomes stale during structural regime shifts (e.g., ETF approvals, halvings).
- **14 days (2 weeks):** Optimal balance capturing gradual market drift without introducing noise.

---

## Phase 5 — Code Maintenance and Testing

### Objective

Preserve long-term pipeline integrity. Unit and integration tests detect regressions, legacy archives prevent duplicate failed experiments, and CI checks ensure code committed to main remains production-ready.

### 5.1 Test Suite Structure (`pytest`)

```
tests/
├── unit/               # Function-level unit tests
│   ├── test_helpers.py            # compute_target Numba, profit_factor
│   ├── test_features.py           # Indicators: RSI, MACD, BB, OBV, EMA
│   ├── test_data_splits.py        # Embargoed dynamic splits
│   ├── test_settings_loader.py    # YAML + bot_state merging
│   └── test_logging_config.py
│
├── integration/        # Cross-module integration tests
│   ├── test_oos_validation.py     # Full walk-forward validation with mock data
│   ├── test_strategy_optimizer.py # Grid search + OOS sanity check
│   ├── test_train.py              # Train factory + pkl serialization
│   ├── test_data_fetcher.py       # Mocked ccxt fetcher
│   └── test_tasks.py              # Orchestrator with Binance/Telegram mocks
│
├── features/           # LEAKAGE PREVENTION TESTS (CRITICAL)
│   ├── test_funding_rate_leakage.py
│   ├── test_trend_htf_leakage.py
│   └── test_taker_buy_ratio_semantics.py
│
├── api/                # External API integration tests
│   ├── test_binance_executor.py   # Sizing + mock exchange filters
│   ├── test_notifier.py           # HTML formatting verification
│   └── test_telegram_handlers.py  # Auth + state machine
│
└── conftest.py         # Shared fixtures (mocks, test dataframes)
```

#### 5.2 Running the Test Suite

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=src --cov-report=html

# Run leakage tests exclusively (MANDATORY before A/B testing)
pytest tests/features/ -v

# Run research pipeline tests
pytest tests/test_oos_validation.py tests/test_strategy_optimizer.py -v
```

### 5.3 Pre-Merge Checklist

**BEFORE** merging any feature branch into `main`:

```
□ Full pytest suite passes 100% without critical skips
□ tests/features/*_leakage.py tests pass (anti-lookahead verification)
□ New A/B profiles are registered in feature_profiles.py
□ HTF/external features use merge_asof(direction='backward')
□ Experiment passed ΔPF_p5 > 0.0 gate in ≥ 2/8 evaluated combinations
□ Failed experiment scripts moved to tools/legacy_archive/ with outcome comments
□ Documentation updated if public CLI parameters modified
```

### 5.4 Legacy Code Archiving

Rule: **DO NOT DELETE failed experiment code.** Move scripts to `tools/legacy_archive/` with a header comment detailing failure causes.

Rationale: In 6 months you may reconsider testing daily EMA200 distances. Having `legacy_archive/exp01_trend_htf_walkforward.py` annotated with `0/8 gate PASS, no orthogonal alpha` saves weeks of redundant effort.

**Current `legacy_archive` Inventory:**

| File | Result | Lesson Learned |
|------|--------|----------------|
| `exp01_trend_htf_walkforward.py` | 0/8 PASS | Daily EMA200 does not add orthogonal value vs naive_long |
| `exp02_funding_rate_walkforward.py` | 1/8 PASS (4h×multiclass3 only) | 1/8 PASS is statistically indistinguishable from noise (~1.2 expected FPs across 24 configs). Lacks cross-formulation support. Discarded. |
| `exp03_taker_buy_ratio_walkforward.py` | 0/8 PASS | Point-in-time taker buy ratio is not an orthogonal signal vs naive |
| `exp04_regression_return_walkforward.py` | 0/6 PASS | Continuous return regression formulation discarded after correcting sentinel bug (-1.0). All fail ΔPF_p5 > 0.0 and exhibit severe prediction variance compression (~26×). |
| `compare_binary_vs_multiclass.py` | Internal benchmark | Multiclass 3 displays slightly better p5 in volatile assets |
| `exp_eth_baseline_oos.py` | ETH Baseline | ETH > BTC in ΔPF vs naive_long |
| `reconcile_naive_target_comparison.py` | Target Debugging | Numba vs Python target implementations match exactly |

### 5.5 Post-Production Monitoring

#### Weekly Performance Review (Every 7 Days)

Query live trade history from Binance:

```bash
# Calculate live performance metrics from Binance exports
python - <<'EOF'
import pandas as pd, numpy as np

# trades = DataFrame containing columns: side, realizedPnl, time
trades = pd.read_csv("trades_export.csv")  # Binance UI export

wins   = trades["realizedPnl"][trades["realizedPnl"] > 0].sum()
losses = trades["realizedPnl"][trades["realizedPnl"] < 0].abs().sum()
pf_real = wins / max(losses, 1e-9)
print(f"Live PF = {pf_real:.3f}  (trades={len(trades)})")
EOF
```

Compare `pf_real` against `oos_pf_point` recorded in `reports/{SYMBOL}/latest_baseline.json`.

#### Backtest vs Live Performance Matrix

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| `pf_real ≈ oos_pf_p5` (±10%) | Model operating within expectation | Continue live trading |
| `pf_real < oos_pf_p5 × 0.9` for 1 week | Potential short-term noise | Monitor closely |
| `pf_real < 0.85` for 2+ weeks | Likely regime shift or silent issue | Pause bot, re-run baseline screening |
| `pf_real > oos_pf_p95 × 1.2` | Favorable market conditions | Continue; do not increase risk parameters |

#### Emergency Shutdown Procedure

```
If live PF < 0.85 for 4 consecutive weeks:
  1. Telegram → [Futures] → [⚠️ PANIC BUTTON] → Confirm
     (Closes ALL open positions)
  2. Telegram → [Bot] → [Pause]
     (Bot halts daily evaluation loops)
  3. In bot_state.json: Remove problematic symbol from symbols.futures
  4. Re-run baseline screening:
     python -m tools.aq baseline {SYMBOL} --timeframes 4h 1h --fetch
  5. If baseline continues to pass: investigate threshold drift
     (Run strategy_optimizer with recent data)
     If baseline fails: asset market regime changed. Discard asset.
```

**Re-activation Rule:** Only re-enable trading for a symbol if new baseline screening yields `ΔPF_p5 > 0.0` (model vs naive_long) on data including the underperformance period.

### 🔴 Why do we do this?

**Mathematical Rationale:** Leakage tests represent the defense against silent regressions. If `add_sentiment` were modified and `direction='backward'` accidentally changed to `direction='nearest'`, unit tests might pass while 22:00 UTC candles inherit next-day F&G values. Backtest PF inflates artificially from 1.05 to 1.25, while live production fails. Only explicit leakage tests in `tests/features/` catch these regressions.

**Quantitative Rationale:** Pytest suites control Type-I error rates. In algorithmic trading systems, false positives (deploying flawed models) carry costs orders of magnitude higher than false negatives (rejecting viable models due to strict testing).

**Engineering Rationale:** Legacy archives record dead ends. Over long research lifecycles, maintaining documented records of failed experiments prevents re-exploring unproductive approaches.

---

## Appendix A — Command Cheat Sheet

```bash
# === DATA ===
python -m src.brain.data_fetcher BTC_USDT --timeframe 4h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --funding-rate

# === DIAGNOSTICS ===
python -m tools.aq diagnose-data BTC_USDT --timeframe 4h
python3 tools/diagnostics/diagnose_naive_baseline.py

# === RESEARCH ===
python -m tools.aq baseline ETH_USDT --timeframes 4h 1h --fetch
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# === PRODUCTION ===
python -m src.brain.strategy_optimizer BTC_USDT --timeframe 4h
python -m src.brain.train BTC_USDT --timeframe 4h
python main.py  # Starts bot + 21:00 ART scheduler

# === TESTS ===
pytest
pytest tests/features/ -v                          # Anti-leakage
pytest tests/test_oos_validation.py -v             # Walk-Forward
```

---

## Appendix B — Repository Success Criteria Reference

| Criterion | Pre-registered Value | Current Implementation |
|-----------|----------------------|------------------------|
| MINIMUM `pooled_trade_count` | `≥ 300` | Hardcoded in `oos_validation.py` |
| Baseline Gate (ΔPF vs naive) | `ΔPF_p5 > 0.0` | Hardcoded in `experiment_defaults.py` (`MIN_BASELINE_DELTA_P5 = 0.0`) and `oos_validation.py` (`MIN_BOOTSTRAP_P5 = 0.0`) — baseline and A/B share identical paired bootstrap mechanism vs naive_long. Absolute PF > 1.0 is NOT a gate. |
| A/B Test Gate (Delta PF) | `ΔPF_p5 > 0.0` | Hardcoded in `oos_validation.py` (`MIN_BOOTSTRAP_P5 = 0.0`) |
| Bootstrap iterations | `1000` | Default in `ExperimentConfig.n_bootstrap` |
| Blocks per window | `8` | Default in `ExperimentConfig.n_blocks` |
| Random state | `42` | Default in `ExperimentConfig.random_state` |
| Default swing period | `10 bars` | Default in `ExperimentConfig.swing_period` |
| Default TP ATR multiplier | `1.5 × ATR` | Default in `ExperimentConfig.tp_multi` |
| Default SL ATR multiplier | `1.0 × ATR` | Default in `ExperimentConfig.sl_multi` |
| Training window duration | `6 months` | Default in `ExperimentConfig.window_months` |
| Step between folds | `6 months` | Default in `ExperimentConfig.step_months` |
| Training cooldown | `14 days` | Constant `TRAINING_COOLDOWN_DAYS` in `tasks.py` |
| Risk per trade | `1%` of balance | `risk_per_trade_pct` in `settings.yaml` |
| Default leverage | `2x` ISOLATED | `default_leverage` in `settings.yaml` |
| Daily Scheduler | `21:00 ART` | Cron trigger in `main.py` + `timezone("America/Argentina/Cordoba")` |
