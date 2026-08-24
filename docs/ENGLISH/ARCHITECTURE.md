# AlphaQuant — General System Architecture

> Production-grade technical specification. Version 1.0 — August 2026.

---

## 1. System Overview

AlphaQuant is an **end-to-end** quantitative platform designed for algorithmic cryptocurrency trading on Binance (USD-M Futures and Spot). The system spans the complete lifecycle from market data ingestion and supervised Machine Learning (XGBoost) research using Out-of-Sample (OOS) Walk-Forward validation, to automated order execution and a Telegram alerting system.

### Functional Scope

| Layer | Responsibility |
|-------|----------------|
| **Research / ML** | Feature engineering, asset screening, feature A/B testing, hyperparameter optimization, OOS validation with paired block bootstrapping. |
| **Execution** | Binance API connection, ISOLATED margin management, 1% risk sizing rule, MARKET entry orders + conditional STOP_MARKET / TAKE_PROFIT_MARKET orders. |
| **Alerts / UX** | Telegram bot with inline keyboard interface: pause/resume bot, add/remove symbols, adjust risk/leverage, panic button, check balance and open positions. |
| **Orchestration** | Daily APScheduler (21:00 ART) evaluating models, broadcasting signals to Telegram, and executing trades on Binance. Automated retraining every 14 days. |

---

## 2. Directory Tree

```
AlphaQuant/
├── src/                              # Production source code
│   ├── api/                          # External API integrations
│   │   ├── binance/
│   │   │   └── binance_executor.py   # Futures order execution (ISOLATED + algorithmic SL/TP)
│   │   └── telegram/
│   │       ├── handlers.py           # ConversationHandler router (auth, callbacks, states)
│   │       ├── _actions.py           # Logic implementation for menu buttons/actions
│   │       ├── _ui.py                # Inline keyboards and text formatting (main menu, futures, bot)
│   │       └── notifier.py           # Asynchronous signal and execution result notifier
│   ├── brain/                        # Research and ML engine
│   │   ├── data_fetcher.py           # OHLCV fetcher (ccxt + Binance REST), Funding Rate, Fear&Greed
│   │   ├── features.py               # Technical indicators + merge_asof for external features (F&G, HTF, Funding, Taker Buy)
│   │   ├── strategy_optimizer.py     # Parameter grid search (swing, ATR TP/SL, XGB HP)
│   │   └── train.py                  # Train factory: final XGBoost model + .pkl serialization
│   ├── config/                       # Configuration and path management
│   │   ├── paths.py                  # Centralized path resolution (CSVs, models, reports)
│   │   ├── settings_loader.py        # Settings merge: settings.yaml (RO) + bot_state.json (RW)
│   │   └── experiment_defaults.py    # ExperimentConfig, FORMULATIONS, pre-registered statistical gates
│   ├── engine/
│   │   └── tasks.py                  # Daily orchestrator: evaluate models → signal → execution → notification
│   ├── pipeline/                     # Reproducible experiment pipeline
│   │   ├── feature_profiles.py       # Declarative profile registry (control, trend_htf, funding_rate, taker_buy_ratio)
│   │   ├── dataset_builder.py        # load_csv → enrichments → compute_target → dropna
│   │   └── walkforward_runner.py     # run_baseline / run_ab_test + JSON reporting
│   └── utils/                        # Domain utilities
│       ├── helpers.py                # compute_target (Numba JIT), train_predict_* formulations, Profit Factor
│       ├── oos_validation.py         # run_walk_forward, paired block bootstrap, statistical gates
│       ├── data_splits.py            # compute_dynamic_split, embargoed train/val/test splits
│       ├── timeframe_utils.py        # Timeframe parsing (1h, 4h, 1d) to hours
│       └── logging_config.py         # Centralized logging configuration
│
├── tools/                            # CLI scripts, diagnostics, and legacy experiments
│   ├── aq.py                         # Unified CLI: baseline / ab-test / diagnose-data / diagnose-naive-baseline / diagnose-regimes-rigorous / diagnose-swing-and-regimes / diagnose-timeframe-swing-sweep
│   ├── visualize_val_signals.py      # Validation signal plotting
│   ├── legacy_archive/               # Failed or superseded experiments & archived diagnostics
│   │   ├── diagnostics/              # Archived diagnostic scripts (now in aq.py)
│   │   │   ├── diagnose_naive_baseline.py         # Use: aq diagnose-naive-baseline
│   │   │   ├── diagnose_regimes_rigorous.py       # Use: aq diagnose-regimes-rigorous
│   │   │   ├── diagnose_swing_and_regimes.py      # Use: aq diagnose-swing-and-regimes
│   │   │   ├── diagnose_timeframe_data.py         # Use: aq diagnose-data
│   │   │   ├── diagnose_timeframe_swing_sweep.py  # Use: aq diagnose-timeframe-swing-sweep
│   │   │   └── README.md             # Archive documentation
│   │   ├── exp01_trend_htf_walkforward.py         # 0/8 PASS — daily EMA200 no orthogonal alpha
│   │   ├── exp02_funding_rate_walkforward.py      # 1/8 PASS — noise level, discarded
│   │   ├── exp03_taker_buy_ratio_walkforward.py   # 0/8 PASS — taker ratio not orthogonal
│   │   ├── exp04_regression_return_walkforward.py # 0/6 PASS — regression formulation discarded
│   │   ├── compare_binary_vs_multiclass.py
│   │   ├── exp_eth_baseline_oos.py
│   │   └── reconcile_naive_target_comparison.py
│
├── tests/                            # Pytest suite (unit + integration + leakage)
│   ├── unit/
│   │   ├── test_helpers.py                # compute_target (Numba), profit_factor
│   │   ├── test_features.py               # Indicators: RSI, MACD, BB, OBV, EMA
│   │   ├── test_data_splits.py            # Embargoed dynamic splits
│   │   ├── test_settings_loader.py        # YAML + bot_state merging
│   │   └── test_logging_config.py
│   ├── integration/
│   │   ├── test_oos_validation.py         # Full walk-forward with mock data
│   │   ├── test_strategy_optimizer.py     # Grid search + OOS sanity check
│   │   ├── test_train.py                  # Train factory + pkl serialization
│   │   ├── test_data_fetcher.py           # Mocked ccxt data fetcher
│   │   └── test_tasks.py                  # Orchestrator with Binance/Telegram mocks
│   ├── features/
│   │   ├── test_funding_rate_leakage.py      # Verify funding rate exists only prior to settlement
│   │   ├── test_trend_htf_leakage.py         # Verify 1d data shifted +1d before merge_asof
│   │   ├── test_taker_buy_ratio_semantics.py # Aggressor volume ratio semantic verification
│   │   ├── test_onchain_active_addresses_leakage.py  # Verify +2d shift for Blockchain.com daily data
│   │   └── test_mempool_fee_rate_leakage.py  # Verify +1d shift for mempool.space daily data
│   ├── api/
│   │   ├── test_binance_executor.py       # Sizing + mock exchange filters
│   │   ├── test_notifier.py               # HTML output format validation
│   │   └── test_telegram_handlers.py      # Authentication + state machine
│   └── conftest.py
│
├── data/                             # Local persistence (not versioned)
│   ├── raw_csv/                      # OHLCV + Funding Rate (by symbol and timeframe)
│   │   └── {SYMBOL_USDT}/
│   │       ├── 1h.csv | 4h.csv | 1d.csv
│   │       ├── funding_rate.csv
│   │       ├── onchain_active_addresses.csv        # Blockchain.com n-unique-addresses (BTC only)
│   │       └── onchain_mempool_fee_rate_p50.csv    # mempool.space fee-rates/all avgFee_50 (BTC only)
│   ├── models/                       # Serialized models + config.json per symbol
│   │   └── {SYMBOL_USDT}/
│   │       ├── config.json           # Features, threshold, HP, last_trained, OOS sanity check
│   │       └── {symbol}_{tp}_{sl}_{swing}_{threshold}.pkl
│   └── plots/
│
├── reports/                          # Experiment JSON reports (baselines, A/B tests)
│   └── {SYMBOL_USDT}/
│       ├── baseline_{timestamp}.json
│       ├── ab_test_{profile}_{timestamp}.json
│       └── latest_baseline.json      # Symlink/copy to most recent report
│
├── benchmarks/                       # Performance benchmarks (Numba helpers)
├── logs/                             # Rotated logs
├── main.py                           # Entry point: initializes bot + scheduler
├── settings.yaml                     # Read-only default settings
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 3. System Components

### 3.1 APIs and External Integrations

#### 3.1.1 Binance API (Ingestion + Execution)

**Modules:** `data_fetcher.py`, `binance_executor.py`

| Layer | Library | Endpoints / Methods |
|-------|---------|---------------------|
| **Standard OHLCV** | `ccxt` (synchronous) | `fetch_ohlcv` — 6 columns (OHLCV + timestamp). 1000 candles per page, exponential retry. |
| **Microstructure OHLCV** | `requests` (Binance REST native) | `/fapi/v1/klines` (Futures) or `/api/v3/klines` (Spot). 12 fields: `quote_volume, n_trades, taker_buy_base_vol, taker_buy_quote_vol`. Flag `--binance-rest`. |
| **Funding Rate** | `ccxt.binanceusdm` | `fetch_funding_rate_history` — 8h settlements (00/08/16 UTC). |
| **Futures Execution** | `python_binance` (Client) | `futures_create_order` (MARKET entry) + Algo Orders API (`STOP_MARKET`, `TAKE_PROFIT_MARKET` with `closePosition=TRUE` and `workingType=MARK_PRICE`). |
| **RT Candles (daily eval)** | `ccxt.async_support.binanceusdm` | `fetch_ohlcv(limit=100)` asynchronous unauthenticated fetch. |

**Executor Business Rules:**

```python
# Maximum 1 concurrent position per symbol (no position averaging)
_has_open_position(symbol) → TRUE → skip new trade

# Position Sizing: 1% of available USDT balance × leverage
margin     = balance * 0.01
notional   = margin * leverage
quantity   = (notional / price) → ROUND_DOWN respecting LOT_SIZE step_size

# Exchange Filters (NOT hardcoded)
LOT_SIZE: step_size, min_qty
MIN_NOTIONAL: minimum notional
PRICE_FILTER: tick_size for SL/TP price string formatting
```

#### 3.1.2 Telegram Bot (Alerts + Operational UX)

**Modules:** `handlers.py`, `_actions.py`, `_ui.py`, `notifier.py`

**Separation of Concerns Architecture:**

```
handlers.py (router) → _actions.py (logic) → _ui.py (presentation)
                           ↓
                     notifier.py (message dispatch)
```

| Layer | Description |
|-------|-------------|
| **Authentication** | `AUTHORIZED_CHAT_ID` (integer) from `.env`. `_is_authorized()` checks every callback and command. Unauthorized users receive "Access Denied". |
| **ConversationHandler** | State machine: `NAVIGATING` (menu), `WAITING_ADD_SYMBOL`, `WAITING_REMOVE_SYMBOL`, `WAITING_LEVERAGE`, `WAITING_RISK`. |
| **Main Menu** | 4 inline sections: Bot, Exchange (placeholder), Futures, Spot (placeholder). |
| **Panic Button** | `action:panic` → confirmation → `close_all_positions()`: MARKET orders to close open positions + `futures_cancel_all_open_orders`. |
| **Automated Notifications** | `send_trade_signal` (new signal detected), `send_execution_result` (trade executed or skipped), `send_execution_error` (execution exception). |

---

## 3.2 Machine Learning Engine

#### 3.2.1 Target Definition (Ternary)

**Function:** `helpers.py → compute_target` (Numba JIT)

```
3-Class Ternary Classification (y ∈ {-1, 0, +1}):
  +1 = Take Profit hit first (1.5 × ATR)
   0 = Timeout: neither TP nor SL hit within swing_period bars
  -1 = Stop Loss hit first (1.0 × ATR)

Tie-break: SL always wins if both levels are touched in the SAME bar
           (prevents optimism during high volatility candles).
```

#### 3.2.2 Formulations (XGBoost)

**Defined in:** `experiment_defaults.py → FORMULATIONS`

| Formulation | Transformed Target | Model Type | Threshold Grid |
|-------------|--------------------|------------|----------------|
| `binary_homerun` | `y_binary = (y == +1)` vs `{0, -1}` | Binary `XGBClassifier` | `(0.50, 0.85, 0.01)` |
| `multiclass_3` | `y_multiclass = y + 1` → `{0,1,2}` | `XGBClassifier` multi:softprob | `(0.25, 0.70, 0.01)` (TP class probability) |
| `regression_return` | `target_ret` continuous realized return | `XGBRegressor` (reg:pseudohubererror) | `(-0.0035, 0.0070, 0.0003)` — DISCARDED (see legacy_archive/exp04_regression_return_walkforward.py) |

**Per-Fold Training Pipeline:**

```
Train split (raw)
  ├── X: 14 base features + treatment feature
  ├── y: binary or multiclass formulation
  │
  ├── HP defaults (production):
  │     n_estimators = 100/200
  │     max_depth    = 2, 3, 4
  │     learning_rate= 0.01, 0.05
  │     scale_pos_weight (binary only) = class imbalance ratio
  │
  └── Early stopping: 10% recent holdout, early_stopping_rounds=10

Val split
  └── Threshold grid search (maximizes net return sum with validation trade count floor)

Test split (OOS)
  └── Blind evaluation using validation optimal threshold (NEVER optimized on test)
```

#### 3.2.3 Feature Profile Registry

**Module:** `feature_profiles.py`

Declarative enrichment profile registry. All external features use `merge_asof(direction='backward')`.

| Profile | Enrichments | Treatment Col | Extra CSV Requirements |
|---------|-------------|---------------|------------------------|
| **control** | `technicals` + `sentiment` | None (baseline) | — |
| **trend_htf** | `technicals` + `sentiment` + `trend_htf` | `trend_htf` | `1d.csv` (Daily EMA200, shifted +1d) |
| **funding_rate** | `technicals` + `sentiment` + `funding_rate` | `funding_rate_current` | `funding_rate.csv` (past settlements only) |
| **taker_buy_ratio** | `technicals` + `sentiment` + `taker_buy_ratio` | `taker_buy_ratio` | CSVs downloaded with `--binance-rest` |
| **onchain_activity** | `technicals` + `sentiment` + `onchain_active_addresses` | `onchain_active_addresses` | `onchain_active_addresses.csv` (Blockchain.com, shift +2d) |
| **onchain_fee_pressure** | `technicals` + `sentiment` + `mempool_fee_rate_p50` | `mempool_fee_rate_p50` | `onchain_mempool_fee_rate_p50.csv` (mempool.space, shift +1d) |

**14 Baseline Control Features:**
```
Momentum:    rsi_14, macd, macd_hist, stoch_k
Trend:       dist_ema_50, adx_14
Volatility:  atr_14, bb_width, bb_pos
Volume:      obv, rel_volume
Sentiment:   fng_value, fng_sma_14, fng_vol_14
```

---

## 3.3 Pipeline and CLI Architecture

#### 3.3.1 Dataset Builder

**Module:** `dataset_builder.py → build_dataset`

```python
Deterministic Flow:
  1. load_csv_data(symbol, timeframe)
  2. for enrichment in profile.enrichments:
       df = ENRICHMENT_REGISTRY[enrichment](df, symbol)
  3. compute_target(swing=10, TP=1.5×ATR, SL=1.0×ATR)
  4. drop(columns=COLS_TO_DROP)    # 9 columns — see table below
  5. dropna()                       (removes bars missing targets/features)
  6. Feature Column Inference:
       - control_features = ALL numeric columns
                               − {close, target, treatment_col}
       - REQUIRED_BASE_FEATURES health check (6 cols)
       - SENTIMENT_COLS health check (3 cols)
```

**Full `COLS_TO_DROP` (9 columns):**

| Column | Rationale |
|--------|-----------|
| `open`, `high`, `low` | Raw OHLCV — model operates on derived indicators |
| `volume` | Replaced by `rel_volume` (normalized relative volume) |
| `ema_50` | Replaced by `dist_ema_50` (percentage distance) |
| `vol_sma_20` | Replaced by `rel_volume` |
| `max_high_future`, `min_low_future` | Auxiliary target computation columns — absolute look-ahead leakage if retained |
| `quote_volume`, `n_trades`, `taker_buy_base_vol`, `taker_buy_quote_vol` | Microstructure fields fetched via `--binance-rest`. Dropped so base model does not consume raw fields directly. Consumed exclusively by `taker_buy_ratio` profile via derived feature. |

> **Note:** The 4 microstructure columns exist in the loaded DataFrame only when fetched with `--binance-rest`. `COLS_TO_DROP` uses `[c for c in COLS_TO_DROP if c in df.columns]` to avoid errors when missing.

**`REQUIRED_BASE_FEATURES` — 6-Column Health Check:**

Distinct from the 14 control features. These represent the minimum required columns that must exist post-enrichment to ensure the pipeline is not silently broken:

```python
REQUIRED_BASE_FEATURES = frozenset({
    "rsi_14", "atr_14", "bb_width", "bb_pos", "obv", "rel_volume"
})
```

If any column is missing, `build_dataset` raises a `RuntimeError` listing available columns.

#### 3.3.2 WalkForwardRunner

**Modules:** `walkforward_runner.py`, `oos_validation.py → run_walk_forward`

```
Default Parameters (ExperimentConfig):
  swing_period   = 10 bars
  tp_multi       = 1.5 × ATR
  sl_multi       = 1.0 × ATR
  window_months  = 6    (6-month rolling train window)
  step_months    = 6    (6-month step between folds)
  n_bootstrap    = 1000 iterations
  n_blocks       = 8    contiguous blocks per OOS window
  random_state   = 42
  fee_rate       = 0.0 (research mode; live execution uses 0.001)
  slippage       = 0.0 (research mode; live execution uses 0.0005)
```

#### 3.3.3 Unified CLI `aq.py`

**Entry point:** `tools/aq.py`

```bash
# === New Asset Screening ===
python -m tools.aq baseline ETH_USDT --timeframes 4h 1h --fetch

# === Orthogonal Feature A/B Testing ===
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# === Data Health Diagnostics ===
python -m tools.aq diagnose-data SOL_USDT --timeframe 4h --swing 10

# === Configuration Overrides ===
--swing-period 10 --tp-multi 1.5 --sl-multi 1.0
--window-months 6 --step-months 6
--fee-rate 0.001 --slippage 0.0005
--n-bootstrap 1000 --n-blocks 8 --random-state 42
```

---

## 4. Metrics and Statistical Gates

### 4.1 Profit Factor (PF)

Definition (per trade, not per bar):

```
          Σ (positive trade returns)
PF = ────────────────────────────────────
     |Σ (negative trade returns)|

Where each trade return = f(y_true[i]):
  y=+1  →  +(atr[i] × tp_multi) / close[i]  − cost_per_trade
  y=−1  →  −(atr[i] × sl_multi) / close[i]  − cost_per_trade
  y=0   →  (close[exit] − close[i]) / close[i] − cost_per_trade

cost_per_trade = 2×fee_rate + 2×slippage  (entry + exit, round-trip)
```

### 4.2 Statistical Bootstrap

The system estimates **ΔPF = PF(treatment) − PF(naive)** distribution using **paired blocks** via `_bootstrap_paired_blocks`:

#### 4.2.1 `_bootstrap_paired_blocks` — A/B Test

**Implementation:** `oos_validation.py → _bootstrap_paired_blocks`

Used in `run_ab_test` (via `run_walk_forward`). Estimates the **ΔPF = PF(treatment) − PF(naive)** distribution using **paired blocks**: each iteration samples the **same block indices** for model and naive baselines, removing market variance.

**Why blocks?** Financial returns exhibit **serial autocorrelation** (volatility clustering). Independent i.i.d. bootstrapping underestimates true variance. Contiguous blocks preserve temporal dependency structures.

```
Global Paired Block Pool (model + naive, same window, same block):
  1. For each OOS window w: divide model and naive trades into B=8 contiguous blocks
  2. Add all block pairs (block_model_i, block_naive_i) to global pool
  3. For each bootstrap iteration b ∈ {1..1000}:
       a. Sample with replacement n_total_blocks indices (SAME for model and naive)
       b. Concatenate blocks_model[idx] → rets_mdl_b
       c. Concatenation blocks_naive[idx] → rets_nav_b
       d. ΔPF_b = PF(rets_mdl_b) − PF(rets_nav_b)

Final Percentiles over {ΔPF_1, ..., ΔPF_1000}:
  p5  = 5th percentile  → conservative lower bound of true ΔPF
  p95 = 95th percentile

Gate: p5 > 0.0  (A/B passes if treatment strictly outperforms control with 95% confidence)
```

### 4.3 Pre-Registered Gates (DO NOT alter per run)

| Gate | Use Case | Formula | Rationale |
|------|----------|---------|-----------|
| **Baseline Gate** | New asset screening | `pooled_trades ≥ 300` **AND** `PF_p5 > 1.0` | `PF_p5 > 1.0` = model is profitable with 95% confidence (better than break-even). Truncates left tail of bootstrap distribution. `≥ 300` pooled trades ensures adequate statistical power (Law of Large Numbers). |
| **A/B Test Gate** | Orthogonal feature improvement | `pooled_trades ≥ 300` **AND** `ΔPF_p5 > 0.0` | `ΔPF_p5 > 0.0` = treatment strictly improves over control with 95% confidence. Prevents p-hacking: treatment must exceed the LOWER bound of the distribution. If p5 > 0, the worst-case scenario (95% CI) is already positive. |

---

## 5. Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    %% ===== EXTERNAL APIs =====
    classDef api fill:#ff7f50,stroke:#333,stroke-width:2px,color:white
    classDef ml fill:#4682b4,stroke:#333,stroke-width:2px,color:white
    classDef pipe fill:#32cd32,stroke:#333,stroke-width:2px,color:white
    classDef storage fill:#ddd,stroke:#333,stroke-width:2px
    classDef cli fill:#9932cc,stroke:#333,stroke-width:2px,color:white
    classDef bot fill:#ff1493,stroke:#333,stroke-width:2px,color:white

    subgraph EXTERNAL[External APIs]
        BINANCE[Binance<br/>Futures + Spot]:::api
        FNG[alternative.me<br/>Fear & Greed API]:::api
        TELEGRAM[Telegram Bot API]:::bot
    end

    subgraph INGESTION[Data Ingestion]
        DF[data_fetcher.py<br/>ccxt + REST Binance]:::ml
    end

    subgraph LOCAL_STORAGE[Local Storage]
        CSV[data/raw_csv/<br/>{SYMBOL}/{TF}.csv<br/>funding_rate.csv]:::storage
        MODELS[data/models/<br/>{SYMBOL}/config.json + .pkl]:::storage
        REPORTS[reports/<br/>{SYMBOL}/baseline_*.json<br/>ab_test_*.json]:::storage
        BOT_STATE[data/bot_state.json<br/>runtime state RW]:::storage
    end

    subgraph RESEARCH[Research Pipeline]
        FEATURES[features.py<br/>Technicals + merge_asof<br/>Sentiment / HTF / Funding / TakerBuy]:::ml
        PROFILES[feature_profiles.py<br/>Declarative Registry]:::pipe
        BUILDER[dataset_builder.py]:::pipe
        WF[oos_validation.py<br/>Walk-Forward + Paired Block Bootstrap]:::ml
        RUNNER[walkforward_runner.py<br/>run_baseline / run_ab_test]:::pipe
        OPT[strategy_optimizer.py<br/>Grid Search HP + Params]:::ml
        TRAIN[train.py<br/>XGBoost Final Train Factory]:::ml
    end

    subgraph CLI_TOOLS[CLI & Diagnostics]
        AQ[aq.py CLI<br/>baseline / ab-test / diagnose-data]:::cli
        DIAG[legacy_archive/diagnostics/*.py<br/>Archived EDA / regimes / naive-baseline]:::cli
    end

    subgraph RUNTIME[Production Execution]
        MAIN[main.py<br/>Entry Point + APScheduler 21:00 ART]:::bot
        TASKS[engine/tasks.py<br/>Daily Orchestrator]:::pipe
        EXEC[binance_executor.py<br/>MARKET + SL/TP Algo Orders]:::api
        BOT_HANDLERS[telegram/handlers.py<br/>ConversationHandler Menus]:::bot
        NOTIFIER[telegram/notifier.py<br/>Signals + Results + Errors]:::bot
    end

    %% ===== Data Flows =====
    BINANCE -->|OHLCV + Funding Rate| DF
    FNG -->|Daily Fear & Greed| DF
    DF -->|Persisted CSV| CSV

    CSV -->|load_csv_data| BUILDER
    PROFILES -->|enrichment chain| BUILDER
    FEATURES -->|merge_asof externals| BUILDER
    BUILDER -->|df + features| WF
    WF -->|Bootstrapped PF + ΔPF| RUNNER

    AQ -->|invokes| RUNNER
    AQ -->|diagnose-data| DIAG
    RUNNER -->|JSON report| REPORTS

    REPORTS -->|screening PF_p5 > 1.0| OPT
    OPT -->|winning config.json| MODELS
    CSV -->|retrain| TRAIN
    OPT -->|HP + features| TRAIN
    TRAIN -->|serialized .pkl| MODELS

    %% ===== Runtime Loop =====
    MAIN -->|21:00 cron| TASKS
    MODELS -->|.pkl + config.json| TASKS
    BINANCE -->|latest 100 candles| TASKS
    FNG -->|updated F&G| TASKS
    SETTINGS[settings.yaml<br/>RO defaults]:::storage -->|merge| BOT_STATE
    BOT_STATE -->|paused? / symbols| TASKS

    TASKS -->|Signal detected| NOTIFIER
    NOTIFIER -->|HTML formatted| TELEGRAM
    TASKS -->|Execute trade| EXEC
    EXEC -->|MARKET + SL/TP| BINANCE
    EXEC -->|Execution result| NOTIFIER

    %% User Interaction
    TELEGRAM -->|/start + inline callbacks| BOT_HANDLERS
    BOT_HANDLERS -->|update symbols / leverage / risk| BOT_STATE
    BOT_HANDLERS -->|trigger train / scan / balance / positions| TASKS
    BOT_HANDLERS -->|Panic Button| EXEC
```

---

## 6. Global Data Flow (Complete Use Case)

1. **Pre-Production Research:**
   ```
   data_fetcher.py (--binance-rest)
     → 4h/1h CSV + funding_rate.csv
     → aq.py baseline → PF_p5 screening
     → aq.py ab-test --profile → ΔPF_p5 gate
     → strategy_optimizer.py (winning grid search)
     → train.py (final .pkl model)
   ```

2. **Daily Runtime (21:00 ART, APScheduler cron):**
   ```
   tasks.daily_market_evaluation()
     ├─ Verify bot_active in bot_state.json
     ├─ For each active symbol in symbols[market]:
     │    ├─ fetch_ohlcv_binance(limit=100) → latest candles
     │    ├─ compute_all_technicals + add_sentiment
     │    ├─ model.predict_proba(last_candle) [.pkl]
     │    ├─ IF proba >= threshold [config.json]:
     │    │    ├─ Calculate TP = close + atr × tp_multi
     │    │    ├─ Calculate SL = close − atr × sl_multi
     │    │    ├─ send_trade_signal() → Telegram
     │    │    └─ executor.execute_futures_trade() → Binance
     │    └─ send_execution_result() / send_execution_error() → Telegram
   ```

3. **Automated Retraining (every 14 days):**
   ```
   _check_training_freshness() > TRAINING_COOLDOWN_DAYS (14)
     → fetch_historical_data (refresh CSV)
     → optimize_strategy (new grid search)
     → train_factory (new .pkl + updated config.json)
   ```

---

## 7. Critical Decoupling Rules (Hard Constraints)

| Rule | Rationale |
|------|-----------|
| **`binance_executor.py` DOES NOT import from `telegram/*`** | Dependency inversion. Executor must be reusable independently of Telegram. |
| **`notifier.py` DOES NOT import from `binance/*`** | Strict separation: notification ≠ execution. |
| **Only `engine/tasks.py` imports from BOTH APIs** | Single point of composition. Simplifies unit testing with mocks. |
| **`settings.yaml` is read-only** | Factory defaults. All user overrides go to `bot_state.json` (atomic `os.replace` writing). |
| **All external feature merges use `merge_asof(direction='backward')`** | Never exact index `merge` or forward `join`. Prevents look-ahead leakage. |
| **Daily HTF (1d) must be shifted +1 day before merging** | A 4h candle on 12/08 CANNOT inherit the 1d value for 12/08 (still open). It must inherit the CLOSED 1d value from 11/08. Verified in `test_trend_htf_leakage.py`. |
| **Funding Rate must only exist for bars starting AFTER settlement** | Settlements occur at 00/08/16 UTC. A 4h candle at 04:00 UTC DOES NOT include the 08:00 UTC funding rate. Verified in `test_funding_rate_leakage.py`. |
| **On-chain daily data (Blockchain.com) must be shifted +2 days before merging** | No latency SLA published. Up to 24h aggregation delay assumed conservatively (no empirical verification). A bar on day D can only see the value from day D-2. Verified in `test_onchain_active_addresses_leakage.py`. |
| **Mempool fee-rate daily data (mempool.space) must be shifted +1 day before merging** | Backend indexes fee_rate_percentiles synchronously from Bitcoin Core RPC in the same block-processing cycle (no async pipeline). +1 day is sufficient and correct, same as trend_htf. Verified in `test_mempool_fee_rate_leakage.py`. |
