# AlphaQuant

**End-to-end** quantitative platform for algorithmic cryptocurrency trading on Binance Futures. Covers the full trading lifecycle: data ingestion, supervised ML research (XGBoost) with Walk-Forward Out-of-Sample validation, automated order execution, and real-time monitoring via Telegram.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env   # edit with Binance API keys and Telegram credentials

# 3. Download historical data (with microstructure fields)
python -m src.brain.data_fetcher BTC_USDT --timeframe 4h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --timeframe 1h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --timeframe 1d
python -m src.brain.data_fetcher BTC_USDT --funding-rate

# 4. Run data health diagnostics
python -m tools.aq diagnose-data BTC_USDT --timeframe 4h

# 5. Validate statistical edge (baseline screening)
python -m tools.aq baseline BTC_USDT --timeframes 4h 1h

# 6. A/B test an orthogonal feature
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# 7. Start bot + scheduler (21:00 ART)
python main.py
```

---

## Documentation

Full technical documentation is available in English and Spanish. Both versions contain identical content.

### English

| Document | Contents |
|----------|----------|
| [docs/ENGLISH/ARCHITECTURE.md](docs/ENGLISH/ARCHITECTURE.md) | Directory tree, system components (Binance + Telegram APIs, ML engine), feature profile registry, statistical metrics and gates, Mermaid architecture diagram |
| [docs/ENGLISH/WORKFLOW.md](docs/ENGLISH/WORKFLOW.md) | Step-by-step operational guide: data ingestion → asset screening → A/B feature testing → production deployment → monitoring. Includes full experiment history (Exp01–Exp06) and on-chain research closure (August 2026) |
| [docs/ENGLISH/CONTRIBUTING.md](docs/ENGLISH/CONTRIBUTING.md) | How to add a new feature profile to the experiment pipeline: fetcher, merge function, enrichment registry, leakage test, A/B test |
| [docs/ENGLISH/DEPLOYMENT.md](docs/ENGLISH/DEPLOYMENT.md) | Environment variables, logging configuration, model rollback, testnet vs production differences |

### Español

| Documento | Contenido |
|-----------|-----------|
| [docs/SPANISH/ARCHITECTURE.md](docs/SPANISH/ARCHITECTURE.md) | Árbol de directorios, componentes del sistema, motor de ML, registro de feature profiles, métricas y gates estadísticos, diagrama Mermaid |
| [docs/SPANISH/WORKFLOW.md](docs/SPANISH/WORKFLOW.md) | Guía operativa paso a paso: ingesta → screening → A/B testing → producción → monitoreo. Incluye historial completo de experimentos (Exp01–Exp06) y cierre de la investigación on-chain (agosto 2026) |
| [docs/SPANISH/CONTRIBUTING.md](docs/SPANISH/CONTRIBUTING.md) | Cómo agregar un nuevo feature profile al pipeline: fetcher, función de merge, registro de enrichment, test de leakage, A/B test |
| [docs/SPANISH/DEPLOYMENT.md](docs/SPANISH/DEPLOYMENT.md) | Variables de entorno, configuración de logging, rollback de modelos, diferencias testnet vs producción |

---

## Experiment History (BTC_USDT)

Six orthogonal feature experiments have been evaluated under the standard protocol (swing=10, tp=1.5×ATR, sl=1.0×ATR, window=6m/step=6m, bootstrap 1000 iterations, seed=42, gate: ΔPF p5 > 0 vs naive long):

| Exp | Feature | Result | Status |
|-----|---------|--------|--------|
| 01 | `trend_htf` — Daily EMA200 distance | 0/8 PASS | Discarded |
| 02 | `funding_rate_current` — Binance Futures 8h settlement rate | 1/8 PASS (noise level) | Discarded |
| 03 | `taker_buy_ratio` — Aggressive buy volume fraction | 0/8 PASS | Discarded |
| 04 | `regression_return` — Continuous return formulation | 0/6 PASS | Discarded |
| 05 | `onchain_active_addresses` — Blockchain.com daily unique addresses | 0/6 PASS | Discarded |
| 06 | `mempool_fee_rate_p50` — mempool.space daily median fee-rate (sat/vB) | 0/6 PASS | Discarded |

A 159-run config sweep (swing × tp_multi × sl_multi grid) run before Exp05–06 confirmed the current default config is not clearly suboptimal. No alternative combination passed the 3-criteria rigor check. **Default unchanged: swing=10, tp=1.5, sl=1.0.**

See [docs/ENGLISH/WORKFLOW.md — Appendix C](docs/ENGLISH/WORKFLOW.md) for full details.

---

## Requirements

- Python ≥ 3.10
- Binance account with USD-M Futures enabled
- Telegram Bot (via BotFather) + `AUTHORIZED_CHAT_ID`
- Environment variables configured in `.env` (see [docs/ENGLISH/DEPLOYMENT.md](docs/ENGLISH/DEPLOYMENT.md))

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | XGBoost, scikit-learn, Numba (JIT-compiled target computation) |
| Technical Indicators | pandas-ta |
| Data Ingestion | ccxt, requests (Binance REST native), pandas |
| On-Chain Data | Blockchain.com API (n-unique-addresses), mempool.space API (fee-rates) |
| Sentiment | alternative.me Fear & Greed index |
| Execution | python-binance (Futures API) |
| Bot / Alerts | python-telegram-bot (v20+, async) |
| Scheduler | APScheduler |
| Testing | pytest, pytest-cov (327 tests) |

---

## Statistical Gate (pre-registered, do not modify)

All experiments use identical pre-registered gates hardcoded in `src/utils/oos_validation.py`:

- **Baseline gate:** `pooled_trades ≥ 300` AND `ΔPF p5 > 0.0` (model vs naive long)
- **A/B test gate:** `pooled_trades ≥ 300` AND `ΔPF p5 > 0.0` (treatment vs naive long)
- Bootstrap: 1000 iterations, 8 contiguous blocks per OOS window, paired sampling
- Walk-forward: 6-month training window, 6-month step, embargo = swing_period bars

`ΔPF` = Profit Factor(model) − Profit Factor(naive_long). The gate isolates classifier alpha from directional market drift. See [docs/ENGLISH/ARCHITECTURE.md §4](docs/ENGLISH/ARCHITECTURE.md) for full derivation.
