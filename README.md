# AlphaQuant

**End-to-end** quantitative platform for algorithmic cryptocurrency trading on Binance Futures. Covers the full trading lifecycle: data ingestion, supervised ML research (XGBoost) with Walk-Forward Out-of-Sample validation, automated order execution, and real-time monitoring via Telegram.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env   # edit with Binance API keys and Telegram credentials

# 3. Download asset historical data
python -m src.brain.data_fetcher BTC_USDT --timeframe 4h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --timeframe 1h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --timeframe 1d

# 4. Validate statistical edge (baseline screening)
python -m tools.aq baseline BTC_USDT --timeframes 4h 1h

# 5. Start bot + scheduler (21:00 ART)
python main.py
```

---

## Documentation

### English Documentation

| Document | Description |
|-----------|-------------|
| [docs/ENGLISH/ARCHITECTURE.md](docs/ENGLISH/ARCHITECTURE.md) | Directory tree, system components, ML pipeline, statistical metrics and gates, Mermaid diagram |
| [docs/ENGLISH/WORKFLOW.md](docs/ENGLISH/WORKFLOW.md) | Step-by-step operational guide: ingestion → screening → A/B testing → production → maintenance |
| [docs/ENGLISH/CONTRIBUTING.md](docs/ENGLISH/CONTRIBUTING.md) | How to add a new feature profile to the experiment pipeline |
| [docs/ENGLISH/DEPLOYMENT.md](docs/ENGLISH/DEPLOYMENT.md) | Environment variables, logging, model rollback, testnet vs production differences |

### Documentación en Español

| Documento | Descripción |
|-----------|-------------|
| [docs/SPANISH/ARCHITECTURE.md](docs/SPANISH/ARCHITECTURE.md) | Documentación de arquitectura en español |
| [docs/SPANISH/WORKFLOW.md](docs/SPANISH/WORKFLOW.md) | Guía de flujo de trabajo en español |
| [docs/SPANISH/CONTRIBUTING.md](docs/SPANISH/CONTRIBUTING.md) | Guía para contribuir en español |
| [docs/SPANISH/DEPLOYMENT.md](docs/SPANISH/DEPLOYMENT.md) | Guía de despliegue en español |

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
| ML | XGBoost, scikit-learn, Numba |
| Data Ingestion | ccxt, requests (Binance REST native), pandas |
| Execution | python-binance (Futures API) |
| Bot | python-telegram-bot (v20+, async) |
| Scheduler | APScheduler |
| Testing | pytest, pytest-cov |