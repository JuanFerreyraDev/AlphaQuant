# AlphaQuant — Deployment & Operations

> Reference for configuration, environment variables, logging, rollback, and testnet vs production differences.

---

## Environment Variables (`.env`)

Copy `.env.example` and fill in:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_TOKEN` | ✅ | Bot token (from BotFather → `/newbot`) |
| `AUTHORIZED_CHAT_ID` | ✅ | Your numeric chat ID (get via @userinfobot) |
| `BINANCE_API_KEY` | ✅ | Binance API Key (permissions: Futures Trading) |
| `BINANCE_API_SECRET` | ✅ | Corresponding API Secret |
| `USE_TESTNET` | ✅ | `True` for testnet, `False` for real production |

> **Security:** Never commit `.env`. It is listed in `.gitignore`. Binance API keys must have IP whitelist restrictions and minimum permissions (Futures Trading only, no withdrawals).

### Minimum Required Permissions on Binance API Key

- ✅ Enable Futures
- ❌ Enable Withdrawals (DO NOT enable)
- ❌ Enable Spot & Margin Trading (not required)
- IP Restriction: Whitelist IP of the server running the bot

---

## Testnet vs Production

| Aspect | Testnet (`USE_TESTNET=True`) | Production (`USE_TESTNET=False`) |
|--------|------------------------------|----------------------------------|
| API Endpoint | `testnet.binancefuture.com` | `fapi.binance.com` |
| Funds | Testnet USDT (no real value) | Real USDT |
| Latency | Higher (different throttling) | Normal |
| Historical Data | Limited, potential low liquidity | Complete |
| Recommendation | **Always test here first** | Only when bot is stable on testnet |

**Steps before going to production:**
1. Run on testnet for ≥ 2 weeks verifying that orders execute properly
2. Verify in Telegram that `send_execution_result` shows real fills (not `skipped`)
3. Confirm that the Panic Button effectively closes open positions
4. Check `logs/` to ensure there are no recurring silent errors

---

## Starting the Bot

```bash
# Ensure models exist
ls data/models/BTC_USDT/

# Start in foreground (for development)
python main.py

# Start in background with nohup (basic production)
nohup python main.py > logs/main.log 2>&1 &

# With systemd (recommended production)
# See systemd section below
```

### systemd Configuration (Recommended Production)

```ini
# /etc/systemd/system/alphaquant.service
[Unit]
Description=AlphaQuant Trading Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/project/AlphaQuant
ExecStart=/path/to/venv/bin/python main.py
Restart=on-failure
RestartSec=30
EnvironmentFile=/path/to/project/AlphaQuant/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable alphaquant
sudo systemctl start alphaquant
sudo systemctl status alphaquant
```

---

## Logs

Logs are written to `logs/` with automatic rotation (configured in `src/utils/logging_config.py`).

| File | Content |
|------|---------|
| `logs/alphaquant.log` | Main log: scheduler, evaluations, detected signals |
| `logs/errors.log` | Errors only (WARNING+) |

**View real-time logs:**
```bash
tail -f logs/alphaquant.log
tail -f logs/errors.log
```

**Common errors and causes:**

| Log Error | Probable Cause | Action |
|-----------|----------------|--------|
| `BinanceAPIException: -2019` | Insufficient margin for order | Check Futures balance, reduce leverage |
| `BinanceAPIException: -1121` | Invalid symbol | Check if pair exists on Binance Futures |
| `Model file not found` | `.pkl` does not exist for symbol | Run `strategy_optimizer` and `train` for that symbol |
| `compute_all_technicals: ATR NaN` | CSV too short (< indicator warmup) | Re-download with more history |
| `sentiment not loaded` | Fear & Greed API not responding | alternative.me API down; retry in a few minutes |

---

## `data/bot_state.json` Structure

The bot persists its runtime state in `data/bot_state.json`. Complete example:

```json
{
  "bot_active": true,
  "symbols": {
    "futures": ["BTC_USDT", "ETH_USDT"],
    "spot": []
  },
  "default_leverage": 2,
  "risk_per_trade_pct": 1.0,
  "margin_type": "ISOLATED"
}
```

> **Atomic Writes:** The bot uses `os.replace()` to write state. Never edit this file while the bot is running without pausing it first (via Telegram or setting `bot_active = false`).

**Manual editing (only with bot paused):**
```bash
# Pause via Telegram: Bot → Pause
# Then edit:
nano data/bot_state.json
# Resume via Telegram: Bot → Resume
```

---

## Model Rollback

If a production model shows real PF < 0.85 and you wish to revert to a previous version:

```bash
# View available models for BTC_USDT
ls -la data/models/BTC_USDT/

# The .pkl files include parameters in their name:
# {symbol}_{tp}_{sl}_{swing}_{threshold}.pkl
# Example: BTC_USDT_1_5_1_0_10_0-42.pkl

# To perform a rollback:
# 1. Identify previous well-performing .pkl
# 2. Rename current file to .pkl.bad
mv data/models/BTC_USDT/BTC_USDT_1_5_1_0_10_0-42.pkl \
   data/models/BTC_USDT/BTC_USDT_1_5_1_0_10_0-42.pkl.bad

# 3. Copy previous model as active
cp data/models/BTC_USDT/BTC_USDT_previous.pkl \
   data/models/BTC_USDT/BTC_USDT_1_5_1_0_10_0-42.pkl

# 4. Verify config.json points to correct parameters
cat data/models/BTC_USDT/config.json
```

> **`config.json`** contains `optimal_threshold`, `features`, `swing_period`, `atr_tp_multi`, `atr_sl_multi`. If you rollback the `.pkl`, `config.json` must also correspond to that model version.

---

## Scheduler — Cron Details

```python
# main.py — APScheduler configuration
scheduler.add_job(
    daily_market_evaluation,
    trigger=CronTrigger(hour=21, minute=0, timezone="America/Argentina/Cordoba"),
)
```

- **Timezone:** `America/Argentina/Cordoba` (UTC-3, no daylight saving — Argentina does not adjust clocks)
- **Frequency:** Once daily at 21:00 ART
- **On-demand execution:** Telegram → Futures → Scan (without waiting for cron)
- **Automatic retraining:** If `last_trained` in `config.json` is older than 14 days, the scheduler automatically retrains before evaluating signals

---

## Production Checklist

```
□ .env configured with production keys (USE_TESTNET=False)
□ IP whitelist enabled on Binance API Key
□ Models trained and config.json present for each active symbol
□ bot_state.json has bot_active=true and correct symbols listed
□ Panic Button test executed on testnet without errors
□ logs/ directory has write permissions
□ Weekly monitoring of real PF vs backtest PF scheduled
```
