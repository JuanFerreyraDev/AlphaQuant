# AlphaQuant — Deployment & Operación

> Referencia de configuración, variables de entorno, logs, rollback y diferencias testnet vs producción.

---

## Variables de Entorno (`.env`)

Copiar `.env.example` y completar:

```bash
cp .env.example .env
```

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `TELEGRAM_TOKEN` | ✅ | Token del bot (BotFather → `/newbot`) |
| `AUTHORIZED_CHAT_ID` | ✅ | Tu chat ID numérico (obtener con @userinfobot) |
| `BINANCE_API_KEY` | ✅ | API Key de Binance (permisos: Futures Trading) |
| `BINANCE_API_SECRET` | ✅ | API Secret correspondiente |
| `USE_TESTNET` | ✅ | `True` para testnet, `False` para producción real |

> **Seguridad:** Nunca commitear `.env`. Está en `.gitignore`. Las API keys de Binance deben tener IP whitelist y permisos mínimos (solo Futures Trading, sin retiros).

### Permisos mínimos necesarios en la API Key de Binance

- ✅ Enable Futures
- ❌ Enable Withdrawals (NO habilitar)
- ❌ Enable Spot & Margin Trading (no necesario)
- IP Restriction: whitelist de la IP del servidor donde corre el bot

---

## Testnet vs Producción

| Aspecto | Testnet (`USE_TESTNET=True`) | Producción (`USE_TESTNET=False`) |
|---------|------------------------------|-----------------------------------|
| API endpoint | `testnet.binancefuture.com` | `fapi.binance.com` |
| Fondos | Testnet USDT (sin valor real) | USDT real |
| Latencia | Mayor (throttling distinto) | Normal |
| Datos históricos | Limitados, posible baja liquidez | Completos |
| Recomendación | **Siempre probar aquí primero** | Solo cuando el bot sea estable en testnet |

**Pasos antes de ir a producción:**
1. Correr en testnet ≥ 2 semanas verificando que las órdenes se ejecutan correctamente
2. Verificar en Telegram que `send_execution_result` muestra fills reales (no `skipped`)
3. Confirmar que el Panic Button cierra posiciones efectivamente
4. Revisar `logs/` que no haya errores silenciosos recurrentes

---

## Arrancar el Bot

```bash
# Asegurarse que los modelos existen
ls data/models/BTC_USDT/

# Iniciar (foreground, para desarrollo)
python main.py

# Iniciar en background con nohup (producción básica)
nohup python main.py > logs/main.log 2>&1 &

# Con systemd (producción recomendada)
# Ver sección systemd más abajo
```

### Configuración systemd (producción recomendada)

```ini
# /etc/systemd/system/alphaquant.service
[Unit]
Description=AlphaQuant Trading Bot
After=network.target

[Service]
Type=simple
User=tu_usuario
WorkingDirectory=/ruta/al/proyecto/AlphaQuant
ExecStart=/ruta/al/venv/bin/python main.py
Restart=on-failure
RestartSec=30
EnvironmentFile=/ruta/al/proyecto/AlphaQuant/.env

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

Los logs se escriben en `logs/` con rotación automática (configurado en `src/utils/logging_config.py`).

| Archivo | Contenido |
|---------|-----------|
| `logs/alphaquant.log` | Log principal: scheduler, evaluaciones, señales detectadas |
| `logs/errors.log` | Solo errores (WARNING+) |

**Ver logs en tiempo real:**
```bash
tail -f logs/alphaquant.log
tail -f logs/errors.log
```

**Errores comunes y su causa:**

| Error en log | Causa probable | Acción |
|-------------|----------------|--------|
| `BinanceAPIException: -2019` | Margen insuficiente para la orden | Revisar balance Futures, reducir leverage |
| `BinanceAPIException: -1121` | Símbolo inválido | Verificar que el par existe en Binance Futures |
| `Model file not found` | `.pkl` no existe para el símbolo | Correr `strategy_optimizer` y `train` para ese símbolo |
| `compute_all_technicals: ATR NaN` | CSV muy corto (< warmup de indicadores) | Re-descargar con más historial |
| `sentiment not loaded` | Fear & Greed API no responde | API de alternative.me caída; reintentar en minutos |

---

## Estructura de `data/bot_state.json`

El bot persiste su estado en `data/bot_state.json`. Ejemplo completo:

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

> **Escritura atómica:** El bot usa `os.replace()` para escribir el estado. Nunca editar el archivo mientras el bot está corriendo sin pausarlo primero (vía Telegram o `bot_active = false`).

**Editar manualmente (solo con bot pausado):**
```bash
# Pausar desde Telegram: Bot → Pause
# Luego editar:
nano data/bot_state.json
# Reanudar desde Telegram: Bot → Resume
```

---

## Rollback de Modelo

Si un modelo en producción muestra PF_real < 0.85 y querés volver a una versión anterior:

```bash
# Ver modelos disponibles para BTC_USDT
ls -la data/models/BTC_USDT/

# Los archivos .pkl incluyen los parámetros en el nombre:
# {symbol}_{tp}_{sl}_{swing}_{threshold}.pkl
# Ejemplo: BTC_USDT_1_5_1_0_10_0-42.pkl

# Para hacer rollback:
# 1. Identificar el .pkl anterior que funcionaba bien
# 2. Renombrar el actual a .pkl.bak
mv data/models/BTC_USDT/BTC_USDT_1_5_1_0_10_0-42.pkl \
   data/models/BTC_USDT/BTC_USDT_1_5_1_0_10_0-42.pkl.bad

# 3. Copiar el anterior como el activo
cp data/models/BTC_USDT/BTC_USDT_anterior.pkl \
   data/models/BTC_USDT/BTC_USDT_1_5_1_0_10_0-42.pkl

# 4. Verificar que config.json apunta a los parámetros correctos
cat data/models/BTC_USDT/config.json
```

> **El `config.json`** contiene `optimal_threshold`, `features`, `swing_period`, `atr_tp_multi`, `atr_sl_multi`. Si hacés rollback del `.pkl`, el `config.json` también debe corresponder a esa versión del modelo.

---

## Scheduler — Detalles del Cron

```python
# main.py — configuración del APScheduler
scheduler.add_job(
    daily_market_evaluation,
    trigger=CronTrigger(hour=21, minute=0, timezone="America/Argentina/Cordoba"),
)
```

- **Zona horaria:** `America/Argentina/Cordoba` (UTC-3, sin daylight saving — Argentina no cambia el reloj)
- **Frecuencia:** Una vez por día a las 21:00 ART
- **Ejecución on-demand:** Telegram → Futures → Scan (sin esperar el cron)
- **Retraining automático:** Si `last_trained` en `config.json` tiene más de 14 días, el scheduler re-entrena automáticamente antes de evaluar señales

---

## Checklist de Producción

```
□ .env configurado con keys de producción (USE_TESTNET=False)
□ IP whitelist en API Key de Binance
□ Modelos entrenados y config.json presente para cada símbolo activo
□ bot_state.json tiene bot_active=true y los símbolos correctos
□ Test de Panic Button ejecutado en testnet sin errores
□ logs/ con permisos de escritura
□ Monitoreo semanal de PF_real vs PF_backtest programado
```
