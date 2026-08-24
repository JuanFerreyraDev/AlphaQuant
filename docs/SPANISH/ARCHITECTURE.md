# AlphaQuant — Arquitectura General del Sistema

> Documento técnico de grado producción. Versión 1.0 — Agosto 2026.

---

## 1. Visión General del Sistema

AlphaQuant es una plataforma cuantitativa **end-to-end** diseñada para el trading algorítmico de criptomonedas en Binance (Futures USD-M y Spot). El sistema cubre el ciclo completo desde la ingesta de datos de mercado, la investigación con Machine Learning supervisado (XGBoost) mediante validación Walk-Forward Out-of-Sample (OOS), hasta la ejecución automatizada de órdenes y el sistema de alertas vía Telegram.

### Alcance Funcional

| Capa | Responsabilidad |
|------|-----------------|
| **Research / ML** | Feature engineering, screening de activos, A/B testing de features, optimización de hiperparámetros, validación OOS con bootstrapping de bloques pareados. |
| **Ejecución** | Conexión con Binance API, gestión de margen ISOLATED, sizing por regla del 1%, órdenes MARKET de entrada + STOP_MARKET / TAKE_PROFIT_MARKET condicionales. |
| **Alertas / UX** | Bot de Telegram con interfaz inline-keyboard: pausar/reanudar bot, agregar/quitar símbolos, modificar riesgo/apalancamiento, botón de pánico, ver balance y posiciones abiertas. |
| **Orquestación** | Scheduler APScheduler diario (21:00 ART) que evalúa modelos, envía señales a Telegram y ejecuta trades en Binance. Retraining automático cada 14 días. |

---

## 2. Árbol de Directorios Actualizado

```
AlphaQuant/
├── src/                              # Código fuente de producción
│   ├── api/                          # Integraciones externas (APIs)
│   │   ├── binance/
│   │   │   └── binance_executor.py   # Ejecución de órdenes Futures (ISOLATED + SL/TP algorítmicos)
│   │   └── telegram/
│   │       ├── handlers.py           # Router de ConversationHandler (auth, callbacks, states)
│   │       ├── _actions.py           # Implementaciones de cada botón/acción del menú
│   │       ├── _ui.py                # Textos y teclados inline (menú principal, futures, bot)
│   │       └── notifier.py           # Envío asíncrono de señales y resultados de ejecución
│   ├── brain/                        # Research y motor de ML
│   │   ├── data_fetcher.py           # Descarga OHLCV (ccxt + Binance REST), Funding Rate, Fear&Greed
│   │   ├── features.py               # Indicadores técnicos + merge_asof de features externas (F&G, HTF, Funding, Taker Buy)
│   │   ├── strategy_optimizer.py     # Grid search de parámetros (swing, TP/SL ATR, XGB HP)
│   │   └── train.py                  # Train factory: modelo final XGBoost + serialización .pkl
│   ├── config/                       # Configuración y gestión de paths
│   │   ├── paths.py                  # Centralización de rutas (CSVs, modelos, reportes)
│   │   ├── settings_loader.py        # Merge settings.yaml (RO) + bot_state.json (RW)
│   │   └── experiment_defaults.py    # ExperimentConfig, FORMULATIONS, gates pre-registrados
│   ├── engine/
│   │   └── tasks.py                  # Orquestador diario: evalúa modelos → señal → ejecución → notificación
│   ├── pipeline/                     # Pipeline reproducible de experimentos
│   │   ├── feature_profiles.py       # Registry declarativo de perfiles (control, trend_htf, funding_rate, taker_buy_ratio)
│   │   ├── dataset_builder.py        # load_csv → enrichments → compute_target → dropna
│   │   └── walkforward_runner.py     # run_baseline / run_ab_test + reportes JSON
│   └── utils/                        # Utilidades de dominio
│       ├── helpers.py                # compute_target (Numba JIT), train_predict_* formulations, Profit Factor
│       ├── oos_validation.py         # run_walk_forward, paired block bootstrap, gates estadísticos
│       ├── data_splits.py            # compute_dynamic_split, train/val/test con embargo
│       ├── timeframe_utils.py        # parseo de timeframes (1h, 4h, 1d) a horas
│       └── logging_config.py         # Configuración centralizada de logging
│
├── tools/                            # Scripts de CLI, diagnósticos y experimentos legacy
│   ├── aq.py                         # CLI unificada: baseline / ab-test / diagnose-data / diagnose-naive-baseline / diagnose-regimes-rigorous / diagnose-swing-and-regimes / diagnose-timeframe-swing-sweep
│   ├── visualize_val_signals.py      # Plotting de señales de validación
│   ├── legacy_archive/               # Experimentos fallidos, supersedados y diagnósticos archivados
│   │   ├── diagnostics/              # Scripts de diagnóstico archivados (ahora en aq.py)
│   │   │   ├── diagnose_naive_baseline.py         # Usar: aq diagnose-naive-baseline
│   │   │   ├── diagnose_regimes_rigorous.py       # Usar: aq diagnose-regimes-rigorous
│   │   │   ├── diagnose_swing_and_regimes.py      # Usar: aq diagnose-swing-and-regimes
│   │   │   ├── diagnose_timeframe_data.py         # Usar: aq diagnose-data
│   │   │   ├── diagnose_timeframe_swing_sweep.py  # Usar: aq diagnose-timeframe-swing-sweep
│   │   │   └── README.md             # Documentación del archivo
│   │   ├── exp01_trend_htf_walkforward.py         # 0/8 PASS — EMA200 diaria sin alpha ortogonal
│   │   ├── exp02_funding_rate_walkforward.py      # 1/8 PASS — nivel de ruido, descartado
│   │   ├── exp03_taker_buy_ratio_walkforward.py   # 0/8 PASS — taker ratio no ortogonal
│   │   ├── exp04_regression_return_walkforward.py # 0/6 PASS — formulación de regresión descartada
│   │   ├── compare_binary_vs_multiclass.py
│   │   ├── exp_eth_baseline_oos.py
│   │   └── reconcile_naive_target_comparison.py
│
├── tests/                            # Suite pytest (unit + integración + leakage)
│   ├── unit/
│   │   ├── test_helpers.py                # compute_target (Numba), profit_factor
│   │   ├── test_features.py               # Indicadores: RSI, MACD, BB, OBV, EMA
│   │   ├── test_data_splits.py            # Dynamic splits con embargo
│   │   ├── test_settings_loader.py        # Merge YAML + bot_state
│   │   └── test_logging_config.py
│   ├── integration/
│   │   ├── test_oos_validation.py         # Walk-forward completo con data mock
│   │   ├── test_strategy_optimizer.py     # Grid search + sanity check OOS
│   │   ├── test_train.py                  # Train factory + serialización pkl
│   │   ├── test_data_fetcher.py           # Descarga ccxt mockada
│   │   └── test_tasks.py                  # Orquestador con mocks de Binance/Telegram
│   ├── features/
│   │   ├── test_funding_rate_leakage.py      # Verifica que funding solo existe para barras pre-liquidación
│   │   ├── test_trend_htf_leakage.py         # Verifica que 1d data desplazada +1d antes de merge_asof
│   │   ├── test_taker_buy_ratio_semantics.py # Tests de semántica del ratio de volumen agresor
│   │   ├── test_onchain_active_addresses_leakage.py  # Verifica shift +2d para datos diarios de Blockchain.com
│   │   └── test_mempool_fee_rate_leakage.py  # Verifica shift +1d para datos diarios de mempool.space
│   ├── api/
│   │   ├── test_binance_executor.py       # Sizing + filtros mock
│   │   ├── test_notifier.py               # Formatos HTML correctos
│   │   └── test_telegram_handlers.py      # Auth + state machine
│   └── conftest.py
│
├── data/                             # Persistencia local (no versionada)
│   ├── raw_csv/                      # OHLCV + Funding Rate (por símbolo y timeframe)
│   │   └── {SYMBOL_USDT}/
│   │       ├── 1h.csv | 4h.csv | 1d.csv
│   │       ├── funding_rate.csv
│   │       ├── onchain_active_addresses.csv        # Blockchain.com n-unique-addresses (solo BTC)
│   │       └── onchain_mempool_fee_rate_p50.csv    # mempool.space fee-rates/all avgFee_50 (solo BTC)
│   ├── models/                       # Modelos serializados + config.json por símbolo
│   │   └── {SYMBOL_USDT}/
│   │       ├── config.json           # Features, threshold, HP, last_trained, OOS sanity check
│   │       └── {symbol}_{tp}_{sl}_{swing}_{threshold}.pkl
│   └── plots/
│
├── reports/                          # Reportes JSON de experimentos (baselines, A/B tests)
│   └── {SYMBOL_USDT}/
│       ├── baseline_{timestamp}.json
│       ├── ab_test_{profile}_{timestamp}.json
│       └── latest_baseline.json      # Symlink/copy al reporte más reciente
│
├── benchmarks/                       # Benchmarks de performance (Numba helpers)
├── logs/                             # Logs rotativos
├── main.py                           # Entry point: inicializa bot + scheduler
├── settings.yaml                     # Configuración read-only de fábrica
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 3. Componentes del Sistema

### 3.1 APIs e Integraciones Externas

#### 3.1.1 Binance API (Ingesta + Ejecución)

**Módulos:** [data_fetcher.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/brain/data_fetcher.py), [binance_executor.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/api/binance/binance_executor.py)

| Capa | Librería | Endpoints / Métodos |
|------|----------|---------------------|
| **OHLCV Standard** | `ccxt` (sincrónico) | `fetch_ohlcv` — 6 columnas (OHLCV + timestamp). Paginación de a 1000 velas, retry exponencial. |
| **OHLCV Microestructura** | `requests` (Binance REST nativo) | `/fapi/v1/klines` (Futures) o `/api/v3/klines` (Spot). 12 campos: `quote_volume, n_trades, taker_buy_base_vol, taker_buy_quote_vol`. Flag `--binance-rest`. |
| **Funding Rate** | `ccxt.binanceusdm` | `fetch_funding_rate_history` — settlements cada 8h (00/08/16 UTC). |
| **Ejecución Futures** | `python_binance` (Client) | `futures_create_order` (MARKET entrada) + Algo Orders API (`STOP_MARKET`, `TAKE_PROFIT_MARKET` con `closePosition=TRUE` y `workingType=MARK_PRICE`). |
| **RT Candles (eval diaria)** | `ccxt.async_support.binanceusdm` | `fetch_ohlcv(limit=100)` asincrónico sin credenciales. |

**Reglas de negocio del executor:**

```python
# 1 posición concurrente MAX por símbolo (sin averaging)
_has_open_position(symbol) → SI → cancelar trade nuevo

# Sizing: 1% del balance USDT disponible × leverage
margin     = balance × 0.01
notional   = margin × leverage
quantity   = (notional / precio) → ROUND_DOWN respetando LOT_SIZE step_size

# Filtros del exchange (NO hardcodear)
LOT_SIZE: step_size, min_qty
MIN_NOTIONAL: notional mínimo
PRICE_FILTER: tick_size para SL/TP string
```

#### 3.1.2 Telegram Bot (Alertas + UX Operativa)

**Módulos:** [handlers.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/api/telegram/handlers.py), [_actions.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/api/telegram/_actions.py), [_ui.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/api/telegram/_ui.py), [notifier.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/api/telegram/notifier.py)

**Arquitectura de separación de concerns:**

```
handlers.py (router) → _actions.py (lógica) → _ui.py (presentación)
                           ↓
                     notifier.py (envío de mensajes)
```

| Capa | Descripción |
|------|-------------|
| **Autenticación** | `AUTHORIZED_CHAT_ID` (entero) del `.env`. `_is_authorized()` chequea cada callback y comando. No autorizado → mensaje "Access Denied". |
| **ConversationHandler** | State machine: `NAVIGATING` (menú), `WAITING_ADD_SYMBOL`, `WAITING_REMOVE_SYMBOL`, `WAITING_LEVERAGE`, `WAITING_RISK`. |
| **Menú principal** | 4 secciones inline: Bot, Exchange (placeholder), Futures, Spot (placeholder). |
| **Botón de Pánico** | `action:panic` → confirmación → `close_all_positions()`: MARKET para cerrar posiciones abiertas + `futures_cancel_all_open_orders`. |
| **Notificaciones automáticas** | `send_trade_signal` (nueva señal detectada), `send_execution_result` (trade ejecutado o skipeado), `send_execution_error` (excepción en ejecución). |

---

### 3.2 Motor de Machine Learning

#### 3.2.1 Target Definition (Ternario)

**Función:** [helpers.py → compute_target](file:///home/juan/Desktop/Projects/AlphaQuant/src/utils/helpers.py#L28-L129) (Numba JIT)

```
Clasificación ternaria de 3 clases (y ∈ {-1, 0, +1}):
  +1 = Take Profit alcanzado primero (1.5 × ATR)
   0 = Timeout: ni TP ni SL alcanzados dentro de swing_period barras
  -1 = Stop Loss alcanzado primero (1.0 × ATR)

Tie-break: SL siempre gana si ambos niveles son tocados en la MISMA vela.
           (previene optimismo en barras de alta volatilidad)
```

#### 3.2.2 Formulaciones (XGBoost)

**Definidas en:** [experiment_defaults.py → FORMULATIONS](file:///home/juan/Desktop/Projects/AlphaQuant/src/config/experiment_defaults.py#L19-L22)

| Formulation | Target transformado | Tipo de modelo | Threshold grid |
|-------------|---------------------|----------------|----------------|
| `binary_homerun` | `y_binary = (y == +1)` vs `{0, -1}` | `XGBClassifier` binario | `(0.50, 0.85, 0.01)` |
| `multiclass_3` | `y_multiclass = y + 1` → `{0,1,2}` | `XGBClassifier` multi:softprob | `(0.25, 0.70, 0.01)` (probabilidad clase +1) |
| `regression_return` | `target_ret` continuous realized return | `XGBRegressor` (reg:pseudohubererror) | `(-0.0035, 0.0070, 0.0003)` — DISCARDED (see legacy_archive/exp04_regression_return_walkforward.py) |

**Pipeline de entrenamiento por fold:**

```
Train split (raw)
  ├── X: 14 features base + treatment feature
  ├── y: formulación binaria o multiclass
  │
  ├── HP defaults (production):
  │     n_estimators = 100/200
  │     max_depth    = 2, 3, 4
  │     learning_rate= 0.01, 0.05
  │     scale_pos_weight (solo binary) = ratio imbalance
  │
  └── Early stopping: 10% holdout reciente, early_stopping_rounds=10

Val split
  └── Grid search de threshold (maximiza suma neta de retornos
      con floor mínimo de trades en val)

Test split (OOS)
  └── Evaluación ciega con threshold óptimo de val
      (NUNCA optimizado en test)
```

#### 3.2.3 Feature Profile Registry

**Módulo:** [feature_profiles.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/pipeline/feature_profiles.py)

Registry declarativo de perfiles de enriquecimiento. Todos usan `merge_asof(direction='backward')` para features externas.

| Profile | Enrichments | Treatment Col | Requisitos CSV extra |
|---------|-------------|---------------|----------------------|
| **control** | `technicals` + `sentiment` | None (baseline) | — |
| **trend_htf** | `technicals` + `sentiment` + `trend_htf` | `trend_htf` | `1d.csv` (EMA200 diaria, shift +1d) |
| **funding_rate** | `technicals` + `sentiment` + `funding_rate` | `funding_rate_current` | `funding_rate.csv` (solo settlements pasados) |
| **taker_buy_ratio** | `technicals` + `sentiment` + `taker_buy_ratio` | `taker_buy_ratio` | CSVs descargados con `--binance-rest` |
| **onchain_activity** | `technicals` + `sentiment` + `onchain_active_addresses` | `onchain_active_addresses` | `onchain_active_addresses.csv` (Blockchain.com, shift +2d) |
| **onchain_fee_pressure** | `technicals` + `sentiment` + `mempool_fee_rate_p50` | `mempool_fee_rate_p50` | `onchain_mempool_fee_rate_p50.csv` (mempool.space, shift +1d) |

**14 Features de Control (baseline):**
```
Momentum:    rsi_14, macd, macd_hist, stoch_k
Trend:       dist_ema_50, adx_14
Volatility:  atr_14, bb_width, bb_pos
Volume:      obv, rel_volume
Sentiment:   fng_value, fng_sma_14, fng_vol_14
```

---

### 3.3 Arquitectura del Pipeline y CLI

#### 3.3.1 Dataset Builder

**Módulo:** [dataset_builder.py → build_dataset](file:///home/juan/Desktop/Projects/AlphaQuant/src/pipeline/dataset_builder.py#L19-L82)

```python
Flujo determinístico:
  1. load_csv_data(symbol, timeframe)
  2. for enrichment in profile.enrichments:
       df = ENRICHMENT_REGISTRY[enrichment](df, symbol)
  3. compute_target(swing=10, TP=1.5×ATR, SL=1.0×ATR)
  4. drop(columns=COLS_TO_DROP)    # 9 columnas — ver tabla abajo
  5. dropna()                       (elimina barras sin target/features)
  6. Inferencia de feature_cols:
       - control_features = TODAS las columnas numéricas
                              − {close, target, treatment_col}
       - REQUIRED_BASE_FEATURES chequeo de salud (6 cols)
       - SENTIMENT_COLS chequeo de salud (3 cols)
```

**`COLS_TO_DROP` completo (9 columnas):**

| Columna | Razón |
|---------|-------|
| `open`, `high`, `low` | OHLCV crudo — el modelo trabaja con indicadores derivados |
| `volume` | Reemplazado por `rel_volume` (volumen relativo normalizado) |
| `ema_50` | Reemplazado por `dist_ema_50` (distancia porcentual) |
| `vol_sma_20` | Reemplazado por `rel_volume` |
| `max_high_future`, `min_low_future` | Columnas auxiliares de `compute_target` — leakage total si quedan |
| `quote_volume`, `n_trades`, `taker_buy_base_vol`, `taker_buy_quote_vol` | Campos de microestructura descargados con `--binance-rest`. Se dropean para que el modelo base no los use directamente. Solo el perfil `taker_buy_ratio` los consume, y lo hace a través del feature derivado `taker_buy_ratio` (no crudo). |

> **Nota:** Las 4 columnas de microestructura existen en el DataFrame cargado solo cuando el CSV fue descargado con `--binance-rest`. `COLS_TO_DROP` usa `[c for c in COLS_TO_DROP if c in df.columns]` para no fallar si no existen.

**`REQUIRED_BASE_FEATURES` — health check de 6 columnas:**

Distinto de las 14 features de control. Son las 6 columnas mínimas que deben existir tras los enrichments para que el pipeline no esté silenciosamente roto:

```python
REQUIRED_BASE_FEATURES = frozenset({
    "rsi_14", "atr_14", "bb_width", "bb_pos", "obv", "rel_volume"
})
```

Si alguna falta, `build_dataset` lanza `RuntimeError` con lista de columnas disponibles.

#### 3.3.2 WalkForwardRunner

**Módulos:** [walkforward_runner.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/pipeline/walkforward_runner.py), [oos_validation.py → run_walk_forward](file:///home/juan/Desktop/Projects/AlphaQuant/src/utils/oos_validation.py)

```
Parámetros por defecto (ExperimentConfig):
  swing_period   = 10 barras
  tp_multi       = 1.5 × ATR
  sl_multi       = 1.0 × ATR
  window_months  = 6    (train de 6 meses corrido)
  step_months    = 6    (avance de 6 meses entre folds)
  n_bootstrap    = 1000 iteraciones
  n_blocks       = 8    bloques contiguos por ventana OOS
  random_state   = 42
  fee_rate       = 0.0 (research mode; ejecución usa 0.001 real)
  slippage       = 0.0 (research mode; ejecución usa 0.0005 real)
```

#### 3.3.3 CLI Unificada `aq.py`

**Entry point:** [tools/aq.py](file:///home/juan/Desktop/Projects/AlphaQuant/tools/aq.py)

```bash
# === Screening de activo nuevo ===
python -m tools.aq baseline ETH_USDT --timeframes 4h 1h --fetch

# === A/B Test de feature ortogonal ===
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# === Diagnóstico de salud de data ===
python -m tools.aq diagnose-data SOL_USDT --timeframe 4h --swing 10

# === Overrides de config ===
--swing-period 10 --tp-multi 1.5 --sl-multi 1.0
--window-months 6 --step-months 6
--fee-rate 0.001 --slippage 0.0005
--n-bootstrap 1000 --n-blocks 8 --random-state 42
```

---

## 4. Métricas y Gates Estadísticos

### 4.1 Profit Factor (PF)

Definición (por trade, no por barra):

```
          Σ (retornos positivos por trade)
PF = ─────────────────────────────────────────
     |Σ (retornos negativos por trade)|

Donde cada trade retorno = f(y_true[i]):
  y=+1  →  +(atr[i] × tp_multi) / close[i]  − cost_per_trade
  y=−1  →  −(atr[i] × sl_multi) / close[i]  − cost_per_trade
  y=0   →  (close[exit] − close[i]) / close[i] − cost_per_trade

cost_per_trade = 2×fee_rate + 2×slippage  (entrada + salida, ambos lados)
```

### 4.2 Bootstrap Estadístico

El sistema estima la distribución del **ΔPF = PF(tratamiento) − PF(naive)** con bloques **pareados** vía `_bootstrap_paired_blocks`:

#### 4.2.1 `_bootstrap_paired_blocks` — A/B Test

**Implementación:** [oos_validation.py → _bootstrap_paired_blocks](file:///home/juan/Desktop/Projects/AlphaQuant/src/utils/oos_validation.py#L181-L239)

Usado en `run_ab_test` (vía `run_walk_forward`). Estima la distribución del **ΔPF = PF(tratamiento) − PF(naive)** con bloques **pareados**: en cada iteración se muestrean los **mismos índices de bloque** para modelo y naive, eliminando varianza debida a condiciones de mercado puntuales.

**¿Por qué bloques?** Los retornos financieros exhiben **autocorrelación serial** (clustering de volatilidad). El bootstrap i.i.d. subestima la varianza real. Bloques contiguos preservan la estructura de dependencia temporal.

```
Pool global de bloques pareados (model + naive, misma ventana, mismo bloque):
  1. Para cada ventana OOS w: dividir trades modelo y naive en B=8 bloques contiguos
  2. Agregar todos los pares (block_model_i, block_naive_i) al pool global
  3. Para cada bootstrap iteración b ∈ {1..1000}:
       a. Muestrear con repetición n_total_blocks índices (MISMOS para model y naive)
       b. Concatenar blocks_model[idx] → rets_mdl_b
       c. Concatenar blocks_naive[idx] → rets_nav_b
       d. ΔPF_b = PF(rets_mdl_b) − PF(rets_nav_b)

Percentiles finales sobre {ΔPF_1, ..., ΔPF_1000}:
  p5  = percentil 5  → cota inferior conservadora del "verdadero" ΔPF
  p95 = percentil 95

Gate: p5 > 0.0  (A/B pasa si con 95% confianza el tratamiento mejora al control)
```

### 4.3 Gates Pre-Registrados (NO alterar por run)

| Gate | Caso de Uso | Fórmula | Justificación |
|------|-------------|---------|---------------|
| **Gate de Baseline** | Screening de activos nuevos | `pooled_trades ≥ 300` **Y** `PF_p5 > 1.0` | `PF_p5 > 1.0` = con 95% de confianza el modelo es rentable (mejor que break-even). Truncamos el lado izquierdo de la distribución bootstrap. `≥ 300` trades pooled asegura potencia estadística suficiente (Ley de Grandes Números en colas). |
| **Gate de A/B Test** | Mejora ortogonal de features | `pooled_trades ≥ 300` **Y** `ΔPF_p5 > 0.0` | `ΔPF_p5 > 0.0` = con 95% de confianza el tratamiento mejora estrictamente el control. Previene p-hacking: el tratamiento tiene que superar la cota INFERIOR de la distribución, no la media. Si el p5 > 0, el caso base (95% worst-case) ya es positivo. |

---

## 5. Diagrama de Arquitectura (Mermaid)

```mermaid
flowchart TD
    %% ===== APIs EXTERNAS =====
    classDef api fill:#ff7f50,stroke:#333,stroke-width:2px,color:white
    classDef ml fill:#4682b4,stroke:#333,stroke-width:2px,color:white
    classDef pipe fill:#32cd32,stroke:#333,stroke-width:2px,color:white
    classDef storage fill:#ddd,stroke:#333,stroke-width:2px
    classDef cli fill:#9932cc,stroke:#333,stroke-width:2px,color:white
    classDef bot fill:#ff1493,stroke:#333,stroke-width:2px,color:white

    subgraph EXTERNAL[APIs Externas]
        BINANCE[Binance<br/>Futures + Spot]:::api
        FNG[alternative.me<br/>Fear & Greed API]:::api
        TELEGRAM[Telegram Bot API]:::bot
    end

    subgraph INGESTA[Ingesta de Datos]
        DF[data_fetcher.py<br/>ccxt + REST Binance]:::ml
    end

    subgraph LOCAL_STORAGE[Almacenamiento Local]
        CSV[data/raw_csv/<br/>{SYMBOL}/{TF}.csv<br/>funding_rate.csv]:::storage
        MODELS[data/models/<br/>{SYMBOL}/config.json + .pkl]:::storage
        REPORTS[reports/<br/>{SYMBOL}/baseline_*.json<br/>ab_test_*.json]:::storage
        BOT_STATE[data/bot_state.json<br/>runtime state RW]:::storage
    end

    subgraph RESEARCH[Research Pipeline]
        FEATURES[features.py<br/>Technicals + merge_asof<br/>Sentiment / HTF / Funding / TakerBuy]:::ml
        PROFILES[feature_profiles.py<br/>Registry declarativo]:::pipe
        BUILDER[dataset_builder.py]:::pipe
        WF[oos_validation.py<br/>Walk-Forward + Paired Block Bootstrap]:::ml
        RUNNER[walkforward_runner.py<br/>run_baseline / run_ab_test]:::pipe
        OPT[strategy_optimizer.py<br/>Grid Search HP + Params]:::ml
        TRAIN[train.py<br/>Train Factory XGBoost Final]:::ml
    end

    subgraph CLI_TOOLS[CLI y Diagnósticos]
        AQ[aq.py CLI<br/>baseline / ab-test / diagnose-data]:::cli
        DIAG[diagnostics/*.py<br/>EDA riguroso / regimes / naive-baseline]:::cli
    end

    subgraph RUNTIME[Ejecución en Producción]
        MAIN[main.py<br/>Entry Point + APScheduler 21:00 ART]:::bot
        TASKS[engine/tasks.py<br/>Orquestador Diario]:::pipe
        EXEC[binance_executor.py<br/>MARKET + SL/TP Algo Orders]:::api
        BOT_HANDLERS[telegram/handlers.py<br/>ConversationHandler Menús]:::bot
        NOTIFIER[telegram/notifier.py<br/>Señales + Resultados + Errores]:::bot
    end

    %% ===== Flujos de Datos =====
    BINANCE -->|OHLCV + Funding Rate| DF
    FNG -->|Fear & Greed diario| DF
    DF -->|CSV persistido| CSV

    CSV -->|load_csv_data| BUILDER
    PROFILES -->|enrichment chain| BUILDER
    FEATURES -->|merge_asof externals| BUILDER
    BUILDER -->|df + features| WF
    WF -->|PF + ΔPF bootstrapped| RUNNER

    AQ -->|invoca| RUNNER
    AQ -->|diagnose-data| DIAG
    RUNNER -->|JSON report| REPORTS

    REPORTS -->|screening PF_p5 > 1.0| OPT
    OPT -->|config.json ganador| MODELS
    CSV -->|retrain| TRAIN
    OPT -->|HP + features| TRAIN
    TRAIN -->|.pkl serializado| MODELS

    %% ===== Runtime Loop =====
    MAIN -->|21:00 cron| TASKS
    MODELS -->|.pkl + config.json| TASKS
    BINANCE -->|últimas 100 velas| TASKS
    FNG -->|F&G actualizado| TASKS
    SETTINGS[settings.yaml<br/>RO defaults]:::storage -->|merge| BOT_STATE
    BOT_STATE -->|paused? / symbols| TASKS

    TASKS -->|Signal detectada| NOTIFIER
    NOTIFIER -->|HTML formatted| TELEGRAM
    TASKS -->|Ejecutar trade| EXEC
    EXEC -->|MARKET + SL/TP| BINANCE
    EXEC -->|Execution result| NOTIFIER

    %% Usuario interactúa
    TELEGRAM -->|/start + callbacks inline| BOT_HANDLERS
    BOT_HANDLERS -->|modificar symbols / leverage / risk| BOT_STATE
    BOT_HANDLERS -->|trigger train / scan / balance / positions| TASKS
    BOT_HANDLERS -->|Panic Button| EXEC
```

---

## 6. Flujo de Datos Global (Caso de Uso Completo)

1. **Investigación pre-producción:**
   ```
   data_fetcher.py (--binance-rest)
     → CSV 4h/1h + funding_rate.csv
     → aq.py baseline → PF_p5 screening
     → aq.py ab-test --profile → ΔPF_p5 gate
     → strategy_optimizer.py (grid search ganador)
     → train.py (modelo final .pkl)
   ```

2. **Runtime diario (21:00 ART, APScheduler cron):**
   ```
   tasks.daily_market_evaluation()
     ├─ Verificar bot_active en bot_state.json
     ├─ Para cada símbolo activo en symbols[market]:
     │    ├─ fetch_ohlcv_binance(limit=100) → últimas velas
     │    ├─ compute_all_technicals + add_sentiment
     │    ├─ model.predict_proba(last_candle) [.pkl]
     │    ├─ IF proba >= threshold [config.json]:
     │    │    ├─ Calcular TP = close + atr × tp_multi
     │    │    ├─ Calcular SL = close − atr × sl_multi
     │    │    ├─ send_trade_signal() → Telegram
     │    │    └─ executor.execute_futures_trade() → Binance
     │    └─ send_execution_result() / send_execution_error() → Telegram
   ```

3. **Retraining automático (cada 14 días):**
   ```
   _check_training_freshness() > TRAINING_COOLDOWN_DAYS (14)
     → fetch_historical_data (refresh CSV)
     → optimize_strategy (nuevo grid search)
     → train_factory (nuevo .pkl + config.json actualizado)
   ```

---

## 7. Reglas de Desacoplamiento Crítico (Hard Constraints)

| Regla | Motivo |
|-------|--------|
| **`binance_executor.py` NO importa de `telegram/*`** | Inversión de dependencias. El executor es reusable sin Telegram. |
| **`notifier.py` NO importa de `binance/*`** | Separación estricta: notificación ≠ ejecución. |
| **Sólo `engine/tasks.py` importa de AMBAS APIs** | Single point of composition. Facilita testing unitario con mocks. |
| **`settings.yaml` es read-only** | Factory defaults. Todo override de usuario va a `bot_state.json` (escritura atómica `os.replace`). |
| **Todo merge de features externas usa `merge_asof(direction='backward')`** | Nunca `merge` exacto por índice ni `join` forward. Previene leakage look-ahead. |
| **HTF diario (1d) debe tener shift +1 día antes del merge** | Una vela 4h del 12/08 NO puede heredar el 1d del 12/08 (todavía abierto). Debe heredar el 1d CERRADO del 11/08. Verificado en `test_trend_htf_leakage.py`. |
| **Funding Rate sólo debe existir para barras con settlement ANTES de su inicio** | Liquidación 00/08/16 UTC. Una vela 4h de las 04:00 UTC NO incluye el funding de 08:00 UTC. Verificado en `test_funding_rate_leakage.py`. |
| **Datos diarios on-chain (Blockchain.com) deben tener shift +2 días antes del merge** | Sin SLA de latencia publicado. Se asume hasta 24h de delay de agregación de forma conservadora (sin verificación empírica). Una barra en el día D solo puede ver el valor del día D-2. Verificado en `test_onchain_active_addresses_leakage.py`. |
| **Datos diarios de mempool fee-rate (mempool.space) deben tener shift +1 día antes del merge** | El backend indexa fee_rate_percentiles de forma sincrónica desde Bitcoin Core RPC en el mismo ciclo de procesamiento de bloque (sin pipeline asíncrono). +1 día es suficiente y correcto, igual que trend_htf. Verificado en `test_mempool_fee_rate_leakage.py`. |
