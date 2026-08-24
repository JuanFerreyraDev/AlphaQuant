# AlphaQuant — Flujo de Trabajo Operativo y Metodología

> Guía paso a paso con justificación técnica en cada fase. Versión 1.0 — Agosto 2026.

---

## Prólogo: Principios Fundamentales No Negociables

Antes de comenzar cualquier operación en AlphaQuant, internalizar:

1. **Un feature ortogonal por experimento.** Nunca stackear múltiples features nuevas en el mismo A/B test. No se puede atribuir causalidad si hay más de una variable independiente cambiando entre control y tratamiento.
2. **Gates estadísticos pre-registrados.** Los umbrales `pooled_trades ≥ 300` y `ΔPF_p5 > 0.0` (model vs naive_long, baseline y A/B) son hardcodeados en el código y NO se modifican post-hoc para acomodar resultados marginales. Cualquier relajación es p-hacking por definición. El baseline NO usa PF absoluto > 1.0 como gate — eso reintroduce sesgo direccional del activo (drift alcista ≠ alpha).
3. **Prevención de leakage = vida.** Todo dato de timeframe superior (HTF), futuro, o que requiera settlement temporal DEBE ser verificado con tests de integración en `tests/features/`. Si no hay test de leakage, el feature no entra al pipeline.
4. **Control fijo + tratamiento variable.** El set de 14 features de control es inmutable. Un tratamiento NUNCA remueve features de control; sólo puede agregar 1 columna ortogonal nueva.

---

## Fase 1 — Ingesta de Datos y Diagnósticos EDA

### Objetivo

Obtener datos crudos de la máxima calidad posible y diagnosticar su validez antes de alimentar ningún modelo. Garbage in → garbage out, en cripto amplificado por la asimetría de información y microstructure.

### Pasos Operativos

#### 1.1 Descarga de OHLCV con microestructura

Siempre preferir `--binance-rest` sobre el endpoint ccxt standard. El endpoint nativo expone campos de microstructure (`taker_buy_base_vol`, `n_trades`, `quote_volume`) necesarios para features de agresividad de flujo.

```bash
# Futures USD-M, timeframe 4h + 1h, con campos de microstructure
python -m src.brain.data_fetcher BTC_USDT --timeframe 4h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --timeframe 1h --binance-rest

# Diario obligatorio para features HTF (trend_htf = distancia a EMA200 1d)
python -m src.brain.data_fetcher BTC_USDT --timeframe 1d

# Funding rate history (solo tiene sentido en perpetual futures)
python -m src.brain.data_fetcher BTC_USDT --funding-rate
```

**Resultado esperado:**

```
data/raw_csv/BTC_USDT/
├── 1h.csv          # open,high,low,close,volume,quote_volume,n_trades,taker_buy_base_vol,taker_buy_quote_vol
├── 4h.csv
├── 1d.csv
└── funding_rate.csv  # timestamp, funding_rate (settlements cada 8h UTC)
```

> **⚠️ Sobre `--fetch` en el CLI unificado:** El flag `--fetch` de `aq baseline` / `aq ab-test` invoca `data_fetcher` **sin** `--binance-rest`. Para el perfil `taker_buy_ratio` (que requiere columnas de microestructura), siempre descargar manualmente con `--binance-rest` **antes** de correr el A/B test. Para los perfiles `control`, `trend_htf` y `funding_rate`, `--fetch` es suficiente.

#### 1.2 Diagnóstico de salud Level-1 (aq diagnose-data)

Antes de entrenar NADA, correr el chequeo de sanidad:

```bash
python -m tools.aq diagnose-data BTC_USDT --timeframe 4h --swing 10
```

Este script reporta:

| Panel                                        | Qué detecta                                                                                                                                                                                                                               |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------------- |
| **(a) Feature health: % NaN / % ==0**        | Features rotos (all-NaN) o degenerados (all-zero). RSI/Stoch/MACD/ADX/ATR deben ser <1% NaN tras warmup. Sentiment <2% NaN.                                                                                                               |
| **(b) Sentiment merge sanity**               | `df.index` debe ser `datetime64[ns]` MONOTÔNICO creciente, SIN duplicados. `fng_value` debe mostrar patrón "staircase": cada valor diario se repite para TODAS las velas del día. Si `days_with_multiple > 0`, el merge de F&G está roto. |
| **(c) Target class balance**                 | Medido en BTC/4h swing=10 target ternario (TP=1.5xATR, SL=1.0xATR): **33.52% TP / 13.51% timeout / 52.97% SL**. Los rangos NO son universales — varían con `swing_period`, `tp_multi` y `sl_multi`: timeout crece si TP/SL son demasiado anchos; SL crece si swing es demasiado corto o SL demasiado tight. Si `target SL > 55%` persistente, revisar el riesgo por barra del setup. |
| **(d) Val vs test regime**                   | Comparación de retorno acumulado y volatilidad train/val/test. Si `test_std > 2 × train_std`, el test es structuralmente diferente: el modelo generalizará mal. Rechazar el activo o extender el periodo train.                           |
| **(e) Point-biserial corr(feature, target)** | Top-5 correlaciones en train. Si TODAS las correlaciones son <                                                                                                                                                                            | 0.03 | , no hay señal lineal detectable → abandona el activo, no pierdas tiempo con XGB. |

#### 1.3 Diagnósticos profundos (CLI unificado)

Si el chequeo Level-1 pasa, correr los sub-comandos de EDA riguroso con `aq`:

```bash
# 1. Separar "beta del mercado" vs "alpha del clasificador"
python -m tools.aq diagnose-naive-baseline BTC_USDT
# Respuesta: Si naive_long PF > model PF en val, tu modelo NO tiene timing
#           edge; todo es drift direccional del activo.

# 2. Comparación de regímenes temporal (bootstrap riguroso)
python -m tools.aq diagnose-regimes-rigorous BTC_USDT
# Respuesta: ¿El split train-val-test preserva estadísticas de régimen?
#           Bootstrap p5/p95 CI en delta model-vs-naive confirma diferencia.

# 3. Barrido de swing × retorno con validación cross-régimen
python -m tools.aq diagnose-swing-and-regimes BTC_USDT
# Respuesta: Sensibilidad del PF a swing_period (Part A). Validación cross-régimen
#           (Part B). Sirve para confirmar que swing=10 no es overfitting accidental.

# 4. Barrido de timeframe × swing
python -m tools.aq diagnose-timeframe-swing-sweep BTC_USDT --timeframe 1h
# Respuesta: Confirma que el modelo funciona en diferentes timeframes y swings.
#           Sanity check de ATR-como-% del precio para validar escaling TP/SL.
```

### 🔴 ¿Por qué hacemos esto?

**Razón matemática:** El teorema de No Free Lunch en ML garantiza que no hay modelo superior sin datos correctos. En trading, el 80% de los "modelos que fallan" en realidad fallan por **contaminación de datos** (NaN propagation, features all-zero, sentiment mal mergeado, targets con leakage), no por el modelo en sí.

**Razón cuantitativa:** Un `fng_sma_14` mal mergeado (NaN propagation en timeframes subdiarios) reduce efectivamente el tamaño de train de N barras a N - 14 en cada fold, sesgando toda la distribución de validación. El `merge_asof(direction='backward')` es obligatorio porque un 4h candle de las 22:00 UTC NO tiene disponible el F&G de las 00:00 UTC del día siguiente.

**Razón de ingeniería:** Los diagnósticos de régimen detectan el "falso positivo más caro en quant trading": un baseline que parece funcionar en val pero fue entrenado en una tendencia alcista sostenida, y en test (bear market o sideways) el PF colapsa a 0.6. El panel (d) cuantifica esta desviación ANTES de gastar 30 minutos en walk-forward.

---

## Fase 2 — Screening de Activos Nuevos (Baseline)

### Objetivo

Determinar si un activo nuevo (por ejemplo: `SOL_USDT`, `ETH_USDT`) tiene suficiente ineficiencia de mercado como para que el set de 14 features de control genere un Profit Factor **estadísticamente superior al naive_long (all-in long)** en OOS. El gate es `ΔPF p5 > 0.0` (modelo vs naive), NO PF absoluto > 1.0 — ese criterio confundía drift direccional del activo con alpha del clasificador. Si NO, abandonar el activo antes de invertir tiempo en features ortogonales.

### Pasos Operativos

#### 2.1 Ejecutar baseline screening

```bash
# Con fetch automático de data si no existe CSV local
python -m tools.aq baseline ETH_USDT --timeframes 4h 1h --fetch
```

**Pipeline ejecutado internamente:**

```
run_baseline(ETH_USDT, timeframes=[4h, 1h])
  │
  ├─ Por cada timeframe:
  │    ├─ build_dataset(profile="control")
  │    │    ├─ load_csv_data → compute_all_technicals → add_sentiment
  │    │    ├─ compute_target(swing=10, TP=1.5×ATR, SL=1.0×ATR)
  │    │    └─ drop(COLS_TO_DROP) + dropna()
  │    │
  │    └─ Por cada formulation ∈ {binary_homerun, multiclass_3}:
  │         └─ run_walk_forward()
  │              ├─ Dynamic splits train/val/test (con embargo de swing barras)
  │              ├─ Por ventana OOS (6m train → step 6m):
  │              │    ├─ Entrenar XGB en train
  │              │    ├─ Grid-search threshold en val (maximiza net return)
  │              │    └─ Evaluar modelo + threshold en test (OOS)
  │              │         → PF_modelo, PF_naive_long (all-in), ΔPF = modelo − naive
  │              │
  │              ├─ Pooled trade count: Σ trades test en todas las ventanas
  │              └─ paired block bootstrap ΔPF (1000 iter, 8 bloques/ventana) → p5, p95
  │                   passes_gate = (trades ≥ 300) AND (ΔPF p5 > 0.0)
  │
  └─ Reporte JSON en reports/ETH_USDT/baseline_{timestamp}.json
```

#### 2.2 Interpretar la tabla SUMMARY

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

> **⚠️ Semántica de las columnas:** `OOS PF` es el Profit Factor puntual **absoluto** del modelo (referencia visual, NO se usa para el gate). `ΔPF p5` es el percentil 5 de la distribución bootstrap de `PF_modelo − PF_naive_long`. Este es el gate real: si ΔPF p5 > 0.0 (con ≥300 trades), con 95% de confianza el modelo supera a un all-in long pasivo. `#w_Δ>0` = cantidad de ventanas OOS individuales donde el modelo beat al naive.

**Matriz de decisión:**

| Caso                                            | Acción                                                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0/4` PASS en todas las combinaciones           | **Rechazar activo.** No hay señal detectable con features de control. No invertir tiempo en experimentos A/B.                                       |
| `1-2/4` PASS (solo 1 timeframe o 1 formulation) | **Condicional.** Si las 2 PASS son el mismo timeframe, ese TF tiene estructura; experimentar solo en ese TF.                                        |
| `2+/4` PASS (múltiples combinaciones PASAN)     | **Aprobar activo.** Pasar a Fase 3 (A/B testing de features ortogonales). Recomendado: seleccionar formulación ganadora para research subsiguiente. |

#### 2.3 Lecciones aprendidas (BTC vs ETH)

Historia del repo (ver `tools/legacy_archive/exp_eth_baseline_oos.py`):

```
BTC_USDT es relativamente más Eficiente que ETH_USDT en 4h cuando
se compara DELTA vs naive_long (el criterio correcto):
  • BTC 4h: ΔPF p5 típicamente ~0.00 a +0.03 (marginal, pocas PASS)
  • ETH 4h: ΔPF p5 típicamente ~+0.02 a +0.06 (más consistentes)

Conclusión: Los altos market-cap deep-liquidity tienen menos ineficiencias
relativas al buy-and-hold. Es más fácil extraer alpha en ETH, SOL, MATIC
que en BTC cuando se compara el mismo feature set. El gate ΔPF_p5 > 0.0
filtra automáticamente esta diferencia de eficiencia SIN contaminar con
sesgo direccional del activo.
```

> **⚠️ Corrección histórica:** Antes de la corrección del gate (agosto 2026), el baseline usaba PF absoluto p5 > 1.0. Esto artificialmente inflaba los PASS en activos con drift alcista sostenido: un modelo PF=1.02 puede aparentar PASS por el sesgo long del mercado, pero en realidad estar empeorando al naive_long (ΔPF < 0). Siempre mirar ΔPF, no PF absoluto.

### 🔴 ¿Por qué hacemos esto?

**Razón matemática:** El gate `ΔPF p5 > 0.0` (modelo − naive_long) es el único criterio que aísla alpha del clasificador del drift direccional del activo. En un activo con tendencia alcista sostenida, incluso un random forest sin entrenamiento puede tener PF absoluto > 1.0 (porque va long por default). El ΔPF remueve ese sesgo: preguntamos "¿con 95% de confianza el modelo MEJORA al all-in long pasivo?". Si `ΔPF p5 = +0.0312`, significa que en el 95% de los escenarios bootstrap el modelo supera al naive por al menos +0.0312.

**Razón cuantitativa:** `pooled_trades ≥ 300` NO es arbitrario. El Profit Factor es un ratio. Su varianza asintótica es inversamente proporcional al número de trades. Con menos de 300 trades pooled, el intervalo de confianza bootstrap es TAN ancho que incluso un ΔPF verdadero de +0.05 puede tener `p5 < 0.0` (falso negativo). Con 300+, el error estándar del PF se estabiliza y el gate tiene Type-I error controlado.

**Razón de ingeniería:** El screening de baseline evita el "sunk cost fallacy". Es tentador seguir agregando features a un activo "porque ya inviertiste 2 horas". Un baseline `0/4` PASS (ΔPF p5 > 0.0) es una señal inequívoca: el activo es demasiado eficiente para este framework, o bien el naive_long ya captura toda la ineficiencia disponible. Cualquier A/B test posterior es p-hacking con certeza.

---

## Fase 3 — Experimentación de Features (A/B Test)

### Objetivo

Evaluar si UN SOLO feature ortogonal nuevo mejora estadísticamente el Profit Factor del baseline, bajo condiciones idénticas (mismo train/val/test, mismo HP, mismo threshold grid).

### Regla de Oro Inviolable

> Cada experimento modifica EXACTAMENTE una variable: agrega una sola columna al set de features de control.
>
> ❌ Prohibido: cambiar swing_period Y agregar trend_htf en el mismo experimento.
> ❌ Prohibido: agregar funding_rate Y taker_buy_ratio simultáneamente.
> ❌ Prohibido: modificar HP de XGB entre control y tratamiento.
> ✅ Permitido: 14 features control → 15 features (14 + feature_nuevo).

### Pasos Operativos

#### 3.1 Registrar el feature profile

En [feature_profiles.py](file:///home/juan/Desktop/Projects/AlphaQuant/src/pipeline/feature_profiles.py#L99-L121):

```python
# Ejemplo: feature "volume_delta_1h" (NO implementado aún — plantilla)
FEATURE_PROFILES["volume_delta_1h"] = FeatureProfile(
    name="volume_delta_1h",
    enrichments=("technicals", "sentiment", "volume_delta_1h"),
    treatment_col="volume_delta_1h",
    extra_csv_requirements=("1h.csv",),  # si requiere data TF adicional
)

ENRICHMENT_REGISTRY["volume_delta_1h"] = _apply_volume_delta_1h
```

El `treatment_col` es la columna que:

- Se INCLUYE en el dataset (queda presente en `df`)
- Se EXCLUYE de `control_features`
- Se AGREGA en `treatment_feats = control_feats + [treatment_col]`

Esto asegura apples-to-apples: exactamente el mismo dataframe para ambas variantes.

#### 3.2 Prevenir leakage — Crear test de integración

Antes de correr NI UN SOLO walk-forward, crear un test en `tests/features/` que verifique que el feature NO contiene datos del futuro.

Ejemplo (modelo de [test_funding_rate_leakage.py](file:///home/juan/Desktop/Projects/AlphaQuant/tests/features/test_funding_rate_leakage.py)):

```python
def test_volume_delta_no_lookahead():
    # Setup: mock dataframe con volume_delta_1h
    # Verificar: para toda barra i, volume_delta_1h[i] NO depende de close[i+1] ni close[i+2] ni close[futuro cualquiera]
    # Verificar: merge_asof(direction='backward') NUNCA con 'forward'
    # Verificar: HTF diario tiene shift +1 día ANTES del merge
    ...
```

**Si el test no pasa → ARREGLAR el feature antes de seguir. No existe "leakage pequeño". Cualquier look-ahead invalida completamente el PF.**

#### 3.3 Ejecutar el A/B test

```bash
# A/B test: trend_htf en BTC_USDT, timeframes 4h y 1h
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# Con descarga previa de data (sin --binance-rest; ver nota en §1.1)
python -m tools.aq ab-test BTC_USDT --profile funding_rate --fetch --timeframes 4h 1h
```

**Pipeline interno (por cada tf × formulation):**

```
build_dataset(profile=TREATMENT_PROFILE) → df_common (ya incluye la col treatment)
  │
  ├─ VARIANT=CONTROL: features = control_feats (14 cols, SIN treatment)
  │    └─ run_walk_forward → PF_control, trades_control
  │
  ├─ VARIANT=TREATMENT: features = control_feats + [treatment_col] (15 cols)
  │    └─ run_walk_forward → PF_treatment, trades_treatment
  │
  └─ Paired Block Bootstrap:
       Por cada ventana OOS (mismo índice temporal):
           ΔPF_w = PF_treatment,w − PF_control,w
       Muestrear 1000 veces 8 bloques contiguos por ventana
       → ΔPF_p5, ΔPF_p95
       → passes_gate = (trades ≥ 300) AND (ΔPF_p5 > 0.0)
```

#### 3.4 Interpretar resultados

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

> **⚠️ Semántica del `p5` en esta tabla:** La columna `p5` muestra el **ΔPF p5 del paired block bootstrap** — es decir, el percentil 5 de la distribución de `PF(variante) − PF(naive_long)`. **No** es el PF absoluto de esa variante. Un `p5=-0.0021` para CONTROL significa que con 95% de confianza el control supera al naive por al menos −0.0021 (es decir, es marginalmente peor o igual al naive). Un `p5=+0.0188` para TREATMENT significa que con 95% de confianza el tratamiento supera al naive por al menos +0.0188.

**Matriz de decisión:**

| Caso                                                         | Acción                                                                                                                                                                                        |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0/8` combinaciones PASS                                     | **Feature fallido.** Mover script a `tools/legacy_archive/`. No volver a tocar.                                                                                                               |
| `1-2/8` PASS pero es siempre la MISMA tf × formulation       | **Resultado sospechoso.** Possible overfitting accidental a esa configuración específica. Replicar con diferente `random_state` o `n_bootstrap=5000`. Si sigue pasando → promover.            |
| `3+/8` combinaciones PASS (consistencia entre formulaciones) | **Feature exitoso.** Integrar al pipeline de producción. Agregar el enrichment al perfil "control" para los próximos baselines? → NO: seguir protocol de 1 feature a la vez hasta saturación. |

#### 3.5 Archivado de experimentos fallidos

TODO experimento que no pasó el gate DEBE ser archivado en `tools/legacy_archive/`. El archivo preserva exactamente el código que lo corrió.

**Razón:** Meta-análisis a largo plazo. Si 3 features basados en volatility fallaron, no volver a proponer un feature volatility sin cambiar algo fundamental. El archivo evita "repetir los mismos errores 6 meses después".

**Historial del repositorio (lecciones aprendidas):**

- **Exp01 trend_htf:** 0/8 PASS. Distancia a EMA200 diaria NO mejora PF vs naive.
- **Exp02 funding_rate_current:** 1/8 PASS (solo 4h×multiclass_3, Δp5=+0.0799). Resultado estadísticamente indistinguible de ruido dado el volumen de pruebas acumuladas (~1.2 falsos positivos esperados en 24 configs evaluadas). No se sostiene cross-formulación en el mismo TF (4h×binary empeora con el mismo feature). Tratado como DESCARTADO, no como hallazgo parcial.
- **Exp03 taker_buy_ratio:** 0/8 PASS. Ratio puntual de volumen agresor NO es señal ortogonal.
- **Exp04 regression_return:** 0/6 PASS. Formulación de regresión de retorno continuo (`target_ret`). 0/6 configs (3 activos × 2 TFs) pasan el gate `ΔPF_p5 > 0.0` tras corregir el bug de sentinel (`THRESHOLD_NOT_FOUND = -1.0`). Muestra compresión masiva de varianza en predicciones (~26× vs varianza real del target). Descartado.

---

## Apéndice C — Cierre de la Investigación On-Chain (Piloto BTC_USDT, Agosto 2026)

### Resumen

Un estudio piloto evaluó dos fuentes de datos on-chain/mempool como features ortogonales en BTC_USDT. La investigación siguió el protocolo estándar de A/B test (swing=10, tp=1.5×ATR, sl=1.0×ATR, ventana=6m/step=6m, bootstrap 1000/8 bloques, seed=42) con las tres formulaciones (binary_homerun, multiclass_3, regression_return) en timeframes 4h y 1h.

**Antes de correr cualquier A/B test**, un barrido de config (159 runs, 24 combinaciones de swing×tp×sl, perfil CONTROL únicamente) confirmó que la config default actual (swing=10, tp=1.5, sl=1.0) no es claramente subóptima. Ninguna combinación alternativa superó el chequeo de rigor de 3 criterios (estabilidad de semilla, consistencia cross-formulación, ancho de CI). Config default sin cambios.

### Features On-Chain Evaluados

#### Exp05 — `onchain_active_addresses` (Blockchain.com)
- **Fuente:** `https://api.blockchain.info/charts/n-unique-addresses?timespan=all&format=json`
- **Métrica:** Conteo diario de direcciones Bitcoin únicas utilizadas — medida directa de actividad de red, sin clasificación de wallets de exchange.
- **Shift aplicado:** +2 días (conservador; Blockchain.com no publica SLA de latencia para este endpoint).
- **Cobertura:** 2009 al presente, 1 request HTTP, sin API key.
- **Resultado:** 0/6 PASS en el gate. Un único PASS bruto (1h×binary_homerun TREATMENT, p5=+0.0082) no superó el rigor: la semilla 99 dio p5=−0.0007, y ambas formulaciones hermanas (multiclass_3 p5=−0.1003, regression_return p5=−0.0020) fallaron. **DESCARTADO.**

#### Exp06 — `mempool_fee_rate_p50` (mempool.space)
- **Fuente:** `https://mempool.space/api/v1/mining/blocks/fee-rates/all`
- **Métrica:** Mediana del fee-rate (sat/vB, p50) agregada sobre ~144–153 bloques confirmados por día calendario. Proxy directo de presión de congestión del mempool.
- **Shift aplicado:** +1 día (igual que trend_htf). El backend de mempool.space indexa fee_rate_percentiles de forma sincrónica en el mismo ciclo de procesamiento de bloque que la confirmación, directamente desde Bitcoin Core RPC — sin pipeline asíncrono ni delay adicional (verificado en `backend/src/api/blocks.ts`).
- **Asignación de día:** Verificada contra 6 entradas reales de la API: los 2420 timestamps de 2020–2026 caen todos entre las 06:43 y las 14:10 UTC (0 entradas en zonas de riesgo de corrimiento de día). `normalize()` siempre asigna el día calendario correcto.
- **Cobertura:** 2009 al presente, 1 request HTTP (~955 KB), sin API key. Sin posibilidad de revisión retroactiva (los bloques Bitcoin confirmados son inmutables).
- **Resultado:** 0/6 PASS en el gate. Un único PASS bruto (1h×regression_return TREATMENT, p5=+0.0094) no superó el rigor: la estabilidad de semilla pasó (las 3 semillas positivas, muy estables), pero ambas formulaciones hermanas fallaron (binary_homerun p5=−0.0785, multiclass_3 p5=−0.0303). **DESCARTADO.**

### Infraestructura Construida (Retenida Independientemente de los Resultados)

Toda la infraestructura del piloto está commiteada y con cobertura de tests. Permanece disponible para uso futuro en otros activos o configuraciones:

| Componente | Archivo | Notas |
|---|---|---|
| Fetcher (Blockchain.com) | `src/brain/data_fetcher.py` → `fetch_onchain_active_addresses` | Shift +2d en merge |
| Fetcher (mempool.space) | `src/brain/data_fetcher.py` → `fetch_mempool_fee_rate_median` | Shift +1d en merge |
| Path helpers | `src/config/paths.py` | `get_onchain_active_addresses_path`, `get_mempool_fee_rate_path`, loaders |
| Funciones de merge | `src/brain/features.py` | `add_onchain_active_addresses` (+2d), `add_mempool_fee_rate_p50` (+1d) |
| Perfiles de enriquecimiento | `src/pipeline/feature_profiles.py` | `onchain_activity`, `onchain_fee_pressure` |
| Tests de leakage | `tests/features/` | `test_onchain_active_addresses_leakage.py` (8 tests), `test_mempool_fee_rate_leakage.py` (9 tests) |

**Suite de tests total tras el piloto:** 327 tests (318 pre-piloto + 9 nuevos).

### Conclusiones

Ninguna fuente on-chain produjo una señal que generalice entre formulaciones bajo el protocolo actual en BTC_USDT. El patrón consistente a través de los 6 experimentos A/B (trend_htf, funding_rate, taker_buy_ratio, formulación regression_return, onchain_active_addresses, mempool_fee_rate_p50) es que los PASS brutos no sobreviven la validación cross-formulación. Esto es consistente con ~54 comparaciones acumuladas generando ruido a la tasa esperada de falsos positivos.

**Próximos pasos (no pre-comprometidos):** Piloto en ETH_USDT con los mismos features on-chain, si está justificado; o fuentes de datos alternativas (Dune Analytics, Etherscan) para métricas específicas de ETH. No hay re-evaluación programada de los 6 features descartados — permanecen como candidatos únicamente si evidencia independiente lo justifica.

### 🔴 ¿Por qué hacemos esto?

**Razón matemática:** El paired block bootstrap es la diferencia de métodos que hace que el gate sea válido. Si usaramos bootstrap INDEPENDIENTE para control y tratamiento (muestreando trades individuales i.i.d.), estaríamos subestimando masivamente la covarianza entre ambos (corren en los MISMO folds!). El pairing captura exactamente: "dada la MISMA historia temporal, ¿qué tan mejor es el tratamiento?". En estadística, este es el diseño "matched pairs" que tiene la potencia máxima para detectar efectos pequeños.

**Razón cuantitativa:** "Un feature ortogonal por experimento" proviene del cálculo de grados de freedom. Si agregas 3 features simultáneamente y el ΔPF p5 = +0.08, ¿cuál de las 3 features aportó la mejora? ¿Las 3? ¿1 sola? ¿2 en interacción? No se puede saber sin un test adicional con 2^3=8 grupos. Manteniendo k=1 variable por test, la atribución causal es directa.

**Razón de ingeniería:** El test de leakage previene el error más embarazoso (y caro) en quant trading: pasar el gate OOS, poner el modelo en producción, y después de 3 meses darse cuenta que el feature "exitoso" usaba el cierre de la vela siguiente para calcularse. Todo el "alpha" era leakage; el P&L real de producción es PF=0.5 y perdiste dinero.

---

## Fase 4 — Integración y Monitoreo (Producción)

### Objetivo

Poner el modelo validado (que pasó baseline gate y A/B test gate) en el loop de evaluación diaria: detección de señales → notificación Telegram → ejecución opcional en Binance Futures.

### Pasos Operativos

#### 4.1 Entrenar el modelo final

```bash
# Ejecutar strategy_optimizer.py para el símbolo aprobado
python -m src.brain.strategy_optimizer BTC_USDT --timeframe 4h
```

Esto escribe `data/models/BTC_USDT/config.json` con:

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

Luego:

```bash
python -m src.brain.train BTC_USDT --timeframe 4h
```

Genera `BTC_USDT_1_5_1_0_10_0-42.pkl` serializado con joblib.

#### 4.2 Configurar entorno (.env)

```bash
cp .env.example .env
# Editar:
TELEGRAM_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
AUTHORIZED_CHAT_ID=987654321
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=True  # Primero PROBAR en Testnet
```

#### 4.3 Registrar símbolo en bot_state.json (vía Telegram)

```
Usuario: /start → Bot envía menú principal
Bot → [Bot] → [➕ Add Symbol] → usuario escribe: ETH_USDT
    → ✅ Symbol added successfully.
```

Esto modifica `data/bot_state.json`:

```json
{
  "symbols": {
    "futures": ["BTC_USDT", "ETH_USDT"]
  }
}
```

#### 4.4 Iniciar el bot + scheduler

```bash
python main.py
```

El scheduler APScheduler ejecuta `daily_market_evaluation()` todos los días a las **21:00 ART (America/Argentina/Cordoba)**.

**Cada ejecución evalúa este flujo:**

```
Para cada .pkl en data/models/{SYMBOL}/*.pkl:
  1. fetch_ohlcv_binance(limit=100) → últimas 100 velas timeframe
  2. compute_all_technicals + add_sentiment
  3. model.predict_proba(last_candle[features])[0, class_1]
  4. IF proba >= optimal_threshold:
       a. TP = close + (atr_14 × atr_tp_multi)
       b. SL = close − (atr_14 × atr_sl_multi)
       c. send_trade_signal → Telegram (HTML: Pair / Strategy / Entry / TP / SL)
       d. IF executor inicializado:
           i.  _has_open_position(symbol) → si hay → skip (sin averaging)
           ii. _configure_symbol → margin ISOLATED, leverage x2
           iii. _calculate_quantity → 1% balance, step_size compliance
           iv.  MARKET order entry
           v.   STOP_MARKET + TAKE_PROFIT_MARKET (closePosition=TRUE)
           vi. send_execution_result → Telegram
```

#### 4.5 Comandos del bot de monitoreo

| Menú → Acción                 | Descripción                                                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bot → Status**              | Activo/Pausado, N símbolos monitoreados, próximo scheduler                                                                                               |
| **Bot → Pause / Resume**      | Toggle `bot_active` en `bot_state.json`. Pausado = NO evalúa señales.                                                                                    |
| **Bot → Train**               | Dispara `run_full_training_pipeline` en background para todos los activos. Se saltea si last_trained < 14 días.                                          |
| **Futures → Balance**         | Consulta USDT disponible en Futures vía API.                                                                                                             |
| **Futures → Positions**       | Lista todas las posiciones abiertas (positionAmt != 0).                                                                                                  |
| **Futures → Scan**            | `daily_market_evaluation()` on-demand (ahora, no espera 21:00).                                                                                          |
| **Futures → Leverage**        | Modificar `default_leverage` (rango: 1–125). Persistido en `bot_state.json`.                                                                             |
| **Futures → Risk %**          | Modificar `risk_per_trade_pct` (0.01–100). Regla del 1% default.                                                                                         |
| **Futures → Margin Toggle**   | ISOLATED ↔ CROSS margin. **Recomendado: siempre ISOLATED.**                                                                                              |
| **Futures → ⚠️ PANIC BUTTON** | Confirmación → `close_all_positions()`: cierra TODAS las posiciones MARKET y cancela TODAS las órdenes abiertas. Úsalo ante flash crash o bug detectado. |

### 🔴 ¿Por qué hacemos esto?

**Razón matemática:** El sizing `1% del balance × leverage` tiene derivación directa del Criterio de Kelly fraccionado. Kelly completo sería demasiado agresivo para modelos con PF ≤ 1.3 (el noise es alto). Fraccionar a 1% regla de pulgar da una probabilidad de ruina (drawdown > 50%) inferior al 0.1% asumiendo que el PF se mantiene en [1.05, 1.3] (intervalo bootstrap p5–p95).

**Razón cuantitativa:** `ISOLATED margin` sobre `CROSS` es hard constraint. En CROSS, una posición con pérdida grande liquida TODO el balance de la cuenta. En ISOLATED, cada trade usa como collateral exclusivo `margin = notional / leverage`. Si el SL se ejecuta correctamente, la pérdida máxima por trade es acotada e independiente de otras posiciones. El riesgo está segmentado.

**Razón de ingeniería:** El `Training Cooldown = 14 días` equilibra dos fuerzas opuestas:

- **Retrain muy seguido (cada 1 día):** Riesgo de "data drift overfit" — cada retrain introduce un chance de cambiar los pesos sin que la distribución subyacente haya cambiado realmente.
- **Retrain muy lento (cada 90 días):** El modelo se vuelve staler ante regime shifts structurales (ej: BTC ETF aprobado, halving, crack FTX).
- **14 días ≈ 2 semanas:** Es el compromiso óptimo: captura drift gradual sin introducir ruido excesivo por re-entrenamiento.

---

## Fase 5 — Mantenimiento de Código y Testing

### Objetivo

Preservar la integridad del pipeline a largo plazo. Los tests detectan regresiones, el archivo legacy evita repetir experimentos fallidos, y el CI asegura que el código mergeado no rompe producción.

### 5.1 Estructura de tests (pytest)

```
tests/
├── unit/               # tests individuales por función
│   ├── test_helpers.py            # compute_target Numba, profit_factor
│   ├── test_features.py           # Indicadores: RSI, MACD, BB, OBV, EMA
│   ├── test_data_splits.py        # Dynamic splits con embargo
│   ├── test_settings_loader.py    # Merge YAML + bot_state
│   └── test_logging_config.py
│
├── integration/        # tests que cruzan ≥2 módulos
│   ├── test_oos_validation.py     # Walk-forward completo con data mock
│   ├── test_strategy_optimizer.py # Grid search + sanity check OOS
│   ├── test_train.py              # Train factory + serialización pkl
│   ├── test_data_fetcher.py       # Descarga ccxt mockada
│   └── test_tasks.py              # Orquestador con mocks de Binance/Telegram
│
├── features/           # tests de PREVENCIÓN DE LEAKAGE (MÁS IMPORTANTES)
│   ├── test_funding_rate_leakage.py
│   ├── test_trend_htf_leakage.py
│   └── test_taker_buy_ratio_semantics.py
│
├── api/                # tests de integraciones externas
│   ├── test_binance_executor.py   # Sizing + filtros mock
│   ├── test_notifier.py           # Formatos HTML correctos
│   └── test_telegram_handlers.py  # Auth + state machine
│
└── conftest.py         # Fixtures comunes (mocks, dataframes de prueba)
```

#### 5.2 Cómo correr la suite completa

```bash
# Todos los tests
pytest

# Con coverage html
pytest --cov=src --cov-report=html

# Solo tests de leakage (IMPORTANTE correr ANTES de cada A/B test)
pytest tests/features/ -v

# Solo tests de research pipeline
pytest tests/test_oos_validation.py tests/test_strategy_optimizer.py -v
```

### 5.3 Pre-merge Checklist

**ANTES** de hacer merge de cualquier feature nuevo a `main`:

```
□ pytest pasa 100% sin SKIPs críticos
□ tests/features/*_leakage.py pasan (verificación explícita anti-lookahead)
□ Si es un nuevo profile A/B: se registró en feature_profiles.py
□ Si es un feature HTF/external: usa merge_asof(direction='backward')
□ El experimento pasó el gate ΔPF_p5 > 0.0 en al menos 2/8 combinaciones
□ Script legacy (si hubo) movido a tools/legacy_archive/ con comentario
□ docs actualizados si cambió la interfaz pública de aq.py
```

### 5.4 Archivado de Código Legacy

Regla: **NO BORRAR código de experimentos fallidos.** Mover a `tools/legacy_archive/` con una línea de comentario en el encabezado explicando POR QUÉ falló.

Motivo: En 6 meses vas a tener "la brillante idea" de probar distance_to_ema200_daily. Si el código ya está en `legacy_archive/exp01_trend_htf_walkforward.py` con el comentario `0/8 gate PASS, no hay alpha ortogonal`, ahorras 2 semanas de trabajo.

**Contenido actual de legacy_archive:**

| Archivo                                | Resultado                      | Lección                                                      |
| -------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| `exp01_trend_htf_walkforward.py`       | 0/8 PASS                       | EMA200 diaria no agrega valor ortogonal vs naive_long        |
| `exp02_funding_rate_walkforward.py`    | 1/8 PASS (solo 4h×multiclass3) | 1/8 PASS es estadísticamente indistinguible de ruido (~1.2 FP esperados en 24 configs). No se sostiene cross-formulación. Descartado. |
| `exp03_taker_buy_ratio_walkforward.py` | 0/8 PASS                       | Taker buy ratio puntual no es señal ortogonal vs naive       |
| `exp04_regression_return_walkforward.py` | 0/6 PASS                     | Formulación de regresión continua descartada tras corregir el bug de sentinel (-1.0). Todos fallan ΔPF_p5 > 0.0 y sufren compresión de varianza (~26×). |
| `compare_binary_vs_multiclass.py`      | Benchmark interno              | Multiclass 3 tiene ligeramente mejor p5 en activos volátiles |
| `exp_eth_baseline_oos.py`              | Baseline ETH                   | ETH > BTC en ΔPF vs naive_long (revisar con gate corregido)  |
| `reconcile_naive_target_comparison.py` | Debug target                   | Numba vs Python target coinciden                             |

### 5.5 Monitoreo Post-Producción

#### Revisión semanal (cada 7 días)

Consultar historial de trades desde Binance:

```bash
# Descargar historial de trades cerrados desde Binance Futures
# (vía API o exportación manual desde la UI de Binance)
# Luego calcular métricas básicas:
python - <<'EOF'
import pandas as pd, numpy as np

# trades = DataFrame con columnas: side, realizedPnl, time
trades = pd.read_csv("trades_export.csv")  # exportación de Binance UI

wins  = trades["realizedPnl"][trades["realizedPnl"] > 0].sum()
losses = trades["realizedPnl"][trades["realizedPnl"] < 0].abs().sum()
pf_real = wins / max(losses, 1e-9)
print(f"PF real = {pf_real:.3f}  (trades={len(trades)})")
EOF
```

Comparar `pf_real` con el `oos_pf_point` del último baseline report en `reports/{SYMBOL}/latest_baseline.json`.

#### Comparación backtest vs real

| Situación                                    | Diagnóstico                                       | Acción                         |
| -------------------------------------------- | ------------------------------------------------- | ------------------------------ |
| `pf_real ≈ oos_pf_p5` (±10%)                 | Modelo operando dentro de expectativas            | Continuar                      |
| `pf_real < oos_pf_p5 × 0.9` durante 1 semana | Posible ruido estadístico                         | Monitorear más de cerca        |
| `pf_real < 0.85` durante 2+ semanas          | Probable regime shift o bug silencioso            | Pausar bot, re-correr baseline |
| `pf_real > oos_pf_p95 × 1.2`                 | Puede ser suerte o mercado inusualmente favorable | Continuar pero no subir riesgo |

#### Procedimiento de apagado de emergencia

```
Si PF_real < 0.85 por 4 semanas consecutivas:
  1. Telegram → [Futures] → [⚠️ PANIC BUTTON] → confirmar
     (cierra TODAS las posiciones abiertas)
  2. Telegram → [Bot] → [Pause]
     (bot deja de evaluar señales en el cron de 21:00)
  3. En bot_state.json: quitar el símbolo problemático de symbols.futures
  4. Re-correr baseline screening:
     python -m tools.aq baseline {SYMBOL} --timeframes 4h 1h --fetch
  5. Si baseline sigue pasando: analizar si el drift es en el threshold
     (probar strategy_optimizer con datos más recientes)
     Si baseline falla: el activo cambió de régimen. Descartar.
```

**Regla de re-activación:** Solo re-activar el símbolo si el nuevo baseline tiene `ΔPF_p5 > 0.0` (modelo vs naive_long) con datos que incluyan el período de bajo rendimiento detectado.

### 🔴 ¿Por qué hacemos esto?

**Razón matemática:** Los tests de leakage son la única capa de defensa contra "regresiones que no rompen el código pero rompen el P&L". Imagina que alguien refactoriza `add_sentiment` y cambia accidentalmente `direction='backward'` por `direction='nearest'`. Ningún test unitario rompe: el dataframe todavía tiene fng_value, no hay NaNs nuevos. PERO ahora las velas de 22:00 UTC heredan el F&G del día siguiente (forward-looking). El PF se infla artificialmente en backtest de 1.05 a 1.25, y en producción colapsa. Solo `tests/features/*.py` detectan este error.

**Razón cuantitativa:** La suite de pytest controla el Type-I error del framework. Cada merge que rompe un test es un "fallo silencioso prevenido". En sistemas de trading, el coste de un falso positivo (modelo aprobado que en realidad es malo) es órdenes de magnitud mayor que el coste de un falso negativo (modelo bueno rechazado por test estricto). Por eso los tests son deliberadamente conservadores.

**Razón de ingeniería:** El legacy archive documenta las "rutas sin salida". El sesgo de recencia hace que recordemos solo los éxitos. En un año de experimentación fallida, tener un archivo de 20 experimentos con sus razones de fallo es oro: te permite hacer meta-aprendizaje ("los features basados en order flow NO funcionan en este timeframe") y no redescubrir la rueda.

---

## Apéndice A — Resumen de Comandos Cheat Sheet

```bash
# === DATA ===
python -m src.brain.data_fetcher BTC_USDT --timeframe 4h --binance-rest
python -m src.brain.data_fetcher BTC_USDT --funding-rate

# === DIAGNOSTICS ===
python -m tools.aq diagnose-data BTC_USDT --timeframe 4h
python -m tools.aq diagnose-naive-baseline BTC_USDT
python -m tools.aq diagnose-regimes-rigorous BTC_USDT
python -m tools.aq diagnose-swing-and-regimes BTC_USDT
python -m tools.aq diagnose-timeframe-swing-sweep BTC_USDT --timeframe 1h

# === RESEARCH ===
python -m tools.aq baseline ETH_USDT --timeframes 4h 1h --fetch
python -m tools.aq ab-test BTC_USDT --profile trend_htf --timeframes 4h 1h

# === PRODUCCIÓN ===
python -m src.brain.strategy_optimizer BTC_USDT --timeframe 4h
python -m src.brain.train BTC_USDT --timeframe 4h
python main.py  # Inicia bot + scheduler 21:00 ART

# === TESTS ===
pytest
pytest tests/features/ -v                          # Anti-leakage
pytest tests/test_oos_validation.py -v             # Walk-Forward
```

## Apéndice B — Criterios de Éxito Históricos del Repositorio

| Criterio                    | Valor Pre-registrado | Estado Actual                                                                 |
| --------------------------- | -------------------- | ----------------------------------------------------------------------------- |
| `pooled_trade_count` MÍNIMO | `≥ 300`              | Hardcodeado en `oos_validation.py`                                            |
| Baseline Gate (ΔPF vs naive) | `ΔPF_p5 > 0.0`      | Hardcodeado en `oos_validation.py` (`MIN_BOOTSTRAP_P5 = 0.0`) — baseline y A/B comparten el mismo mecanismo paired bootstrap vs naive_long. PF absoluto > 1.0 NO es gate. |
| A/B Test Gate (Delta PF)    | `ΔPF_p5 > 0.0`       | Hardcodeado en `oos_validation.py` (`MIN_BOOTSTRAP_P5 = 0.0`)                 |
| Bootstrap iteraciones       | `1000`               | Default en `ExperimentConfig.n_bootstrap`                                     |
| Bloques por ventana         | `8`                  | Default en `ExperimentConfig.n_blocks`                                        |
| Random state                | `42`                 | Default en `ExperimentConfig.random_state`                                    |
| Swing period default        | `10 barras`          | Default en `ExperimentConfig.swing_period`                                    |
| TP default                  | `1.5 × ATR`          | Default en `ExperimentConfig.tp_multi`                                        |
| SL default                  | `1.0 × ATR`          | Default en `ExperimentConfig.sl_multi`                                        |
| Train window                | `6 meses`            | Default en `ExperimentConfig.window_months`                                   |
| Step entre folds            | `6 meses`            | Default en `ExperimentConfig.step_months`                                     |
| Training cooldown           | `14 días`            | Constante `TRAINING_COOLDOWN_DAYS` en `tasks.py`                              |
| Riesgo por trade            | `1%` de balance      | `risk_per_trade_pct` en `settings.yaml`                                       |
| Leverage default            | `2x` ISOLATED        | `default_leverage` en `settings.yaml`                                         |
| Scheduler diario            | `21:00 ART`          | Cron en `main.py` + `timezone("America/Argentina/Cordoba")`                   |
