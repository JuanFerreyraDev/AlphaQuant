# AlphaQuant — Cómo Agregar un Nuevo Feature Profile

> Guía paso a paso para incorporar un nuevo enrichment al pipeline experimental.
> Sigue exactamente este orden para no romper ningún gate estadístico.

---

## Regla de Oro

> **Un feature ortogonal por experimento.**  
> Un profile nuevo agrega exactamente **una columna** nueva al set de control.  
> Nunca agregar 2 features en el mismo A/B test.

---

## Paso 1 — Implementar la función de enrichment

En `src/brain/features.py`, agregar una función pura que:
- Reciba un DataFrame con el OHLCV ya procesado
- Devuelva `(df, bool)` donde el bool indica si el feature fue calculado correctamente
- Use exclusivamente datos del pasado (sin `shift()` negativo, sin `rolling().shift(-n)`)

```python
# src/brain/features.py

def add_mi_nuevo_feature(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Calcula X a partir de Y. Solo usa datos históricos.

    Returns:
        (df_enriquecido, has_feature): el bool es False si faltan columnas requeridas.
    """
    if "columna_requerida" not in df.columns:
        return df, False

    df = df.copy()
    df["mi_nuevo_feature"] = ...  # lógica usando solo pasado

    return df, True
```

**Reglas anti-leakage:**
- ❌ No usar `df["close"].shift(-n)` con n positivo (datos futuros)
- ❌ No usar `df.rolling(n).mean()` en columnas de precio futuro
- ✅ Usar `merge_asof(direction='backward')` para features de timeframe superior
- ✅ Si el feature es HTF (diario): aplicar `.shift(1)` **antes** del merge (ver `add_trend_htf`)

---

## Paso 2 — Crear el test de leakage (OBLIGATORIO antes del A/B)

En `tests/features/test_mi_nuevo_feature_leakage.py`:

```python
"""Test que verifica que mi_nuevo_feature no contiene datos del futuro."""
import pandas as pd
import numpy as np
import pytest
from src.brain.features import add_mi_nuevo_feature


def _make_mock_ohlcv(n: int = 200) -> pd.DataFrame:
    """Crea un DataFrame OHLCV sintético con DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)


def test_no_lookahead_by_construction():
    """El valor en la barra i solo debe depender de close[:i+1]."""
    df = _make_mock_ohlcv(100)
    df_feat, has_feat = add_mi_nuevo_feature(df)

    assert has_feat, "Feature no fue calculado — verificar columnas requeridas"
    assert "mi_nuevo_feature" in df_feat.columns

    # Verificar que el feature en la barra 50 no cambia al alterar datos futuros
    original_val = df_feat["mi_nuevo_feature"].iloc[50]

    df_modified = df.copy()
    df_modified.loc[df_modified.index[51:], "close"] *= 999  # perturbar el futuro
    df_modified_feat, _ = add_mi_nuevo_feature(df_modified)

    assert df_modified_feat["mi_nuevo_feature"].iloc[50] == pytest.approx(original_val), (
        "LEAKAGE DETECTADO: el valor en barra 50 cambió al modificar datos futuros."
    )


def test_no_nan_after_warmup():
    """Tras el warmup esperado, no debe haber NaN."""
    df = _make_mock_ohlcv(200)
    df_feat, _ = add_mi_nuevo_feature(df)

    WARMUP = 30  # ajustar según la ventana de cálculo del feature
    tail = df_feat["mi_nuevo_feature"].iloc[WARMUP:]
    nan_pct = tail.isna().mean()
    assert nan_pct < 0.01, f"NaN fuera del warmup: {nan_pct:.1%}"


def test_merge_asof_direction():
    """Si el feature usa merge_asof, verificar que direction='backward'."""
    # Ejemplo para features HTF: la barra 4h de 2024-01-02 12:00 UTC
    # no puede tener el valor 1d de 2024-01-02 (todavía no cerró).
    # Si es HTF, agregar este test específico.
    pass
```

Correr antes de continuar:
```bash
pytest tests/features/test_mi_nuevo_feature_leakage.py -v
```

**Si el test no pasa → ARREGLAR el feature antes de seguir. No existe "leakage pequeño".**

---

## Paso 3 — Registrar el profile en `feature_profiles.py`

En `src/pipeline/feature_profiles.py`:

```python
# 1. Agregar la función _apply_* (adapta la firma a (df, symbol) -> df)
def _apply_mi_nuevo_feature(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df, has_feat = add_mi_nuevo_feature(df)
    if not has_feat:
        raise RuntimeError(
            f"mi_nuevo_feature no calculado para {symbol} — "
            "verificar que el CSV tiene las columnas requeridas"
        )
    return df

# 2. Registrar en ENRICHMENT_REGISTRY
ENRICHMENT_REGISTRY["mi_nuevo_feature"] = _apply_mi_nuevo_feature

# 3. Agregar el FeatureProfile
FEATURE_PROFILES["mi_nuevo_feature"] = FeatureProfile(
    name="mi_nuevo_feature",
    enrichments=("technicals", "sentiment", "mi_nuevo_feature"),
    treatment_col="mi_nuevo_feature",        # nombre de la columna en df
    extra_csv_requirements=(),               # si requiere CSV adicional, indicarlo
)
```

> **`treatment_col`:** La columna que se incluye en el DataFrame pero se **excluye** de `control_features` y se agrega solo en la variante TREATMENT. Garantiza que el A/B compare apples-to-apples.

---

## Paso 4 — Correr el A/B test

```bash
# Asegurarse que el baseline del símbolo ya pasó el gate (PF_p5 > 1.0)
python -m tools.aq ab-test BTC_USDT --profile mi_nuevo_feature --timeframes 4h 1h
```

El reporte se guarda en `reports/BTC_USDT/ab_test_mi_nuevo_feature_{timestamp}.json`.

**Gate de éxito:** `pooled_trades ≥ 300` Y `ΔPF_p5 > 0.0` en al menos **3 de 8** combinaciones (2 timeframes × 2 formulations × 2 variantes).

---

## Paso 5 — Decisión

### Si el A/B test PASA el gate (3+/8)

1. Actualizar `docs/ARCHITECTURE.md` §3.2.3 con la nueva fila en la tabla de Feature Profiles
2. Actualizar `docs/WORKFLOW.md` §3.5 con el resultado histórico
3. Para incorporar a producción: re-correr `strategy_optimizer` y `train` con el feature activado
4. Actualizar `data/models/{SYMBOL}/config.json` con la nueva lista de features

### Si el A/B test FALLA (0-2/8)

1. Mover el script de experimento (si lo hay) a `tools/legacy_archive/` con comentario de resultado
2. Documentar la lección en `docs/WORKFLOW.md` §5.4 (tabla de historial)
3. No eliminar el FeatureProfile del código (puede servir para meta-análisis futuro)

---

## Checklist Pre-Merge

```
□ test de leakage pasa 100% (pytest tests/features/ -v)
□ pytest completo pasa sin regresiones
□ FeatureProfile registrado en feature_profiles.py
□ _apply_* registrado en ENRICHMENT_REGISTRY
□ A/B test corrido y reporte guardado en reports/
□ ARCHITECTURE.md §3.2.3 actualizado si el feature se promueve
□ WORKFLOW.md §3.5 actualizado con el resultado (PASS o FAIL + lección)
```

---

## Referencia de Profiles Existentes

| Profile | Treatment col | CSV extra requerido | Estado |
|---------|---------------|---------------------|--------|
| `control` | None (baseline) | — | ✅ Producción |
| `trend_htf` | `trend_htf` | `1d.csv` | ❌ 0/8 gate (ver legacy_archive) |
| `funding_rate` | `funding_rate_current` | `funding_rate.csv` | ⚠️ 1/8 gate (solo 4h×multiclass3) |
| `taker_buy_ratio` | `taker_buy_ratio` | Requiere `--binance-rest` | ❌ 0/8 gate (ver legacy_archive) |
