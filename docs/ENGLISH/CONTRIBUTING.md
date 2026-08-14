# AlphaQuant — How to Add a New Feature Profile

> Step-by-step guide to incorporating a new enrichment into the experimental pipeline.
> Follow this exact order to avoid breaking any statistical gates.

---

## Golden Rule

> **One orthogonal feature per experiment.**  
> A new profile adds exactly **one new column** to the control set.  
> Never add 2 features in the same A/B test.

---

## Step 1 — Implement the enrichment function

In `src/brain/features.py`, add a pure function that:
- Receives a DataFrame with pre-processed OHLCV
- Returns `(df, bool)` where the bool indicates whether the feature was correctly calculated
- Exclusively uses past data (no negative `shift()`, no `rolling().shift(-n)`)

```python
# src/brain/features.py

def add_my_new_feature(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Calculates X from Y. Uses historical data only.

    Returns:
        (enriched_df, has_feature): bool is False if required columns are missing.
    """
    if "required_column" not in df.columns:
        return df, False

    df = df.copy()
    df["my_new_feature"] = ...  # logic using past data only

    return df, True
```

**Anti-leakage rules:**
- ❌ Do not use `df["close"].shift(-n)` with positive n (future data)
- ❌ Do not use `df.rolling(n).mean()` on future price columns
- ✅ Use `merge_asof(direction='backward')` for higher timeframe (HTF) features
- ✅ If the feature is HTF (daily): apply `.shift(1)` **before** merging (see `add_trend_htf`)

---

## Step 2 — Create the leakage test (MANDATORY before A/B)

In `tests/features/test_my_new_feature_leakage.py`:

```python
"""Test verifying that my_new_feature contains no future data."""
import pandas as pd
import numpy as np
import pytest
from src.brain.features import add_my_new_feature


def _make_mock_ohlcv(n: int = 200) -> pd.DataFrame:
    """Creates a synthetic OHLCV DataFrame with a DatetimeIndex."""
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
    """Value at bar i must only depend on close[:i+1]."""
    df = _make_mock_ohlcv(100)
    df_feat, has_feat = add_my_new_feature(df)

    assert has_feat, "Feature was not calculated — check required columns"
    assert "my_new_feature" in df_feat.columns

    # Verify that the feature at bar 50 does not change when altering future data
    original_val = df_feat["my_new_feature"].iloc[50]

    df_modified = df.copy()
    df_modified.loc[df_modified.index[51:], "close"] *= 999  # perturb future data
    df_modified_feat, _ = add_my_new_feature(df_modified)

    assert df_modified_feat["my_new_feature"].iloc[50] == pytest.approx(original_val), (
        "LEAKAGE DETECTED: value at bar 50 changed when modifying future data."
    )


def test_no_nan_after_warmup():
    """After expected warmup, there should be no NaNs."""
    df = _make_mock_ohlcv(200)
    df_feat, _ = add_my_new_feature(df)

    WARMUP = 30  # adjust according to calculation window
    tail = df_feat["my_new_feature"].iloc[WARMUP:]
    nan_pct = tail.isna().mean()
    assert nan_pct < 0.01, f"NaN outside warmup: {nan_pct:.1%}"


def test_merge_asof_direction():
    """If feature uses merge_asof, verify direction='backward'."""
    # Example for HTF features: 4h bar at 2024-01-02 12:00 UTC
    # cannot have the 1d value for 2024-01-02 (not yet closed).
    # If HTF, add this specific test.
    pass
```

Run before proceeding:
```bash
pytest tests/features/test_my_new_feature_leakage.py -v
```

**If test fails → FIX the feature before continuing. There is no such thing as "minor leakage".**

---

## Step 3 — Register profile in `feature_profiles.py`

In `src/pipeline/feature_profiles.py`:

```python
# 1. Add _apply_* helper (adapts signature to (df, symbol) -> df)
def _apply_my_new_feature(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df, has_feat = add_my_new_feature(df)
    if not has_feat:
        raise RuntimeError(
            f"my_new_feature not calculated for {symbol} — "
            "check that CSV has required columns"
        )
    return df

# 2. Register in ENRICHMENT_REGISTRY
ENRICHMENT_REGISTRY["my_new_feature"] = _apply_my_new_feature

# 3. Add FeatureProfile
FEATURE_PROFILES["my_new_feature"] = FeatureProfile(
    name="my_new_feature",
    enrichments=("technicals", "sentiment", "my_new_feature"),
    treatment_col="my_new_feature",        # column name in df
    extra_csv_requirements=(),               # specify if extra CSV required
)
```

> **`treatment_col`:** The column included in the DataFrame but **excluded** from `control_features`, added only in the TREATMENT variant. Guarantees that A/B compares apples-to-apples.

---

## Step 4 — Run A/B test

```bash
# Ensure baseline for symbol already passed the gate (PF_p5 > 1.0)
python -m tools.aq ab-test BTC_USDT --profile my_new_feature --timeframes 4h 1h
```

Report is saved in `reports/BTC_USDT/ab_test_my_new_feature_{timestamp}.json`.

**Success Gate:** `pooled_trades ≥ 300` AND `ΔPF_p5 > 0.0` in at least **3 out of 8** combinations (2 timeframes × 2 formulations × 2 variants).

---

## Step 5 — Decision

### If A/B test PASSES gate (3+/8)

1. Update `docs/ARCHITECTURE.md` §3.2.3 with new row in Feature Profiles table
2. Update `docs/WORKFLOW.md` §3.5 with historical result
3. To incorporate into production: re-run `strategy_optimizer` and `train` with feature enabled
4. Update `data/models/{SYMBOL}/config.json` with new features list

### If A/B test FAILS (0-2/8)

1. Move experiment script (if any) to `tools/legacy_archive/` with result comment
2. Document lesson in `docs/WORKFLOW.md` §5.4 (history table)
3. Do not remove FeatureProfile from code (may serve future meta-analysis)

---

## Pre-Merge Checklist

```
□ leakage test passes 100% (pytest tests/features/ -v)
□ full pytest suite passes without regressions
□ FeatureProfile registered in feature_profiles.py
□ _apply_* registered in ENRICHMENT_REGISTRY
□ A/B test executed and report saved under reports/
□ ARCHITECTURE.md §3.2.3 updated if feature is promoted
□ WORKFLOW.md §3.5 updated with result (PASS or FAIL + lesson)
```

---

## Existing Profiles Reference

| Profile | Treatment col | Extra CSV required | Status |
|---------|---------------|--------------------|--------|
| `control` | None (baseline) | — | ✅ Production |
| `trend_htf` | `trend_htf` | `1d.csv` | ❌ 0/8 gate (see legacy_archive) |
| `funding_rate` | `funding_rate_current` | `funding_rate.csv` | ⚠️ 1/8 gate (4h×multiclass3 only) |
| `taker_buy_ratio` | `taker_buy_ratio` | Requires `--binance-rest` | ❌ 0/8 gate (see legacy_archive) |
