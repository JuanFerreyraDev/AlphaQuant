"""helpers.py — Trading domain utility functions.

Includes data loading, target computation, temporal split with embargo,
strategy construction, and model training/evaluation.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score

from src.config.settings_loader import get_project_root

logger = logging.getLogger(__name__)


# --- CONSTANTS ---
SENTIMENT_COLS: list[str] = ["fng_value", "fng_sma_14", "fng_vol_14"]

DEFAULT_HP: dict[str, Any] = {
    "n_estimators": 50,
    "max_depth": 2,
    "learning_rate": 0.01,
}


# --- DATA LOADING ---
def load_csv_data(filename: str) -> pd.DataFrame:
    """Load an OHLCV CSV and normalize columns / time index.

    Args:
        filename: CSV filename inside ``data/raw_csv/``.

    Returns:
        DataFrame with a normalized datetime index.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = get_project_root() / "data" / "raw_csv" / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df.set_index("timestamp", inplace=True)
    return df


# --- TARGET ---
def compute_target(
    df: pd.DataFrame,
    swing_days: int,
    atr_tp_multi: float = 0.0,
    atr_sl_multi: float = 0.0,
) -> pd.DataFrame:
    """Compute the ``target`` column using dynamic ATR-based TP/SL.

    Args:
        df: DataFrame with OHLCV columns and ``atr_14``.
        swing_days: Number of future days to evaluate TP/SL.
        atr_tp_multi: ATR multiplier for Take Profit.
        atr_sl_multi: ATR multiplier for Stop Loss.

    Returns:
        DataFrame with ``target`` column added.

    Raises:
        ValueError: If the ``atr_14`` column is missing.
    """
    if "atr_14" not in df.columns:
        raise ValueError(
            "The 'atr_tp_sl' mode requires the 'atr_14' column. "
            "Run compute_all_technicals() before compute_target()."
        )
    df["max_high_future"] = df["high"].shift(-1).rolling(window=swing_days).max()
    df["min_low_future"] = df["low"].shift(-1).rolling(window=swing_days).min()

    tp_price = df["close"] + (df["atr_14"] * atr_tp_multi)
    sl_price = df["close"] - (df["atr_14"] * atr_sl_multi)

    df["target"] = (
        (df["max_high_future"] >= tp_price) & (df["min_low_future"] > sl_price)
    ).astype(int)

    return df


def cleanup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove OHLCV and auxiliary columns that are not features.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame without auxiliary columns and without NaN rows.
    """
    cols_to_drop = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_50",
        "vol_sma_20",
        "max_high_future",
        "min_low_future",
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df.dropna(inplace=True)
    return df


# --- TEMPORAL SPLIT ---
def temporal_split_with_embargo(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.1,
    embargo_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train/val/test with embargo between blocks.

    Args:
        df: Temporally ordered DataFrame.
        train_pct: Proportion of data for training.
        val_pct: Proportion of data for validation.
        embargo_days: Candles of separation to prevent data leakage.

    Returns:
        Tuple ``(df_train, df_val, df_test)``.
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_start = train_end + embargo_days
    val_end = int(n * (train_pct + val_pct))
    test_start = val_end + embargo_days

    return (
        df.iloc[:train_end],
        df.iloc[val_start:val_end],
        df.iloc[test_start:],
    )


# --- OPTIMAL THRESHOLD ---
def find_optimal_threshold(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tp_val: float,
    sl_val: float,
) -> tuple[float, float]:
    """Find the threshold that maximizes net profit on the validation set.

    Args:
        model: Trained XGBoost model.
        X_val: Validation features.
        y_val: Validation targets.
        tp_val: Take Profit multiplier.
        sl_val: Stop Loss multiplier.

    Returns:
        Tuple ``(best_threshold, best_profit)``.
    """
    proba: np.ndarray = model.predict_proba(X_val)[:, 1]

    best_threshold: float = 0.50
    best_profit: float = -np.inf

    min_signals_val = max(8, int(len(y_val) * 0.05))

    for thresh in np.arange(0.50, 0.85, 0.01):
        preds = (proba >= thresh).astype(int)
        n_signals = int(preds.sum())

        if n_signals < min_signals_val:
            continue

        aciertos = int(((y_val == 1) & (preds == 1)).sum())
        fallos = int(((y_val == 0) & (preds == 1)).sum())
        profit = aciertos * tp_val - fallos * sl_val

        if profit > best_profit:
            best_profit = profit
            best_threshold = float(thresh)

    if best_profit == -np.inf:
        return 0.65, 0.0

    return best_threshold, best_profit


# --- STRATEGIES ---
def build_strategies(
    df: pd.DataFrame,
    has_sentiment: bool,
) -> dict[str, list[str]]:
    """Generate the complete strategy dictionary (base + sentiment variants).

    Args:
        df: DataFrame with computed features.
        has_sentiment: Whether sentiment data is available.

    Returns:
        Dictionary ``{strategy_name: [feature_list]}``.
    """
    base_strategies: dict[str, list[str]] = {
        "Puramente Momentum": ["rsi_14", "macd", "macd_hist", "stoch_k"],
        "Seguidor de Tendencia": ["dist_ema_50", "adx_14"],
        "Cazador de Volatilidad": ["atr_14", "bb_width", "bb_pos"],
        "El Confirmador (Volumen)": ["obv", "rel_volume"],
        "Momentum + Volatilidad": ["rsi_14", "macd_hist", "atr_14", "bb_width"],
        "Tendencia + Volumen": ["dist_ema_50", "adx_14", "obv", "rel_volume"],
        "EL FRANKENSTEIN (Técnicos)": [
            c for c in df.columns if c not in ["target"] + SENTIMENT_COLS
        ],
    }

    all_strategies: dict[str, list[str]] = {}
    for name, features in base_strategies.items():
        all_strategies[name] = features
        if has_sentiment:
            all_strategies[f"{name} + Sentimiento"] = features + SENTIMENT_COLS

    return all_strategies


# --- TRAINING AND EVALUATION (per strategy) ---
def train_and_evaluate(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    tp_val: float,
    sl_val: float,
    hyperparams: Optional[dict[str, Any]] = None,
) -> tuple[xgb.XGBClassifier, dict[str, Any], np.ndarray, list[Any], float]:
    """Train an XGBClassifier and return metrics evaluated on the validation set.

    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        y_train: Training target.
        y_val: Validation target.
        y_test: Test target.
        tp_val: TP multiplier for profit calculation.
        sl_val: SL multiplier for profit calculation.
        hyperparams: Optional XGBoost hyperparameters.

    Returns:
        Tuple ``(model, metrics, preds_test, buy_dates_all, best_threshold)``.
    """
    hp = hyperparams or DEFAULT_HP

    imbalance = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        scale_pos_weight=imbalance,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    best_threshold, val_profit = find_optimal_threshold(
        model, X_val, y_val, tp_val, sl_val
    )

    y_probs_val: np.ndarray = model.predict_proba(X_val)[:, 1]
    preds_val = (y_probs_val >= best_threshold).astype(int)

    prec_val: float = precision_score(y_val, preds_val, zero_division=0)
    señales_val = int(sum(preds_val))
    aciertos_val = int(sum((preds_val == 1) & (y_val == 1)))

    profit_neto_val = aciertos_val * tp_val - (señales_val - aciertos_val) * sl_val
    profit_factor_val = (aciertos_val * tp_val) / max(
        (señales_val - aciertos_val) * sl_val, 0.01
    )

    metrics: dict[str, Any] = {
        "Profit_Neto": round(profit_neto_val, 2),
        "Profit_Factor": round(profit_factor_val, 2),
        "Precisión": round(prec_val * 100, 2),
        "Señales_Val": señales_val,
        "Aciertos_Val": aciertos_val,
        "Umbral_Opt": round(best_threshold, 3),
    }

    y_probs_test: np.ndarray = model.predict_proba(X_test)[:, 1]
    preds_test = (y_probs_test >= best_threshold).astype(int)

    buy_dates_val = y_val.index[preds_val == 1].tolist()
    buy_dates_test = y_test.index[preds_test == 1].tolist()
    buy_dates_all = buy_dates_val + buy_dates_test

    return model, metrics, preds_test, buy_dates_all, best_threshold
