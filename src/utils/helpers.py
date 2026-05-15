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

from src.config.settings_loader import get_project_root, get_trading_settings

logger = logging.getLogger(__name__)


# --- CONSTANTS ---
SENTIMENT_COLS: list[str] = ["fng_value", "fng_sma_14", "fng_vol_14"]

DEFAULT_HP: dict[str, Any] = {
    "n_estimators": 200,
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
    lower_tf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute the ``target`` column using dynamic ATR-based TP/SL.

    Args:
        df: DataFrame with OHLCV columns and ``atr_14``.  Index must be a
            timezone-naive ``DatetimeIndex`` (daily bars).
        swing_days: Number of future days to evaluate TP/SL.
        atr_tp_multi: ATR multiplier for Take Profit.
        atr_sl_multi: ATR multiplier for Stop Loss.
        lower_tf_df: Optional lower-timeframe DataFrame (e.g. 1 h or 15 m)
            with a ``DatetimeIndex`` and ``high`` / ``low`` columns.  When
            provided it is used **exclusively** to determine which level —
            Take Profit or Stop Loss — is reached first within the
            ``swing_days`` window.  Its columns are **never** added to ``df``
            as XGBoost features.  When ``None`` the function falls back to the
            daily approximation so the existing data pipeline keeps working.

    Returns:
        DataFrame with ``target`` column added in-place.

    Raises:
        ValueError: If the ``atr_14`` column is missing.
        ValueError: If ``lower_tf_df`` is missing ``high`` or ``low``.
    """
    if "atr_14" not in df.columns:
        raise ValueError(
            "The 'atr_tp_sl' mode requires the 'atr_14' column. "
            "Run compute_all_technicals() before compute_target()."
        )

    tp_price: pd.Series = df["close"] + (df["atr_14"] * atr_tp_multi)
    sl_price: pd.Series = df["close"] - (df["atr_14"] * atr_sl_multi)

    if lower_tf_df is not None:
        # --- Multi-timeframe path -------------------------------------------
        missing = {"high", "low"} - set(lower_tf_df.columns)
        if missing:
            raise ValueError(
                f"lower_tf_df is missing required columns: {missing}"
            )

        targets: list[int] = []
        for i, (date, row) in enumerate(df.iterrows()):
            window_start = date + pd.Timedelta(days=1)
            window_end = date + pd.Timedelta(days=swing_days)
            ltf_window = lower_tf_df.loc[
                (lower_tf_df.index >= window_start)
                & (lower_tf_df.index <= window_end)
            ]

            label = 0  # default: SL hit or neither level reached
            for _, ltf_row in ltf_window.iterrows():
                # Pessimistic intrabar resolution: if both levels are touched
                # in the same bar, assume Stop Loss was hit first.
                if ltf_row["low"] <= sl_price.iloc[i]:
                    label = 0
                    break
                if ltf_row["high"] >= tp_price.iloc[i]:
                    label = 1
                    break
            targets.append(label)

        df["target"] = targets
    else:
        # --- Daily fallback path (original logic) ----------------------------
        df["max_high_future"] = df["high"].shift(-1).rolling(window=swing_days).max()
        df["min_low_future"] = df["low"].shift(-1).rolling(window=swing_days).min()

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
    prices_val: pd.DataFrame,
    fee_rate: float,
    slippage: float,
) -> tuple[float, float]:
    """Find the threshold that maximises net return on the validation set.

    The profit/loss per trade is computed as an **actual return percentage**
    so that ``tp_val`` / ``sl_val`` (ATR multipliers) are converted to price
    distances before comparing them to ``fee_rate`` and ``slippage``.

    Args:
        model: Trained XGBoost model.
        X_val: Validation features.
        y_val: Validation targets.
        tp_val: ATR multiplier for Take Profit.
        sl_val: ATR multiplier for Stop Loss.
        prices_val: DataFrame aligned with ``X_val`` containing at least
            ``close`` and ``atr_14`` columns.
        fee_rate: Exchange fee rate per side (e.g. ``0.001`` for 0.1 %).
        slippage: Estimated slippage per trade (e.g. ``0.0005``).

    Returns:
        Tuple ``(best_threshold, best_profit)`` where ``best_profit`` is the
        total net return across all signals at that threshold.
    """
    proba: np.ndarray = model.predict_proba(X_val)[:, 1]
    cost_per_trade: float = (2.0 * fee_rate) + (2.0 * slippage)

    best_threshold: float = 0.50
    best_profit: float = -np.inf

    min_signals_val = max(8, int(len(y_val) * 0.05))

    for thresh in np.arange(0.50, 0.85, 0.01):
        preds = (proba >= thresh).astype(int)
        n_signals = int(preds.sum())

        if n_signals < min_signals_val:
            continue

        signal_mask = preds == 1
        win_mask = signal_mask & (y_val.values == 1)
        loss_mask = signal_mask & (y_val.values == 0)

        atr = prices_val["atr_14"].values
        close = prices_val["close"].values

        win_returns = (atr[win_mask] * tp_val) / close[win_mask] - cost_per_trade
        loss_returns = -(atr[loss_mask] * sl_val) / close[loss_mask] - cost_per_trade

        profit = float(win_returns.sum() + loss_returns.sum())

        if profit > best_profit:
            best_profit = profit
            best_threshold = float(thresh)

    if best_profit == -np.inf:
        return 0.65, 0.0

    return best_threshold, best_profit


# --- FITNESS SCORE (VALIDATION SET ONLY) ---
def fitness_score(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    prices_val: pd.DataFrame,
    tp_val: float,
    sl_val: float,
    fee_rate: float,
    slippage: float,
    threshold: float,
) -> tuple[float, dict[str, Any]]:
    """Compute a composite fitness score for model selection on the **validation set**.

    The score balances Profit Factor, Maximum Drawdown, and trade frequency
    so that the optimizer selects robust strategies rather than over-fitted
    ones with high raw profit on a tiny number of trades.

    Args:
        model: Trained XGBoost model.
        X_val: Validation features.
        y_val: Validation targets.
        prices_val: DataFrame with ``close`` and ``atr_14`` aligned to
            ``X_val``.
        tp_val: ATR multiplier for Take Profit.
        sl_val: ATR multiplier for Stop Loss.
        fee_rate: Exchange fee rate per side.
        slippage: Estimated slippage per trade.
        threshold: Decision boundary from ``find_optimal_threshold``.

    Returns:
        Tuple ``(score, metrics_dict)`` where ``metrics_dict`` contains
        ``profit_factor``, ``max_drawdown``, and ``trade_count``.
        Returns ``(-999.0, metrics_dict)`` when Profit Factor ≤ 1.0.
    """
    proba: np.ndarray = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)
    cost_per_trade: float = (2.0 * fee_rate) + (2.0 * slippage)

    signal_mask = preds == 1
    win_mask = signal_mask & (y_val.values == 1)
    loss_mask = signal_mask & (y_val.values == 0)

    atr = prices_val["atr_14"].values
    close = prices_val["close"].values

    win_returns = (atr[win_mask] * tp_val) / close[win_mask] - cost_per_trade
    loss_returns = -(atr[loss_mask] * sl_val) / close[loss_mask] - cost_per_trade

    gross_profit = float(win_returns.sum()) if win_returns.size > 0 else 0.0
    gross_loss = float(abs(loss_returns.sum())) if loss_returns.size > 0 else 0.0
    trade_count = int(signal_mask.sum())

    pf = gross_profit / max(gross_loss, 1e-9)

    # Maximum Drawdown on the equity curve (chronological order)
    all_returns = np.concatenate([win_returns, loss_returns])
    if all_returns.size > 0:
        trade_indices = np.concatenate([np.where(win_mask)[0], np.where(loss_mask)[0]])
        trade_rets = np.concatenate([win_returns, loss_returns])
        order = np.argsort(trade_indices)
        equity = np.cumsum(trade_rets[order])

        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        peak = np.maximum.accumulate(np.abs(equity))
        peak[peak == 0] = 1e-9
        mdd = float((drawdowns / peak).max())
    else:
        mdd = 0.0

    metrics: dict[str, Any] = {
        "profit_factor": round(pf, 4),
        "max_drawdown": round(mdd, 4),
        "trade_count": trade_count,
    }

    if pf <= 1.0:
        return -999.0, metrics

    frequency_penalty = 1.0 if trade_count >= 5 else 0.5
    score = pf * (1.0 - mdd) * frequency_penalty

    return float(score), metrics


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
        "Pure Momentum": ["rsi_14", "macd", "macd_hist", "stoch_k"],
        "Trend Follower": ["dist_ema_50", "adx_14"],
        "Volatility Hunter": ["atr_14", "bb_width", "bb_pos"],
        "Volume Confirmer": ["obv", "rel_volume"],
        "Momentum + Volatility": ["rsi_14", "macd_hist", "atr_14", "bb_width"],
        "Trend + Volume": ["dist_ema_50", "adx_14", "obv", "rel_volume"],
        "THE FRANKENSTEIN (Technicals)": [
            c for c in df.columns if c not in ["target"] + SENTIMENT_COLS
        ],
    }

    all_strategies: dict[str, list[str]] = {}
    for name, features in base_strategies.items():
        all_strategies[name] = features
        if has_sentiment:
            all_strategies[f"{name} + Sentiment"] = features + SENTIMENT_COLS

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
    prices_val: pd.DataFrame,
    hyperparams: Optional[dict[str, Any]] = None,
) -> tuple[xgb.XGBClassifier, dict[str, Any], np.ndarray, list[Any], float]:
    """Train an XGBClassifier and return metrics evaluated on the validation set.

    Training uses ``early_stopping_rounds=10`` with ``eval_set=[(X_val, y_val)]``
    so the model stops when validation loss stops improving, preventing
    severe overfitting (Task 4).

    Profit/loss per trade is calculated as an **actual return percentage**
    using ``(atr * multiplier) / close`` so that ATR multipliers are never
    treated as direct fee offsets (Task 2).

    Model selection is driven by ``fitness_score`` evaluated exclusively on
    the validation set (Task 3).

    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        y_train: Training target.
        y_val: Validation target.
        y_test: Test target.
        tp_val: ATR multiplier for Take Profit.
        sl_val: ATR multiplier for Stop Loss.
        prices_val: DataFrame aligned to ``X_val`` with ``close`` and
            ``atr_14`` columns — used for realistic return calculations and
            the fitness score.
        hyperparams: Optional XGBoost hyperparameters.

    Returns:
        Tuple ``(model, metrics, preds_test, buy_dates_all, best_threshold)``.
    """
    hp = hyperparams or DEFAULT_HP
    ts = get_trading_settings()
    fee_rate: float = ts["fee_rate"]
    slippage: float = ts["slippage"]

    imbalance = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        scale_pos_weight=imbalance,
        early_stopping_rounds=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_threshold, val_profit = find_optimal_threshold(
        model, X_val, y_val, tp_val, sl_val, prices_val, fee_rate, slippage
    )

    y_probs_val: np.ndarray = model.predict_proba(X_val)[:, 1]
    preds_val = (y_probs_val >= best_threshold).astype(int)

    prec_val: float = precision_score(y_val, preds_val, zero_division=0)
    val_signals = int(sum(preds_val))
    val_hits = int(sum((preds_val == 1) & (y_val == 1)))

    # Compute net profit using correct return-% math (Task 2)
    cost_per_trade: float = (2.0 * fee_rate) + (2.0 * slippage)
    win_mask = (preds_val == 1) & (y_val.values == 1)
    loss_mask = (preds_val == 1) & (y_val.values == 0)
    atr = prices_val["atr_14"].values
    close = prices_val["close"].values
    win_ret = (atr[win_mask] * tp_val) / close[win_mask] - cost_per_trade
    loss_ret = -(atr[loss_mask] * sl_val) / close[loss_mask] - cost_per_trade
    net_profit_val = float(win_ret.sum() + loss_ret.sum())
    gross_profit = float(win_ret.sum()) if win_ret.size > 0 else 0.0
    gross_loss = float(abs(loss_ret.sum())) if loss_ret.size > 0 else 0.0
    profit_factor_val = gross_profit / max(gross_loss, 1e-9)

    score, fit_metrics = fitness_score(
        model, X_val, y_val, prices_val, tp_val, sl_val, fee_rate, slippage, best_threshold
    )

    metrics: dict[str, Any] = {
        "net_profit_pct": round(net_profit_val, 4),
        "profit_factor": round(profit_factor_val, 4),
        "accuracy": round(prec_val * 100, 2),
        "val_signals": val_signals,
        "val_hits": val_hits,
        "opt_threshold": round(best_threshold, 3),
        "fitness_score": round(score, 6),
        "max_drawdown": fit_metrics["max_drawdown"],
        "trade_count": fit_metrics["trade_count"],
    }

    y_probs_test: np.ndarray = model.predict_proba(X_test)[:, 1]
    preds_test = (y_probs_test >= best_threshold).astype(int)

    buy_dates_val = y_val.index[preds_val == 1].tolist()
    buy_dates_test = y_test.index[preds_test == 1].tolist()
    buy_dates_all = buy_dates_val + buy_dates_test

    return model, metrics, preds_test, buy_dates_all, best_threshold
