"""helpers.py — Trading domain utility functions.

Includes data loading, target computation, temporal split with embargo,
strategy construction, and model training/evaluation.
"""

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score

from src.config.settings_loader import get_project_root, get_trading_settings

logger = logging.getLogger(__name__)


# --- CONSTANTS ---
SENTIMENT_COLS: list[str] = ["fng_value", "fng_sma_14", "fng_vol_14"]


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

            label = 0 
            for _, ltf_row in ltf_window.iterrows():
                if ltf_row["low"] <= sl_price.iloc[i]:
                    label = 0
                    break
                if ltf_row["high"] >= tp_price.iloc[i]:
                    label = 1
                    break
            targets.append(label)

        df["target"] = targets
    else:
        df["max_high_future"] = df["high"].rolling(window=swing_days).max().shift(-swing_days)
        df["min_low_future"] = df["low"].rolling(window=swing_days).min().shift(-swing_days)

        df["target"] = (
            (df["max_high_future"] >= tp_price) & (df["min_low_future"] > sl_price)
        ).astype(int)

    return df


def cleanup_columns(df: pd.DataFrame, drop_nan: bool = True) -> pd.DataFrame:
    """Remove OHLCV and auxiliary columns that are not features.

    Args:
        df: DataFrame to clean.
        drop_nan: Whether to drop rows with NaN values.

    Returns:
        DataFrame without auxiliary columns and conditionally without NaN rows.
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
    if drop_nan:
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
    val_size = int(n * val_pct)
    val_end = val_start + val_size
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
    swing_period: int = 5,
) -> tuple[float, float]:
    """Find the threshold that maximises net return on the validation set.

    The profit/loss per trade is computed as an **actual return percentage**
    so that ``tp_val`` / ``sl_val`` (ATR multipliers) are converted to price
    distances before comparing them to ``fee_rate`` and ``slippage``.

    Execution is **sequential**: once a trade is triggered at bar ``i``, the
    next eligible bar is ``i + swing_period`` (cooldown), mirroring the live
    engine which holds one position at a time.

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
        swing_period: Cooldown bars after each trade (default 5).

    Returns:
        Tuple ``(best_threshold, best_profit)`` where ``best_profit`` is the
        total net return across all executable sequential trades.
    """
    proba: np.ndarray = model.predict_proba(X_val)[:, 1]
    cost_per_trade: float = (2.0 * fee_rate) + (2.0 * slippage)
    atr = prices_val["atr_14"].values
    close = prices_val["close"].values
    y_arr = y_val.values
    n = len(proba)

    best_threshold: float = 0.50
    best_profit: float = -np.inf

    min_trades = max(5, int(n / (swing_period + 15)))

    for thresh in np.arange(0.50, 0.85, 0.01):
        profit = 0.0
        trade_count = 0
        i = 0
        while i < n:
            if proba[i] >= thresh:
                if y_arr[i] == 1:
                    profit += (atr[i] * tp_val) / close[i] - cost_per_trade
                else:
                    profit -= (atr[i] * sl_val) / close[i] + cost_per_trade
                trade_count += 1
                i += swing_period
            else:
                i += 1

        if trade_count < min_trades:
            continue

        if profit > best_profit:
            best_profit = profit
            best_threshold = float(thresh)

    if best_profit == -np.inf:
        return -1.0, 0.0

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
    swing_period: int = 5,
) -> tuple[float, dict[str, Any]]:
    """Compute a composite fitness score for model selection on the **validation set**.

    The score balances Profit Factor, Maximum Drawdown, and trade frequency.
    Execution is **sequential**: once a trade fires at bar ``i`` the next
    eligible bar is ``i + swing_period``, matching the live one-position-at-a-time
    constraint.  The frequency penalty is a linear ramp calibrated to the
    validation set length so that smaller datasets are not over-penalised.

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
        swing_period: Cooldown bars after each trade (default 5).

    Returns:
        Tuple ``(score, metrics_dict)`` where ``metrics_dict`` contains
        ``profit_factor``, ``max_drawdown``, ``trade_count``, ``gross_profit``,
        and ``gross_loss``.
        Returns ``(-999.0, metrics_dict)`` when Profit Factor ≤ 1.0.
    """
    proba: np.ndarray = model.predict_proba(X_val)[:, 1]
    cost_per_trade: float = (2.0 * fee_rate) + (2.0 * slippage)
    atr = prices_val["atr_14"].values
    close = prices_val["close"].values
    y_arr = y_val.values
    n = len(proba)

    # --- Sequential simulation (one trade at a time) ---
    trade_returns: list[float] = []
    i = 0
    while i < n:
        if proba[i] >= threshold:
            if y_arr[i] == 1:
                ret = (atr[i] * tp_val) / close[i] - cost_per_trade
            else:
                ret = -((atr[i] * sl_val) / close[i]) - cost_per_trade
            trade_returns.append(ret)
            i += swing_period
        else:
            i += 1

    trade_count = len(trade_returns)
    rets_arr = np.array(trade_returns)

    gross_profit = float(rets_arr[rets_arr > 0].sum()) if trade_count > 0 else 0.0
    gross_loss = float(abs(rets_arr[rets_arr < 0].sum())) if trade_count > 0 else 0.0
    pf = gross_profit / max(gross_loss, 1e-9)

    if trade_count > 0:
        account_equity = np.cumprod(1.0 + rets_arr)
        running_max = np.maximum.accumulate(account_equity)
        drawdowns = (running_max - account_equity) / running_max
        mdd = float(drawdowns.max())
    else:
        mdd = 0.0

    metrics: dict[str, Any] = {
        "profit_factor": round(pf, 4),
        "max_drawdown": round(mdd, 4),
        "trade_count": trade_count,
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
    }

    if pf <= 1.0:
        return -999.0, metrics

    capped_pf = min(pf, 10.0)
    safe_dd = max(mdd, 0.01)

    # Frequency penalty: relative to the theoretical maximum trades
    # achievable given the val set length and swing_period.
    # A config is penalized if it trades less than 15% of its theoretical max.
    # The 0.15 factor represents
    # the minimum acceptable market participation rate.
    max_possible_trades = max(1, len(y_val) / swing_period)
    target_min_trades = max(12, int(max_possible_trades * 0.15))
    frequency_penalty = min(1.0, trade_count / target_min_trades)

    score = ((capped_pf - 1.0) / safe_dd) * math.log(max(trade_count, 1)) * frequency_penalty

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
    prices_test: pd.DataFrame,
    hyperparams: Optional[dict[str, Any]] = None,
    swing_period: int = 5,
) -> tuple[xgb.XGBClassifier, dict[str, Any], np.ndarray, list[Any], float]:
    """Train an XGBClassifier and return metrics evaluated on the validation set.

    Training uses ``early_stopping_rounds=10`` with ``eval_set=[(X_val, y_val)]``
    so the model stops when validation loss stops improving, preventing
    severe overfitting (Task 4).

    Profit/loss per trade is calculated as an **actual return percentage**
    using ``(atr * multiplier) / close`` so that ATR multipliers are never
    treated as direct fee offsets (Task 2).

    Model selection is driven by ``fitness_score`` evaluated exclusively on
    the validation set (Task 3).  Execution is **sequential**: only one trade
    is open at a time; subsequent signals within ``swing_period`` bars are
    skipped.  ``net_profit_pct`` and ``profit_factor`` are derived directly
    from ``fitness_score`` so there is no duplicate calculation.

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
        swing_period: Cooldown bars after each trade (default 5).

    Returns:
        Tuple ``(model, metrics, preds_test, buy_dates_all, best_threshold)``.
    """
    hp = hyperparams
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

    best_threshold, _ = find_optimal_threshold(
        model, X_val, y_val, tp_val, sl_val, prices_val, fee_rate, slippage,
        swing_period=swing_period,
    )

    y_probs_val: np.ndarray = model.predict_proba(X_val)[:, 1]
    preds_val = (y_probs_val >= best_threshold).astype(int)

    score, val_fit_metrics = fitness_score(
        model, X_val, y_val, prices_val, tp_val, sl_val, fee_rate, slippage,
        best_threshold, swing_period=swing_period,
    )

    y_probs_test: np.ndarray = model.predict_proba(X_test)[:, 1]
    preds_test = (y_probs_test >= best_threshold).astype(int)

    test_score, test_fit_metrics = fitness_score(
        model, X_test, y_test, prices_test, tp_val, sl_val, fee_rate, slippage,
        best_threshold, swing_period=swing_period,
    )

    prec_test: float = precision_score(y_test, preds_test, zero_division=0)
    test_signals = int(sum(preds_test))
    test_hits = int(sum((preds_test == 1) & (y_test == 1)))

    net_profit_test = test_fit_metrics["gross_profit"] - test_fit_metrics["gross_loss"]
    profit_factor_test = test_fit_metrics["profit_factor"]
    net_profit_val = val_fit_metrics["gross_profit"] - val_fit_metrics["gross_loss"]

    metrics: dict[str, Any] = {
        # Test-set metrics (out-of-sample reporting only)
        "net_profit_pct": round(net_profit_test, 4),
        "test_profit_factor": round(profit_factor_test, 4),
        "accuracy": round(prec_test * 100, 2),
        "val_fitness_score": round(score, 6),
        "test_fitness_score": round(test_score, 6),
        "test_max_drawdown": test_fit_metrics["max_drawdown"],
        "test_trade_count": test_fit_metrics["trade_count"],
        "test_signals": test_signals,
        "test_hits": test_hits,
        "opt_threshold": round(best_threshold, 3),
        # Validation-set metrics for model selection (no test-set leakage)
        "val_profit_factor": val_fit_metrics["profit_factor"],
        "val_max_drawdown": val_fit_metrics["max_drawdown"],
        "val_trade_count": val_fit_metrics["trade_count"],
        "val_net_profit_pct": round(net_profit_val, 4),
    }

    buy_dates_val = y_val.index[preds_val == 1].tolist()
    buy_dates_test = y_test.index[preds_test == 1].tolist()
    buy_dates_all = buy_dates_val + buy_dates_test

    return model, metrics, preds_test, buy_dates_all, best_threshold
