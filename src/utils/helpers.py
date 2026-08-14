"""helpers.py — Trading domain utility functions.

Includes data loading, target computation, temporal split with embargo,
strategy construction, and model training/evaluation.
"""

import logging
import math
from typing import Any, Callable, Optional

import numpy as np
from numba import njit
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score

from src.config.paths import get_raw_csv_path
from src.config.settings_loader import get_project_root, get_trading_settings
from src.utils.timeframe_utils import parse_timeframe_hours

logger = logging.getLogger(__name__)


# --- CONSTANTS ---
SENTIMENT_COLS: list[str] = ["fng_value", "fng_sma_14", "fng_vol_14"]


@njit
def _resolve_targets_numba(
    daily_times: np.ndarray,
    tp_prices: np.ndarray,
    sl_prices: np.ndarray,
    ltf_times: np.ndarray,
    ltf_highs: np.ndarray,
    ltf_lows: np.ndarray,
    ns_1_day: int,
    ns_swing: int,
    atr: np.ndarray,
    close: np.ndarray,
    tp_multi: float,
    sl_multi: float,
    swing_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba JIT core: Linear scan (Two-Pointer) inside the window.

    Args:
        daily_times: Daily times array.
        tp_prices: Take profit prices array.
        sl_prices: Stop loss prices array.
        ltf_times: Lower timeframe times array.
        ltf_highs: Lower timeframe highs array.
        ltf_lows: Lower timeframe lows array.
        ns_1_day: Number of seconds in 1 day.
        ns_swing: Number of seconds in the swing period.
        atr: ATR values array (same timeframe as daily_times).
        close: Close prices array (same timeframe as daily_times).
        tp_multi: ATR multiplier for Take Profit.
        sl_multi: ATR multiplier for Stop Loss.
        swing_days: Number of future bars (same timeframe) for timeout exit.

    Returns:
        Tuple of (targets, target_rets) — ternary labels and continuous returns.
    """
    n_daily = len(daily_times)
    n_ltf = len(ltf_times)
    targets = np.zeros(n_daily, dtype=np.int32)
    target_rets = np.zeros(n_daily, dtype=np.float64)

    ltf_idx = 0

    for i in range(n_daily):
        window_start = daily_times[i] + ns_1_day
        window_end = daily_times[i] + ns_swing

        while ltf_idx < n_ltf and ltf_times[ltf_idx] < window_start:
            ltf_idx += 1

        j = ltf_idx
        label = 0
        while j < n_ltf and ltf_times[j] <= window_end:
            if ltf_lows[j] <= sl_prices[i]:
                label = -1
                break
            if ltf_highs[j] >= tp_prices[i]:
                label = 1
                break
            j += 1

        targets[i] = label
        if label == 1:
            target_rets[i] = (atr[i] * tp_multi) / close[i]
        elif label == -1:
            target_rets[i] = -(atr[i] * sl_multi) / close[i]
        else:
            exit_idx = min(i + swing_days, n_daily - 1)
            target_rets[i] = (close[exit_idx] - close[i]) / close[i]

    return targets, target_rets


@njit
def _resolve_targets_same_tf(
    tp_prices: np.ndarray,
    sl_prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    close: np.ndarray,
    tp_multi: float,
    sl_multi: float,
    swing_days: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba JIT core for same-timeframe target resolution.

    Scans future bars chronologically (bar-by-bar) for each signal bar i up to
    swing_days bars ahead. The first level touched (TP or SL) determines the target:
      - 1: Take Profit reached first.
      - -1: Stop Loss reached first.
      - 0: Timeout (neither level hit within swing_days bars, or trailing window).

    Tie-break rule: If both SL and TP are touched in the same bar j, SL (lows[j] <= sl)
    is checked first and wins (-1), matching _resolve_targets_numba.

    For rows near the end of the array (i >= n - swing_days), scanning stops at index n - 1.
    If neither level is reached in the remaining available bars, target defaults to 0.

    Args:
        tp_prices: Array of Take Profit price targets.
        sl_prices: Array of Stop Loss price targets.
        highs: Array of High prices.
        lows: Array of Low prices.
        atr: ATR values array.
        close: Close prices array.
        tp_multi: ATR multiplier for Take Profit.
        sl_multi: ATR multiplier for Stop Loss.
        swing_days: Number of future bars to scan.
        n: Total number of bars (len of array).

    Returns:
        Tuple of (targets, target_rets) — ternary labels and continuous returns.
    """
    targets = np.zeros(n, dtype=np.int32)
    target_rets = np.zeros(n, dtype=np.float64)
    for i in range(n):
        label = 0
        max_j = min(i + swing_days + 1, n)
        for j in range(i + 1, max_j):
            if lows[j] <= sl_prices[i]:
                label = -1
                break
            if highs[j] >= tp_prices[i]:
                label = 1
                break
        targets[i] = label
        if label == 1:
            target_rets[i] = (atr[i] * tp_multi) / close[i]
        elif label == -1:
            target_rets[i] = -(atr[i] * sl_multi) / close[i]
        else:
            exit_idx = min(i + swing_days, n - 1)
            target_rets[i] = (close[exit_idx] - close[i]) / close[i]
    return targets, target_rets


@njit
def _compute_threshold_loop(
    proba: np.ndarray,
    y_arr: np.ndarray,
    atr: np.ndarray,
    close: np.ndarray,
    tp_val: float,
    sl_val: float,
    cost_per_trade: float,
    swing_period: int,
    n: int,
    min_trades: int
) -> tuple[float, float]:
    """
    Mathematical core compiled in C for maximum speed.

    Args:
        proba: Probability of the trade.
        y_arr: Target array.
        atr: ATR array.
        close: Close price array.
        tp_val: Take profit value.
        sl_val: Stop loss value.
        cost_per_trade: Cost per trade.
        swing_period: Swing period.
        n: Number of trades.
        min_trades: Minimum trades.

    Returns:
        Tuple of best threshold and best profit.
    """
    best_threshold = 0.50
    best_profit = -np.inf

    thresholds = np.arange(0.50, 0.85, 0.01)

    for thresh in thresholds:
        profit = 0.0
        trade_count = 0
        i = 0
        while i < n:
            if proba[i] >= thresh:
                if y_arr[i] == 1:
                    profit += (atr[i] * tp_val) / close[i] - cost_per_trade
                elif y_arr[i] == -1:
                    profit -= (atr[i] * sl_val) / close[i] + cost_per_trade
                else:
                    # Timeout: Cierre a mercado en la última barra del swing
                    exit_idx = min(i + swing_period, n - 1)
                    profit += (close[exit_idx] - close[i]) / close[i] - cost_per_trade
                
                trade_count += 1
                i += swing_period
            else:
                i += 1

        if trade_count < min_trades:
            continue

        if profit > best_profit:
            best_profit = profit
            best_threshold = thresh

    return best_threshold, best_profit


@njit
def _simulate_fitness_sequential(
    proba: np.ndarray,
    y_arr: np.ndarray,
    atr: np.ndarray,
    close: np.ndarray,
    threshold: float,
    tp_val: float,
    sl_val: float,
    cost_per_trade: float,
    swing_period: int,
    n: int
) -> tuple[int, float, float, float]:
    """Numba JIT core: Simulates trades and calculates the iterative MDD (O(1) memory).

    Args:
        proba: Probability of the trade.
        y_arr: Target array.
        atr: ATR array.
        close: Close price array.
        threshold: Threshold value.
        tp_val: Take profit value.
        sl_val: Stop loss value.
        cost_per_trade: Cost per trade.
        swing_period: Swing period.
        n: Number of trades.

    Returns:
        Tuple of trade count, gross profit, gross loss, and max drawdown.
    """
    trade_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    account_equity = 1.0
    running_max = 1.0
    max_drawdown = 0.0
    
    i = 0
    while i < n:
        if proba[i] >= threshold:
            if y_arr[i] == 1:
                ret = (atr[i] * tp_val) / close[i] - cost_per_trade
                gross_profit += ret
            elif y_arr[i] == -1:
                ret = -((atr[i] * sl_val) / close[i]) - cost_per_trade
                gross_loss += abs(ret)
            else:
                # Timeout
                exit_idx = min(i + swing_period, n - 1)
                ret = (close[exit_idx] - close[i]) / close[i] - cost_per_trade
                if ret > 0:
                    gross_profit += ret
                else:
                    gross_loss += abs(ret)
                
            trade_count += 1
            
            account_equity *= (1.0 + ret)
            if account_equity > running_max:
                running_max = account_equity
            
            drawdown = (running_max - account_equity) / running_max
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
            i += swing_period
        else:
            i += 1
            
    return trade_count, gross_profit, gross_loss, max_drawdown


def _train_core(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    tp_val: float,
    sl_val: float,
    prices_val: pd.DataFrame,
    hyperparams: Optional[dict[str, Any]],
    swing_period: int,
    n_jobs: int = -1,
) -> tuple[xgb.XGBClassifier, float, dict[str, Any]]:
    """Train the model and compute validation-only metrics.

    Shared core between `train_and_evaluate` (full, test-evaluating) and
    `train_and_evaluate_val_only` (fast, val-only). Test set and X_live
    are never touched here.

    Args:
        X_train: Training features.
        X_val: Validation features.
        y_train: Training target.
        y_val: Validation target.
        tp_val: ATR multiplier for Take Profit.
        sl_val: ATR multiplier for Stop Loss.
        prices_val: DataFrame aligned to X_val with close/atr_14.
        hyperparams: XGBoost hyperparameters.
        swing_period: Cooldown bars after each trade.
        n_jobs: Passed to XGBClassifier. Use 1 for reproducible final
            reruns of a winning config (see train_and_evaluate).

    Returns:
        Tuple (model, best_threshold, val_metrics) where val_metrics
        contains val_fitness_score, val_profit_factor, val_max_drawdown,
        val_trade_count, val_net_profit_pct.
    """
    hp = hyperparams
    ts = get_trading_settings()
    fee_rate: float = ts["fee_rate"]
    slippage: float = ts["slippage"]
    
    y_train_xgb = (y_train == 1).astype(int)
    y_val_xgb = (y_val == 1).astype(int)

    imbalance = sum(y_train_xgb == 0) / sum(y_train_xgb == 1) if sum(y_train_xgb == 1) > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        scale_pos_weight=imbalance,
        early_stopping_rounds=10,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train_xgb, eval_set=[(X_val, y_val_xgb)], verbose=False)

    best_threshold, _ = find_optimal_threshold(
        model, X_val, y_val, tp_val, sl_val, prices_val, fee_rate, slippage,
        swing_period=swing_period,
    )

    if best_threshold == -1.0:
        return model, best_threshold, {}

    score, val_fit_metrics = fitness_score(
        model, X_val, y_val, prices_val, tp_val, sl_val, fee_rate, slippage,
        best_threshold, swing_period=swing_period,
    )

    net_profit_val = val_fit_metrics["gross_profit"] - val_fit_metrics["gross_loss"]

    val_metrics: dict[str, Any] = {
        "val_fitness_score": round(score, 6),
        "val_profit_factor": val_fit_metrics["profit_factor"],
        "val_max_drawdown": val_fit_metrics["max_drawdown"],
        "val_trade_count": val_fit_metrics["trade_count"],
        "val_net_profit_pct": round(net_profit_val, 4),
    }

    return model, best_threshold, val_metrics

# --- DATA LOADING ---
def load_csv_data(symbol: str, timeframe: str = "1d") -> pd.DataFrame:
    """Load an OHLCV CSV and normalize columns / time index.

    Uses the per-symbol directory layout::

        data/raw_csv/{symbol}/{timeframe}.csv

    For daily (``1d``) and coarser timeframes, timestamps are truncated
    to midnight via ``.dt.normalize()`` for consistent joining with
    daily sentiment data.  For sub-daily timeframes (``4h``, ``1h``,
    etc.) the full timestamp is preserved so that distinct intraday
    candles are not collapsed onto the same index value.

    Args:
        symbol: Trading pair in safe format (e.g. ``'BTC_USDT'``).
        timeframe: Candle interval (e.g. ``'1d'``, ``'4h'``).

    Returns:
        DataFrame with a ``DatetimeIndex``.

    Raises:
        FileNotFoundError: If the file does not exist. The message names
            the expected path and the ``data_fetcher`` command to run to
            fetch it, since a symbol/timeframe can be configured (e.g. in
            ``bot_state.json``) before the corresponding CSV has actually
            been downloaded.
    """
    filepath = get_raw_csv_path(symbol, timeframe)
    if not filepath.exists():
        raise FileNotFoundError(
            f"No data found for {symbol} at timeframe {timeframe} "
            f"(expected {filepath}). Run "
            f"`python3 -m src.brain.data_fetcher {symbol} --timeframe {timeframe}` "
            f"first."
        )

    df = pd.read_csv(filepath)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Only normalize (truncate to midnight) for daily or coarser bars.
    # Sub-daily timeframes need the full timestamp to keep candles unique.
    tf_hours = parse_timeframe_hours(timeframe)
    if tf_hours >= 24.0:
        df["timestamp"] = df["timestamp"].dt.normalize()
    df.set_index("timestamp", inplace=True)
    return df


# --- TARGET ---
def compute_target(
    df: pd.DataFrame,
    swing_days: int,
    atr_tp_multi: float = 0.0,
    atr_sl_multi: float = 0.0,
    lower_tf_df: Optional[pd.DataFrame] = None,
    timeframe_hours: float = 24.0,
) -> pd.DataFrame:
    """Compute the ``target`` column using dynamic ATR-based TP/SL.

    Args:
        df: DataFrame with OHLCV columns and ``atr_14``.  Index must be a
            timezone-naive ``DatetimeIndex``.
        swing_days: Number of future **bars** to evaluate TP/SL.  Despite
            the legacy name (kept for config compatibility), this is
            interpreted as a bar count — the real-world duration is
            ``swing_days * timeframe_hours`` hours.
        atr_tp_multi: ATR multiplier for Take Profit.
        atr_sl_multi: ATR multiplier for Stop Loss.
        lower_tf_df: Optional lower-timeframe DataFrame (e.g. 1 h or 15 m)
            with a ``DatetimeIndex`` and ``high`` / ``low`` columns.  When
            provided it is used **exclusively** to determine which level —
            Take Profit or Stop Loss — is reached first within the
            ``swing_days`` window.  Its columns are **never** added to ``df``
            as XGBoost features.  When ``None`` the function falls back to the
            vectorized approximation so the existing data pipeline keeps working.
        timeframe_hours: Duration of one bar in hours (e.g. ``24.0`` for
            daily, ``4.0`` for 4 h).  Controls the lookahead window in
            both the Numba and the vectorized fallback path.  Defaults to
            ``24.0`` for backward compatibility with the pre-existing daily
            pipeline.

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
    atr_arr: np.ndarray = df["atr_14"].values
    close_arr: np.ndarray = df["close"].values
    swing_days_int = int(swing_days)

    if lower_tf_df is not None:
        # --- Multi-timeframe path (Numba Optimized) -------------------------
        missing = {"high", "low"} - set(lower_tf_df.columns)
        if missing:
            raise ValueError(
                f"lower_tf_df is missing required columns: {missing}"
            )

        daily_times = df.index.to_numpy().astype("datetime64[ns]").astype(np.int64)
        ltf_times = lower_tf_df.index.to_numpy().astype("datetime64[ns]").astype(np.int64)
        
        tp_prices_arr = tp_price.values
        sl_prices_arr = sl_price.values
        ltf_highs = lower_tf_df["high"].values
        ltf_lows = lower_tf_df["low"].values
        
        # The window starts one bar after the signal bar and extends
        # swing_days bars into the future.  For daily data this is
        # identical to the original pd.Timedelta(days=...) logic; for
        # sub-daily data it correctly scales the window.
        ns_1_bar = int(pd.Timedelta(hours=timeframe_hours).value)
        ns_swing = int(pd.Timedelta(hours=timeframe_hours * swing_days).value)

        targets, target_rets = _resolve_targets_numba(
            daily_times=daily_times,
            tp_prices=tp_prices_arr,
            sl_prices=sl_prices_arr,
            ltf_times=ltf_times,
            ltf_highs=ltf_highs,
            ltf_lows=ltf_lows,
            ns_1_day=ns_1_bar,
            ns_swing=ns_swing,
            atr=atr_arr,
            close=close_arr,
            tp_multi=float(atr_tp_multi),
            sl_multi=float(atr_sl_multi),
            swing_days=swing_days_int,
        )
        
        df["target"] = targets
        df["target_ret"] = target_rets

    else:
        # --- Same-Timeframe Fallback (Numba Bar-by-Bar Chronological) -------
        tp_prices_arr = tp_price.values
        sl_prices_arr = sl_price.values
        highs_arr = df["high"].values
        lows_arr = df["low"].values
        n = len(df)

        targets, target_rets = _resolve_targets_same_tf(
            tp_prices=tp_prices_arr,
            sl_prices=sl_prices_arr,
            highs=highs_arr,
            lows=lows_arr,
            atr=atr_arr,
            close=close_arr,
            tp_multi=float(atr_tp_multi),
            sl_multi=float(atr_sl_multi),
            swing_days=swing_days_int,
            n=n,
        )

        df["target"] = targets
        df["target_ret"] = target_rets

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
        Returns ``(-1.0, 0.0)`` if no valid threshold meets minimum trade criteria.
    """
    proba: np.ndarray = model.predict_proba(X_val)[:, 1]
    cost_per_trade: float = (2.0 * fee_rate) + (2.0 * slippage)
    atr: np.ndarray = prices_val["atr_14"].values
    close: np.ndarray = prices_val["close"].values
    y_arr: np.ndarray = y_val.values
    n: int = len(proba)

    max_possible_trades = max(1.0, n / swing_period)
    min_trades = max(10, int(max_possible_trades * 0.15))

    best_threshold, best_profit = _compute_threshold_loop(
        proba=proba,
        y_arr=y_arr,
        atr=atr,
        close=close,
        tp_val=float(tp_val),
        sl_val=float(sl_val),
        cost_per_trade=float(cost_per_trade),
        swing_period=int(swing_period),
        n=int(n),
        min_trades=int(min_trades)
    )

    if best_profit == -np.inf:
        return -1.0, 0.0

    return float(best_threshold), float(best_profit)


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
    atr: np.ndarray = prices_val["atr_14"].values
    close: np.ndarray = prices_val["close"].values
    y_arr: np.ndarray = y_val.values
    n: int = len(proba)

    trade_count, gross_profit, gross_loss, mdd = _simulate_fitness_sequential(
        proba=proba,
        y_arr=y_arr,
        atr=atr,
        close=close,
        threshold=float(threshold),
        tp_val=float(tp_val),
        sl_val=float(sl_val),
        cost_per_trade=float(cost_per_trade),
        swing_period=int(swing_period),
        n=int(n)
    )

    pf = gross_profit / max(gross_loss, 1e-9)

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
    # The 0.15 factor represents the minimum acceptable market participation rate.
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
def train_and_evaluate_val_only(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    tp_val: float,
    sl_val: float,
    prices_val: pd.DataFrame,
    hyperparams: Optional[dict[str, Any]] = None,
    swing_period: int = 5,
) -> tuple[xgb.XGBClassifier, dict[str, Any], float]:
    """Fast path: train and evaluate on validation only. Never touches
    the test set or X_live. Use this inside the grid-search inner loop;
    reserve `train_and_evaluate` for the winning config.

    Returns:
        Tuple (model, val_metrics, best_threshold). `best_threshold` is
        -1.0 when no threshold met the minimum trade count — callers
        should `continue` in that case rather than scoring the config.
    """
    model, best_threshold, val_metrics = _train_core(
        X_train, X_val, y_train, y_val, tp_val, sl_val, prices_val,
        hyperparams, swing_period, n_jobs=-1,
    )
    return model, val_metrics, best_threshold


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
    n_jobs: int = -1,
) -> tuple[xgb.XGBClassifier, dict[str, Any], np.ndarray, list[Any], float]:
    """Full path: train, then evaluate on both validation and test, plus
    X_live-ready predictions. Reserved for audit mode or for the single
    winning config after grid search.

    `n_jobs` defaults to -1 for backward compatibility, but callers doing
    a reproducible final rerun of a winning config (after fast-path
    selection) should pass `n_jobs=1` — with tree_method='hist' and
    n_jobs=-1, floating-point aggregation order across threads is not
    guaranteed identical between runs even with a fixed random_state.
    """
    model, best_threshold, val_metrics = _train_core(
        X_train, X_val, y_train, y_val, tp_val, sl_val, prices_val,
        hyperparams, swing_period, n_jobs=n_jobs,
    )

    if best_threshold == -1.0:
        empty_metrics: dict[str, Any] = {
            "net_profit_pct": 0.0, "test_profit_factor": 0.0, "accuracy": 0.0,
            "val_fitness_score": -999.0, "test_fitness_score": -999.0,
            "test_max_drawdown": 0.0, "test_trade_count": 0,
            "test_signals": 0, "test_hits": 0, "opt_threshold": -1.0,
            "val_profit_factor": 0.0, "val_max_drawdown": 1.0,
            "val_trade_count": 0, "val_net_profit_pct": 0.0,
        }
        return model, empty_metrics, np.array([]), [], -1.0

    ts = get_trading_settings()
    fee_rate: float = ts["fee_rate"]
    slippage: float = ts["slippage"]

    y_probs_val: np.ndarray = model.predict_proba(X_val)[:, 1]
    preds_val = (y_probs_val >= best_threshold).astype(int)

    y_probs_test: np.ndarray = model.predict_proba(X_test)[:, 1]
    preds_test = (y_probs_test >= best_threshold).astype(int)

    test_score, test_fit_metrics = fitness_score(
        model, X_test, y_test, prices_test, tp_val, sl_val, fee_rate, slippage,
        best_threshold, swing_period=swing_period,
    )
    
    y_test_xgb = (y_test == 1).astype(int)
    prec_test: float = precision_score(y_test_xgb, preds_test, zero_division=0)
    test_signals = int(sum(preds_test))
    test_hits = int(sum((preds_test == 1) & (y_test_xgb == 1)))

    net_profit_test = test_fit_metrics["gross_profit"] - test_fit_metrics["gross_loss"]

    metrics: dict[str, Any] = {
        "net_profit_pct": round(net_profit_test, 4),
        "test_profit_factor": round(test_fit_metrics["profit_factor"], 4),
        "accuracy": round(prec_test * 100, 2),
        "test_fitness_score": round(test_score, 6),
        "test_max_drawdown": test_fit_metrics["max_drawdown"],
        "test_trade_count": test_fit_metrics["trade_count"],
        "test_signals": test_signals,
        "test_hits": test_hits,
        "opt_threshold": round(best_threshold, 3),
        **val_metrics,
    }

    buy_dates_val = y_val.index[preds_val == 1].tolist()
    buy_dates_test = y_test.index[preds_test == 1].tolist()
    buy_dates_all = buy_dates_val + buy_dates_test

    return model, metrics, preds_test, buy_dates_all, best_threshold


# --- MODEL TRAINING FACTORIES FOR WALK-FORWARD VALIDATION ---
# Recommended threshold_grid for each factory:
#   - Binary homerun: sigmoid output, standard (0.50, 0.85, 0.01)
#   - Multiclass 3: softmax 3-way split, TP class rarely exceeds ~0.39
BINARY_HOMERUN_THRESHOLD_GRID: tuple[float, float, float] = (0.50, 0.85, 0.01)
MULTICLASS_3_THRESHOLD_GRID: tuple[float, float, float] = (0.25, 0.70, 0.01)


def train_predict_binary_homerun(
    X_train: pd.DataFrame,
    y_train_raw: pd.Series,
    X_val: pd.DataFrame,
    y_val_raw: pd.Series,
    **kwargs: Any,
) -> tuple[xgb.XGBClassifier, np.ndarray, Callable[[xgb.XGBClassifier, pd.DataFrame], np.ndarray]]:
    """Train a binary XGBoost model isolating Take Profit against all other outcomes.

    This formulation treats the problem as a "home run" binary classification,
    where hitting the Take Profit is class 1, and hitting the Stop Loss or
    timing out is class 0. It handles class imbalance using `scale_pos_weight`.

    Recommended ``threshold_grid`` for run_walk_forward:
        ``BINARY_HOMERUN_THRESHOLD_GRID`` = (0.50, 0.85, 0.01)

    Args:
        X_train: Training features DataFrame.
        y_train_raw: Training ternary targets Series (-1, 0, 1).
        X_val: Validation features DataFrame.
        y_val_raw: Validation ternary targets Series (-1, 0, 1).
        **kwargs: Additional keyword arguments (e.g., random_state).

    Returns:
        A tuple containing:
            - The fitted XGBClassifier.
            - An array of validation probabilities for class 1 (Take Profit).
            - A prediction closure to be used for OOS inference.
    """
    y_tr = (y_train_raw == 1).astype(int)
    y_va = (y_val_raw == 1).astype(int)

    n_pos = max(1, int((y_tr == 1).sum()))
    spw = int((y_tr == 0).sum()) / n_pos

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="binary:logistic",
        scale_pos_weight=spw,
        early_stopping_rounds=10,
        eval_metric="logloss",
        tree_method="hist",
        random_state=kwargs.get("random_state", 42),
        n_jobs=-1,
    )

    model.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)

    def predict_fn(mod: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
        return mod.predict_proba(X)[:, 1]

    return model, predict_fn(model, X_val), predict_fn


train_predict_binary_homerun.threshold_grid = BINARY_HOMERUN_THRESHOLD_GRID  # type: ignore[attr-defined]


def train_predict_multiclass_3(
    X_train: pd.DataFrame,
    y_train_raw: pd.Series,
    X_val: pd.DataFrame,
    y_val_raw: pd.Series,
    **kwargs: Any,
) -> tuple[xgb.XGBClassifier, np.ndarray, Callable[[xgb.XGBClassifier, pd.DataFrame], np.ndarray]]:
    """Train a 3-class XGBoost model mapping outcomes to Stop Loss, Timeout, and Take Profit.

    This formulation explicitly models the three possible outcomes of a trade
    using a `multi:softprob` objective. It balances classes using individual
    sample weights based on class frequency.

    Recommended ``threshold_grid`` for run_walk_forward:
        ``MULTICLASS_3_THRESHOLD_GRID`` = (0.25, 0.70, 0.01)
    The softmax of 3 classes rarely exceeds ~0.39 for the TP class, so the
    default binary grid (0.50, 0.85, 0.01) will yield zero trades at every
    threshold and silently fail with ``threshold_failed``.

    Args:
        X_train: Training features DataFrame.
        y_train_raw: Training ternary targets Series (-1, 0, 1).
        X_val: Validation features DataFrame.
        y_val_raw: Validation ternary targets Series (-1, 0, 1).
        **kwargs: Additional keyword arguments (e.g., random_state).

    Returns:
        A tuple containing:
            - The fitted XGBClassifier.
            - An array of validation probabilities for class 2 (Take Profit).
            - A prediction closure to be used for OOS inference.
    """
    raw_to_cls = {-1: 0, 0: 1, 1: 2}
    y_tr = y_train_raw.map(raw_to_cls).astype(int).values
    y_va = y_val_raw.map(raw_to_cls).astype(int).values

    classes, counts = np.unique(y_tr, return_counts=True)
    w_dict = {
        int(c): len(y_tr) / (len(classes) * max(1, cnt))
        for c, cnt in zip(classes, counts)
    }

    w_tr = np.array([w_dict[int(c)] for c in y_tr], dtype=np.float64)
    w_va = np.array([w_dict[int(c)] for c in y_va], dtype=np.float64)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        early_stopping_rounds=10,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=kwargs.get("random_state", 42),
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_va)],
        sample_weight_eval_set=[w_va],
        verbose=False,
    )

    def predict_fn(mod: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
        return mod.predict_proba(X)[:, 2]

    return model, predict_fn(model, X_val), predict_fn


train_predict_multiclass_3.threshold_grid = MULTICLASS_3_THRESHOLD_GRID  # type: ignore[attr-defined]


REGRESSION_RETURN_THRESHOLD_GRID: tuple[float, float, float] = (-0.0035, 0.0070, 0.0003)


def train_predict_regression_return(
    X_train: pd.DataFrame,
    y_train_raw: pd.Series,
    X_val: pd.DataFrame,
    y_val_raw: pd.Series,
    **kwargs: Any,
) -> tuple[Any, np.ndarray, Callable[[Any, pd.DataFrame], np.ndarray]]:
    """Train an XGBoost regression model to predict continuous trade returns.

    This formulation replaces the ternary classification (SL/Timeout/TP)
    with a regression that directly predicts the continuous realized return
    ``target_ret = (exit_price - entry_price) / entry_price``.  It preserves
    gradient information from the timeout bucket (13–15% of trades) where
    the economic outcome is continuous but was previously collapsed into a
    single class.

    The prediction function returns the raw predicted return directly
    (no sigmoid / softmax) so the threshold grid should be expressed in
    return units (e.g. 0.0 = cost-neutral, 0.005 = +50 bps net of cost).

    Args:
        X_train: Training features DataFrame.
        y_train_raw: Training continuous-return targets (target_ret column).
        X_val: Validation features DataFrame.
        y_val_raw: Validation continuous-return targets (target_ret column).
        **kwargs: Additional keyword arguments (e.g., random_state).

    Returns:
        A tuple containing:
            - The fitted XGBRegressor.
            - An array of validation predictions (predicted returns, raw values).
            - A prediction closure to be used for OOS inference.
    """
    y_tr = y_train_raw.astype(np.float64).values
    y_va = y_val_raw.astype(np.float64).values

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:pseudohubererror",
        early_stopping_rounds=10,
        eval_metric="mphe",
        tree_method="hist",
        random_state=kwargs.get("random_state", 42),
        n_jobs=-1,
    )

    model.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)

    def predict_fn(mod, X: pd.DataFrame) -> np.ndarray:
        return mod.predict(X).astype(np.float64)

    return model, predict_fn(model, X_val), predict_fn


train_predict_regression_return.threshold_grid = REGRESSION_RETURN_THRESHOLD_GRID  # type: ignore[attr-defined]