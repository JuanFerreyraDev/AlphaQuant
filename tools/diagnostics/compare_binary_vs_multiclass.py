"""Compare two REAL model formulations on ternary targets + ternary sim.

Clarifies a prior table ambiguity:
  The rows labeled "NEW_target+NEW_sim" vs "NEW+binarized_y" were NOT
  two different model formulations. Both trained the SAME binary
  XGBClassifier with y=(target==1). The only difference was whether
  simulation/threshold used ternary labels (-1/0/1) or a binarized copy
  that collapses SL into timeout. That is an evaluation-encoding issue,
  not multi:softprob vs binary.

This script compares the two intentional design choices:

  A) binary_homerun
       XGBClassifier binary (objective binary:logistic)
       y_xgb = (target == 1).astype(int)   # TP vs {SL, timeout}
       imbalance: scale_pos_weight = n_neg / n_pos
       signal proba = P(class=1) = P(TP)

  B) multiclass_3
       XGBClassifier objective multi:softprob, num_class=3
       labels remapped {-1,0,1} -> {0,1,2} = {SL, timeout, TP}
       imbalance: per-sample balanced weights
         w_c = n / (n_classes * count_c)
       signal proba = P(TP) from the 3-way softmax

Both are scored with the SAME ternary payoff simulation
(TP / SL / timeout market-close). Threshold search maximises net return
on P(TP) >= thr, identical grid to find_optimal_threshold.

Also reports bootstrap OOS deltas on oos_val_2024H2 and oos_test_2025
(model trained only on bars prior to each window).

Run:  .venv/bin/python tools/diagnostics/compare_binary_vs_multiclass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brain.features import compute_all_technicals
from src.utils.data_splits import (
    compute_dynamic_split,
    compute_split_boundaries,
    get_calibrated_constants,
)
from src.utils.helpers import cleanup_columns, compute_target, load_csv_data

SYMBOL = "BTC_USDT"
SWING = 10
TP = 1.5
SL = 1.0
COST = 0.0
FEATURES = ["atr_14", "bb_width", "bb_pos"]
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)

# Multiclass label map: raw target -> xgb class index
#   SL=-1 -> 0, timeout=0 -> 1, TP=1 -> 2
CLS_SL, CLS_TO, CLS_TP = 0, 1, 2
RAW_TO_CLS = {-1: CLS_SL, 0: CLS_TO, 1: CLS_TP}

OOS_WINDOWS = {
    "oos_val_2024H2": ("2024-08-14", "2025-08-10"),
    "oos_test_2025": ("2025-08-12", "2026-08-07"),
}

TIMEFRAMES = {
    "4h": 4.0,
    "1h": 1.0,
}


def to_multiclass(y: pd.Series) -> np.ndarray:
    return y.map(RAW_TO_CLS).astype(int).values


def balanced_sample_weights(y_cls: np.ndarray) -> np.ndarray:
    """sklearn-style balanced weights: n / (n_classes * count_c)."""
    n = len(y_cls)
    classes, counts = np.unique(y_cls, return_counts=True)
    n_classes = len(classes)
    weight_of = {int(c): n / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_of[int(c)] for c in y_cls], dtype=np.float64)


def per_trade_returns(
    prices: pd.DataFrame,
    y: pd.Series,
    proba_tp: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Ternary-aware sequential returns (matches _simulate_fitness_sequential)."""
    atr = prices["atr_14"].values
    close = prices["close"].values
    y_arr = y.values.astype(np.float64)
    n = len(proba_tp)
    rets: list[float] = []
    i = 0
    while i < n:
        if proba_tp[i] >= threshold:
            if y_arr[i] == 1:
                rets.append((atr[i] * TP) / close[i] - COST)
            elif y_arr[i] == -1:
                rets.append(-((atr[i] * SL) / close[i]) - COST)
            else:
                exit_idx = min(i + SWING, n - 1)
                rets.append((close[exit_idx] - close[i]) / close[i] - COST)
            i += SWING
        else:
            i += 1
    return np.asarray(rets, dtype=np.float64)


def profit_factor(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return float("nan")
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    return float(gains / max(losses, 1e-9))


def bootstrap_delta(
    model_rets: np.ndarray, naive_rets: np.ndarray, n: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    deltas = np.empty(n)
    for i in range(n):
        m = RNG.choice(model_rets, size=len(model_rets), replace=True)
        nv = RNG.choice(naive_rets, size=len(naive_rets), replace=True)
        deltas[i] = profit_factor(m) - profit_factor(nv)
    return (
        float(np.percentile(deltas, 5)),
        float(np.percentile(deltas, 50)),
        float(np.percentile(deltas, 95)),
    )


def threshold_search(
    proba_tp: np.ndarray,
    y: pd.Series,
    prices: pd.DataFrame,
    *,
    formulation: str,
) -> float:
    """Economic threshold search on P(TP); min_trades matches production.

    Binary: grid 0.50–0.85 (same as find_optimal_threshold) — calibrated
    for a 2-class probability.

    Multiclass: grid 0.25–0.70. Under a 3-way softmax, P(TP) is typically
    well below 0.50 even when TP is the modal class, so the binary floor
    of 0.50 rejects every candidate (thr=-1). Lower bound 0.25 is just
    under the uniform prior 1/3.
    """
    n = len(proba_tp)
    max_possible = max(1.0, n / SWING)
    min_trades = max(10, int(max_possible * 0.15))
    if formulation == "binary_homerun":
        grid = np.arange(0.50, 0.86, 0.01)
    else:
        grid = np.arange(0.25, 0.71, 0.01)
    best_thr, best_net = -1.0, -1e18
    for thr in grid:
        rets = per_trade_returns(prices, y, proba_tp, float(thr))
        if len(rets) < min_trades:
            continue
        net = float(rets.sum())
        if net > best_net:
            best_net = net
            best_thr = float(thr)
    return best_thr


def train_binary(X_train, y_train_raw, X_val, y_val_raw):
    y_tr = (y_train_raw == 1).astype(int)
    y_va = (y_val_raw == 1).astype(int)
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="binary:logistic",
        scale_pos_weight=spw,
        early_stopping_rounds=10,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
    bal_info = {
        "kind": "scale_pos_weight",
        "scale_pos_weight": spw,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "class_counts_raw": {
            "-1": int((y_train_raw == -1).sum()),
            "0": int((y_train_raw == 0).sum()),
            "1": int((y_train_raw == 1).sum()),
        },
    }
    return model, bal_info


def train_multiclass(X_train, y_train_raw, X_val, y_val_raw):
    y_tr = to_multiclass(y_train_raw)
    y_va = to_multiclass(y_val_raw)
    w_tr = balanced_sample_weights(y_tr)
    w_va = balanced_sample_weights(y_va)
    # Class counts / weights for reporting
    classes, counts = np.unique(y_tr, return_counts=True)
    weight_of = {
        int(c): len(y_tr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)
    }
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        early_stopping_rounds=10,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
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
    bal_info = {
        "kind": "balanced_sample_weight",
        "class_weights": {f"cls{c}": weight_of[int(c)] for c in classes},
        "class_counts_xgb": {f"cls{c}": int(cnt) for c, cnt in zip(classes, counts)},
        "class_counts_raw": {
            "-1": int((y_train_raw == -1).sum()),
            "0": int((y_train_raw == 0).sum()),
            "1": int((y_train_raw == 1).sum()),
        },
        "note": "cls0=SL, cls1=timeout, cls2=TP; proba_signal=P(cls2)",
    }
    return model, bal_info


def predict_tp_proba(model, X, formulation: str) -> np.ndarray:
    proba = model.predict_proba(X)
    if formulation == "binary_homerun":
        return proba[:, 1]
    # multiclass: column order follows sorted class labels 0,1,2
    return proba[:, CLS_TP]


def score_window(proba_tp, y, prices, threshold):
    model_rets = per_trade_returns(prices, y, proba_tp, threshold)
    naive_rets = per_trade_returns(prices, y, np.ones(len(y)), 0.0)
    return {
        "model_pf": profit_factor(model_rets),
        "naive_pf": profit_factor(naive_rets),
        "delta": profit_factor(model_rets) - profit_factor(naive_rets),
        "model_tc": int(len(model_rets)),
        "naive_tc": int(len(naive_rets)),
        "model_rets": model_rets,
        "naive_rets": naive_rets,
    }


def prepare_frame(timeframe: str, hours: float):
    df = load_csv_data(SYMBOL, timeframe)
    compute_all_technicals(df)
    # Skip FNG — Volatility Hunter features do not use sentiment.
    compute_target(
        df, swing_days=SWING, atr_tp_multi=TP, atr_sl_multi=SL, timeframe_hours=hours
    )
    prices = df[["close", "atr_14"]].copy()
    df_model = df.copy()
    cleanup_columns(df_model)
    prices = prices.loc[df_model.index]

    cal = get_calibrated_constants(timeframe)
    split = compute_dynamic_split(
        n_bars=len(df_model),
        swing_period=SWING,
        embargo_days=SWING,
        bars_per_trade_safety_factor=cal["bars_per_trade_safety_factor"],
        min_val_trades=cal["stat_floor_val_trades"],
        min_test_trades=cal["stat_floor_test_trades"],
        max_val_test_share=cal["max_val_test_share"],
    )
    assert split is not None
    n_train, n_val, n_test = split
    train_sl, val_sl, test_sl = compute_split_boundaries(
        n_train, n_val, n_test, embargo_days=SWING
    )
    parts = {
        "train": df_model.iloc[train_sl],
        "val": df_model.iloc[val_sl],
        "test": df_model.iloc[test_sl],
        "all": df_model,
        "prices": prices,
        "split": split,
    }
    return parts


def run_production_split(parts, formulation: str):
    df_tr, df_va, df_te = parts["train"], parts["val"], parts["test"]
    prices = parts["prices"]
    X_tr, X_va, X_te = df_tr[FEATURES], df_va[FEATURES], df_te[FEATURES]
    y_tr, y_va, y_te = df_tr["target"], df_va["target"], df_te["target"]
    p_va, p_te = prices.loc[df_va.index], prices.loc[df_te.index]

    if formulation == "binary_homerun":
        model, bal = train_binary(X_tr, y_tr, X_va, y_va)
    else:
        model, bal = train_multiclass(X_tr, y_tr, X_va, y_va)

    proba_va = predict_tp_proba(model, X_va, formulation)
    thr = threshold_search(proba_va, y_va, p_va, formulation=formulation)
    results = {
        "threshold": thr,
        "balance": bal,
        "formulation": formulation,
        "proba_tp_val_quantiles": {
            "p10": float(np.quantile(proba_va, 0.10)),
            "p50": float(np.quantile(proba_va, 0.50)),
            "p90": float(np.quantile(proba_va, 0.90)),
            "max": float(proba_va.max()),
        },
    }
    if thr < 0:
        results["val"] = None
        results["test"] = None
        return results, model

    results["val"] = score_window(proba_va, y_va, p_va, thr)
    proba_te = predict_tp_proba(model, X_te, formulation)
    results["test"] = score_window(proba_te, y_te, p_te, thr)
    return results, model


def run_oos_bootstrap(parts, formulation: str, timeframe: str):
    df_all = parts["all"]
    prices = parts["prices"]
    rows = []
    for name, (start, end) in OOS_WINDOWS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (df_all.index >= start_ts) & (df_all.index <= end_ts)
        df_w = df_all.loc[mask]
        if len(df_w) < 200:
            rows.append({"window": name, "status": "insufficient_window"})
            continue
        df_prior = df_all.loc[df_all.index < start_ts]
        if len(df_prior) < 500:
            rows.append({"window": name, "status": "insufficient_prior"})
            continue
        n_prior = len(df_prior)
        n_val_prior = max(200, int(n_prior * 0.15))
        df_prior_train = df_prior.iloc[:-n_val_prior]
        df_prior_val = df_prior.iloc[-n_val_prior:]
        p_prior_val = prices.loc[df_prior_val.index]
        p_w = prices.loc[df_w.index]

        if formulation == "binary_homerun":
            model, bal = train_binary(
                df_prior_train[FEATURES],
                df_prior_train["target"],
                df_prior_val[FEATURES],
                df_prior_val["target"],
            )
        else:
            model, bal = train_multiclass(
                df_prior_train[FEATURES],
                df_prior_train["target"],
                df_prior_val[FEATURES],
                df_prior_val["target"],
            )

        proba_val = predict_tp_proba(model, df_prior_val[FEATURES], formulation)
        thr = threshold_search(
            proba_val, df_prior_val["target"], p_prior_val, formulation=formulation
        )
        if thr < 0:
            rows.append({"window": name, "status": "threshold_failed", "balance": bal})
            continue

        proba_w = predict_tp_proba(model, df_w[FEATURES], formulation)
        scored = score_window(proba_w, df_w["target"], p_w, thr)
        boot = {"p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
        if scored["model_tc"] >= 20 and scored["naive_tc"] >= 20:
            p5, p50, p95 = bootstrap_delta(scored["model_rets"], scored["naive_rets"])
            boot = {"p5": p5, "p50": p50, "p95": p95}
        rows.append(
            {
                "window": name,
                "status": "ok",
                "timeframe": timeframe,
                "formulation": formulation,
                "threshold": thr,
                "model_pf": scored["model_pf"],
                "naive_pf": scored["naive_pf"],
                "delta": scored["delta"],
                "model_tc": scored["model_tc"],
                "naive_tc": scored["naive_tc"],
                **boot,
            }
        )
    return rows


def fmt_row(formulation, split, r, thr, bal_short):
    if r is None:
        return (
            f"{formulation:<18} {split:<5} {'—':>8} {'—':>8} {'—':>8} "
            f"{'—':>8} {'—':>8} thr={thr}  {bal_short}"
        )
    return (
        f"{formulation:<18} {split:<5} "
        f"{r['naive_pf']:8.4f} {r['model_pf']:8.4f} {r['delta']:+8.4f} "
        f"{r['naive_tc']:8d} {r['model_tc']:8d} thr={thr:.2f}  {bal_short}"
    )


def main() -> None:
    print("=" * 88)
    print("CLARIFICATION (prior table)")
    print("=" * 88)
    print(
        "Neither prior 'NEW_*' row trained multi:softprob.\n"
        "Both trained binary XGB with y=(target==1) and scale_pos_weight.\n"
        "Difference was ONLY evaluation y encoding:\n"
        "  - ternary y in sim  => correct SL/timeout/TP payoffs\n"
        "  - binarized y in sim => SL(-1) collapsed to 0 => scored as timeout\n"
        "That second path is an evaluation bug relative to the ternary payoff\n"
        "engine, NOT a distinct intentional model formulation.\n"
    )
    print(
        "This run compares the two intentional formulations:\n"
        "  binary_homerun  = binary:logistic + scale_pos_weight + P(TP)\n"
        "  multiclass_3    = multi:softprob + balanced sample_weight + P(TP)\n"
        "Both scored with ternary simulation.\n"
    )

    for tf, hours in TIMEFRAMES.items():
        print("=" * 88)
        print(f"TIMEFRAME {tf}  swing={SWING} tp={TP} sl={SL} cost={COST}")
        print("=" * 88)
        parts = prepare_frame(tf, hours)
        print(f"split train/val/test = {parts['split']}")
        print(
            f"val  [{parts['val'].index[0]} .. {parts['val'].index[-1]}]  "
            f"n={len(parts['val'])}"
        )
        print(
            f"test [{parts['test'].index[0]} .. {parts['test'].index[-1]}]  "
            f"n={len(parts['test'])}"
        )
        print()

        header = (
            f"{'formulation':<18} {'split':<5} {'naive_pf':>8} {'model_pf':>8} "
            f"{'delta':>8} {'naive_tc':>8} {'model_tc':>8}"
        )
        print("--- Production split point estimates ---")
        print(header)
        print("-" * len(header))

        for formulation in ("binary_homerun", "multiclass_3"):
            res, _ = run_production_split(parts, formulation)
            bal = res["balance"]
            if formulation == "binary_homerun":
                bal_short = f"spw={bal['scale_pos_weight']:.3f}"
            else:
                cw = bal["class_weights"]
                bal_short = (
                    f"w[SL/TO/TP]="
                    f"{cw.get('cls0', float('nan')):.2f}/"
                    f"{cw.get('cls1', float('nan')):.2f}/"
                    f"{cw.get('cls2', float('nan')):.2f}"
                )
            print(fmt_row(formulation, "val", res["val"], res["threshold"], bal_short))
            print(fmt_row(formulation, "test", res["test"], res["threshold"], bal_short))
            q = res["proba_tp_val_quantiles"]
            print(
                f"  P(TP)|val quantiles: p10={q['p10']:.3f} p50={q['p50']:.3f} "
                f"p90={q['p90']:.3f} max={q['max']:.3f}"
            )
            print(f"  balance detail: {bal}")
            # Bootstrap the production-split point estimates themselves
            # (resample trades from this fixed model/threshold — NOT walk-forward).
            for split_name in ("val", "test"):
                r = res[split_name]
                if r is None or r["model_tc"] < 20 or r["naive_tc"] < 20:
                    print(f"  boot[{split_name}] (prod-split trades): insufficient")
                    continue
                p5, p50, p95 = bootstrap_delta(r["model_rets"], r["naive_rets"])
                print(
                    f"  boot[{split_name}] delta vs naive (prod-split trades): "
                    f"[{p5:+.4f}, {p50:+.4f}, {p95:+.4f}]"
                )

        print()
        print(
            f"--- OOS bootstrap (n={N_BOOTSTRAP}) — train on bars BEFORE window only ---"
        )
        boot_header = (
            f"{'formulation':<18} {'window':<16} {'m_pf':>7} {'n_pf':>7} {'delta':>7} "
            f"{'m_tc':>5} {'n_tc':>5} {'boot_p5':>8} {'boot_p50':>8} {'boot_p95':>8}"
        )
        print(boot_header)
        print("-" * len(boot_header))
        for formulation in ("binary_homerun", "multiclass_3"):
            for row in run_oos_bootstrap(parts, formulation, tf):
                if row.get("status") != "ok":
                    print(
                        f"{formulation:<18} {row['window']:<16} "
                        f"status={row.get('status')}"
                    )
                    continue
                print(
                    f"{formulation:<18} {row['window']:<16} "
                    f"{row['model_pf']:7.4f} {row['naive_pf']:7.4f} {row['delta']:+7.4f} "
                    f"{row['model_tc']:5d} {row['naive_tc']:5d} "
                    f"{row['p5']:+8.4f} {row['p50']:+8.4f} {row['p95']:+8.4f}"
                )
        print()


if __name__ == "__main__":
    main()
