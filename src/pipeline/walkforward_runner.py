"""Run baseline and A/B walk-forward experiments and write JSON reports."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config.experiment_defaults import (
    DEFAULT_TIMEFRAMES,
    ExperimentConfig,
    FORMULATIONS,
    MIN_BASELINE_DELTA_P5,
)
from src.config.paths import _sanitize_symbol, get_report_dir, get_report_path
from src.pipeline.dataset_builder import build_dataset
from src.pipeline.feature_profiles import get_profile
from src.utils.oos_validation import (
    MIN_POOLED_TRADES,
    WalkForwardResult,
    _profit_factor,
    run_walk_forward,
)


def _window_stats(windows: list) -> dict[str, int]:
    n_used = sum(1 for w in windows if w.skipped_reason is None)
    n_pos = sum(
        1 for w in windows
        if w.skipped_reason is None and w.model_pf > 1.0
    )
    n_delta_pos = sum(
        1 for w in windows
        if w.skipped_reason is None and w.delta > 0
    )
    n_skip = sum(1 for w in windows if w.skipped_reason is not None)
    return {
        "total_windows": len(windows),
        "windows_used": n_used,
        "windows_pf_gt_1": n_pos,
        "windows_delta_gt_0": n_delta_pos,
        "windows_skipped": n_skip,
    }


def _run_walk_forward(
    df,
    features: list[str],
    symbol: str,
    timeframe: str,
    train_fn,
    threshold_grid: tuple[float, float, float],
    config: ExperimentConfig,
) -> WalkForwardResult:
    return run_walk_forward(
        df_raw=df,
        symbol=symbol,
        timeframe=timeframe,
        train_predict_fn=train_fn,
        features=features,
        threshold_grid=threshold_grid,
        **config.walk_forward_kwargs(),
    )


def _baseline_row(
    symbol: str,
    timeframe: str,
    form_name: str,
    result: WalkForwardResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    pooled_rets = (
        np.concatenate([a for a in result.model_rets_all if len(a) > 0])
        if result.model_rets_all
        else np.array([])
    )
    pooled_pf_point = _profit_factor(pooled_rets)
    p5_delta, p95_delta = result.pooled_delta_bootstrap
    passes = result.passes_gate
    stats = _window_stats(result.windows)
    return {
        "symbol": _sanitize_symbol(symbol),
        "timeframe": timeframe,
        "formulation": form_name,
        "gate_mode": "delta_pf_p5_gt_0",
        "oos_pf_point": float(pooled_pf_point),
        "oos_delta_pf_p5": float(p5_delta) if np.isfinite(p5_delta) else None,
        "oos_delta_pf_p95": float(p95_delta) if np.isfinite(p95_delta) else None,
        "pooled_trades": int(result.pooled_trade_count),
        "passes_gate": bool(passes),
        **stats,
        "windows": [asdict(w) for w in result.windows],
    }


def _ab_test_row(
    symbol: str,
    timeframe: str,
    form_name: str,
    variant: str,
    profile_name: str,
    result: WalkForwardResult,
) -> dict[str, Any]:
    p5, p95 = result.pooled_delta_bootstrap
    stats = _window_stats(result.windows)
    n_pos_delta = sum(
        1 for w in result.windows
        if w.skipped_reason is None and w.delta > 0
    )
    return {
        "symbol": _sanitize_symbol(symbol),
        "timeframe": timeframe,
        "formulation": form_name,
        "variant": variant,
        "profile": profile_name,
        "gate_mode": "delta_pf_p5_gt_0",
        "pooled_delta_bootstrap_p5": p5,
        "pooled_delta_bootstrap_p95": p95,
        "pooled_trade_count": result.pooled_trade_count,
        "passes_gate": result.passes_gate,
        "windows_delta_gt_0": n_pos_delta,
        **stats,
        "windows": [asdict(w) for w in result.windows],
    }


def write_report(payload: dict[str, Any], path: Path) -> Path:
    """Write JSON report and update ``latest_{experiment}.json`` symlink/copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    experiment = payload.get("experiment", "report")
    latest = path.parent / f"latest_{experiment}.json"
    shutil.copy2(path, latest)
    return path


def run_baseline(
    symbol: str,
    timeframes: list[str] | None = None,
    config: ExperimentConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Screen a symbol with the 14-feature control baseline across formulations."""
    cfg = config or ExperimentConfig()
    tfs = timeframes or list(DEFAULT_TIMEFRAMES)
    safe = _sanitize_symbol(symbol)
    t0 = time.time()

    print("=" * 90)
    print(f"BASELINE SCREEN — {safe} (14 control features)")
    print(
        f"Swing={cfg.swing_period}  TP={cfg.tp_multi}xATR  SL={cfg.sl_multi}xATR  "
        f"Window={cfg.window_months}m  Step={cfg.step_months}m  "
        f"Gate: ΔPF p5 > {MIN_BASELINE_DELTA_P5} (model vs naive_long)"
    )
    print("=" * 90)

    master: dict[str, tuple] = {}
    for tf in tfs:
        print(f"\n⟳ Building dataset {safe}/{tf} [profile=control]...")
        master[tf] = build_dataset(safe, tf, profile="control", config=cfg)
        df, feats = master[tf]
        print(f"  ✓ {len(df)} rows × {len(feats)} features")
        print(f"    Range: {df.index.min().date()} → {df.index.max().date()}")

    rows: list[dict[str, Any]] = []
    run_n = 0
    total_runs = len(tfs) * len(FORMULATIONS)

    for tf in tfs:
        df, feats = master[tf]
        for form_name, train_fn, grid in FORMULATIONS:
            run_n += 1
            tag = f"[{run_n}/{total_runs}] {tf} × {form_name}"
            print(f"\n{'─' * 90}\n{tag}\n{'─' * 90}")

            t_start = time.time()
            result = _run_walk_forward(df, feats, safe, tf, train_fn, grid, cfg)
            row = _baseline_row(safe, tf, form_name, result, cfg)
            rows.append(row)

            gate = "✅ PASS" if row["passes_gate"] else "❌ FAIL"
            p5 = row["oos_delta_pf_p5"]
            p5_str = f"{p5:+.4f}" if p5 is not None else "nan"
            print(
                f"  OOS PF: point={row['oos_pf_point']:.4f}  ΔPF p5={p5_str}  "
                f"trades={row['pooled_trades']}  gate={gate}  "
                f"⏱ {time.time() - t_start:.1f}s"
            )

    print("\n" + "=" * 120)
    print(f"SUMMARY — {safe} Baseline")
    print("=" * 120)
    hdr = (
        f"{'Config':<28} {'OOS PF':>8} {'ΔPF p5':>8} {'Trades':>8} "
        f"{'Gate':>6} {'#w_used':>8} {'#w_Δ>0':>8}"
    )
    print(hdr)
    print("-" * 120)
    for r in rows:
        p5 = r["oos_delta_pf_p5"] if r["oos_delta_pf_p5"] is not None else float("nan")
        gate = "PASS" if r["passes_gate"] else "FAIL"
        label = f"{r['timeframe']} × {r['formulation']}"
        print(
            f"{label:<28} {r['oos_pf_point']:>8.4f} {p5:>+8.4f} "
            f"{r['pooled_trades']:>8d} {gate:>6} "
            f"{r['windows_used']:>8d} {r['windows_delta_gt_0']:>8d}"
        )
    print("-" * 120)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir or get_report_dir(safe)
    out_path = report_dir / f"baseline_{stamp}.json"
    payload = {
        "experiment": "baseline",
        "symbol": safe,
        "profile": "control",
        "config": asdict(cfg),
        "gate_threshold_delta_pf_p5": MIN_BASELINE_DELTA_P5,
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": rows,
    }
    write_report(payload, out_path)
    print(f"\n💾 Report: {out_path}")
    print(f"⏱ Total: {time.time() - t0:.1f}s")
    return payload


def run_ab_test(
    symbol: str,
    profile: str,
    timeframes: list[str] | None = None,
    config: ExperimentConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run control vs treatment walk-forward for a feature profile."""
    cfg = config or ExperimentConfig()
    prof = get_profile(profile)
    if prof.treatment_col is None:
        raise ValueError(
            f"Profile {profile!r} has no treatment column — use 'baseline' instead."
        )

    tfs = timeframes or list(DEFAULT_TIMEFRAMES)
    safe = _sanitize_symbol(symbol)
    t0 = time.time()

    print("=" * 90)
    print(f"A/B TEST — {safe}  profile={prof.name}  treatment={prof.treatment_col}")
    print(
        f"Swing={cfg.swing_period}  TP={cfg.tp_multi}xATR  SL={cfg.sl_multi}xATR  "
        f"Window={cfg.window_months}m  Step={cfg.step_months}m  "
        f"Gate: ΔPF p5 > 0"
    )
    print("=" * 90)

    master: dict[str, tuple] = {}
    for tf in tfs:
        print(f"\n⟳ Building dataset {safe}/{tf} [profile={prof.name}]...")
        master[tf] = build_dataset(safe, tf, profile=prof, config=cfg)
        df, control_feats = master[tf]
        print(f"  ✓ {len(df)} rows × {len(control_feats)} control features "
              f"(+ {prof.treatment_col} for treatment)")

    all_results: list[dict[str, Any]] = []
    run_n = 0
    total_runs = len(tfs) * len(FORMULATIONS) * 2

    for tf in tfs:
        df, control_feats = master[tf]
        treatment_feats = control_feats + [prof.treatment_col]

        for form_name, train_fn, grid in FORMULATIONS:
            for variant, features in [
                ("CONTROL", control_feats),
                ("TREATMENT", treatment_feats),
            ]:
                run_n += 1
                tag = f"[{run_n}/{total_runs}] {tf} × {form_name} — {variant}"
                print(f"\n{'─' * 90}\n{tag}\n{'─' * 90}")

                t_start = time.time()
                result = _run_walk_forward(df, features, safe, tf, train_fn, grid, cfg)
                row = _ab_test_row(
                    safe, tf, form_name, variant, prof.name, result,
                )
                all_results.append(row)

                p5, p95 = result.pooled_delta_bootstrap
                gate = "✅ PASS" if result.passes_gate else "❌ FAIL"
                print(
                    f"  ΔPF bootstrap: p5={p5:+.4f}  p95={p95:+.4f}  "
                    f"trades={result.pooled_trade_count}  gate={gate}  "
                    f"⏱ {time.time() - t_start:.1f}s"
                )

    print("\n" + "=" * 100)
    print(f"SUMMARY — {safe} A/B  profile={prof.name}")
    print("=" * 100)
    for r in all_results:
        gate = "PASS" if r["passes_gate"] else "FAIL"
        label = f"{r['timeframe']} × {r['formulation']}"
        print(
            f"{label:<32} {r['variant']:<12} "
            f"p5={r['pooled_delta_bootstrap_p5']:+.4f}  "
            f"trades={r['pooled_trade_count']:>5d}  gate={gate}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir or get_report_dir(safe)
    exp_slug = f"ab_test_{prof.name}"
    out_path = get_report_path(safe, exp_slug, stamp)

    payload = {
        "experiment": exp_slug,
        "symbol": safe,
        "profile": prof.name,
        "treatment_col": prof.treatment_col,
        "config": asdict(cfg),
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": all_results,
    }
    write_report(payload, out_path)
    print(f"\n💾 Report: {out_path}")
    print(f"⏱ Total: {time.time() - t0:.1f}s")
    return payload
