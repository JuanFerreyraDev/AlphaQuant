"""sweep_config_grid.py — Config grid sweep over (swing, tp_multi, sl_multi).

Evaluates the CONTROL profile (14 features, no treatment) over a pre-defined
grid of (swing, tp, sl) combinations, applying the neutral%>50% exclusion
criterion independently per (combo, timeframe).

Excluded combos (documented before running, not post-hoc):
  TP < SL (structural RR < 1.0): (1.0,1.5), (1.0,2.0), (1.5,2.0) — any swing
  neutral% > 50% per TF:
    swing=5,  tp=2.0, sl=1.5  (4h=57%, 1h=59%)
    swing=5,  tp=2.0, sl=2.0  (4h=67%, 1h=68%)
    swing=7,  tp=2.0, sl=2.0  (4h=55%, 1h=57%)
    swing=5,  tp=1.5, sl=1.5  excluded in 1h only (1h=51%) — asymmetric, runs in 4h

Total included: 27 runs in 4h, 26 runs in 1h = 53 combo-TF slots × 3 formulations
= 159 runs total.

Persistence strategy: each result is written to a JSONL (newline-delimited JSON)
file immediately after the run completes, so a timeout never loses previously
completed runs. On restart, already-completed runs are skipped.

Usage:
    python -m tools.sweep_config_grid [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.experiment_defaults import FORMULATIONS, ExperimentConfig
from src.config.paths import _sanitize_symbol, get_report_dir
from src.pipeline.dataset_builder import build_dataset
from src.pipeline.walkforward_runner import _run_walk_forward

# ---------------------------------------------------------------------------
# Grid definition — pre-registered, not modified after seeing results
# ---------------------------------------------------------------------------

_TP_SL_PAIRS: list[tuple[float, float]] = [
    (1.0, 1.0),
    (1.5, 1.0),
    (1.5, 1.5),
    (2.0, 1.0),
    (2.0, 1.5),
    (2.0, 2.0),
]
_SWINGS: list[int] = [5, 7, 10, 15, 20]
_TIMEFRAMES: list[str] = ["4h", "1h"]

# neutral% > 50% exclusions (per TF, determined from exploratory run, pre-registered)
_EXCLUDED: set[tuple[int, float, float, str]] = {
    (5, 2.0, 1.5, "4h"), (5, 2.0, 1.5, "1h"),
    (5, 2.0, 2.0, "4h"), (5, 2.0, 2.0, "1h"),
    (7, 2.0, 2.0, "4h"), (7, 2.0, 2.0, "1h"),
    (5, 1.5, 1.5, "1h"),   # asymmetric: only 1h excluded
}

SYMBOL = "BTC_USDT"


def _build_grid() -> list[tuple[int, float, float, str]]:
    """Return list of (swing, tp, sl, tf) tuples to run (combo-TF slots)."""
    slots = []
    for swing in _SWINGS:
        for tp, sl in _TP_SL_PAIRS:
            for tf in _TIMEFRAMES:
                if (swing, tp, sl, tf) not in _EXCLUDED:
                    slots.append((swing, tp, sl, tf))
    return slots


def _load_completed(jsonl_path: Path) -> set[tuple]:
    """Return set of (swing, tp, sl, tf, formulation) keys already in the JSONL file."""
    if not jsonl_path.exists():
        return set()
    completed = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add((
                    row["swing"], row["tp_multi"], row["sl_multi"],
                    row["timeframe"], row["formulation"],
                ))
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def main(resume: bool = False) -> None:
    grid = _build_grid()
    safe = _sanitize_symbol(SYMBOL)

    total_combo_tf = len(grid)
    total_runs = total_combo_tf * len(FORMULATIONS)

    report_dir = get_report_dir(safe)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Fixed JSONL path — same file for resume
    jsonl_path = report_dir / "config_sweep_results.jsonl"

    completed = _load_completed(jsonl_path) if resume else set()
    if resume and completed:
        print(f"⟳ Resuming — {len(completed)} runs already completed, skipping.")
    elif not resume and jsonl_path.exists():
        # Fresh run: back up previous and start clean
        backup = jsonl_path.with_suffix(
            f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl.bak"
        )
        jsonl_path.rename(backup)
        print(f"  Previous results backed up to: {backup}")
        completed = set()

    print("=" * 90)
    print(f"CONFIG GRID SWEEP — {safe} | profile=control (14 features)")
    print(f"Grid: {total_combo_tf} combo-TF slots × {len(FORMULATIONS)} formulations = {total_runs} runs")
    print(f"Output (append-per-run): {jsonl_path}")
    print(f"Seed: 42 (fixed)  |  Bootstrap: 1000 iters / 8 blocks")
    print("=" * 90)

    # Pre-build all unique (swing, tp, sl, tf) datasets
    print("\n⟳ Pre-building datasets (one per unique config×TF)...")
    dataset_cache: dict[tuple, tuple] = {}
    for swing, tp, sl, tf in grid:
        key = (swing, tp, sl, tf)
        if key not in dataset_cache:
            cfg = ExperimentConfig(swing_period=swing, tp_multi=tp, sl_multi=sl)
            df, feats = build_dataset(SYMBOL, tf, profile="control", config=cfg)
            dataset_cache[key] = (df, feats, cfg)
    print(f"  ✓ {len(dataset_cache)} datasets cached\n")

    run_n = 0
    skipped = 0
    t_sweep_start = time.time()

    with open(jsonl_path, "a") as fh:
        for swing, tp, sl, tf in grid:
            key = (swing, tp, sl, tf)
            df, feats, cfg = dataset_cache[key]

            for form_name, train_fn, grid_thresh, target_col in FORMULATIONS:
                run_n += 1
                run_key = (swing, tp, sl, tf, form_name)

                if run_key in completed:
                    skipped += 1
                    print(f"[{run_n:>3}/{total_runs}] SKIP  "
                          f"sw={swing:>2} tp={tp} sl={sl} {tf} × {form_name}")
                    continue

                t_run = time.time()
                result = _run_walk_forward(
                    df, feats, safe, tf, train_fn, grid_thresh, cfg,
                    target_col=target_col,
                )
                p5, p95 = result.pooled_delta_bootstrap
                gate = result.passes_gate
                elapsed = time.time() - t_run

                row = {
                    "swing":         swing,
                    "tp_multi":      tp,
                    "sl_multi":      sl,
                    "rr":            round(tp / sl, 3),
                    "timeframe":     tf,
                    "formulation":   form_name,
                    "p5":            float(p5),
                    "p95":           float(p95),
                    "ci_width":      float(p95 - p5),
                    "pooled_trades": int(result.pooled_trade_count),
                    "passes_gate":   bool(gate),
                    "windows_used":  sum(
                        1 for w in result.windows if w.skipped_reason is None
                    ),
                    "windows_delta_gt_0": sum(
                        1 for w in result.windows
                        if w.skipped_reason is None and w.delta > 0
                    ),
                    "elapsed_s": round(elapsed, 1),
                }

                # Append immediately — timeout-safe
                fh.write(json.dumps(row) + "\n")
                fh.flush()

                gate_str = "✅ PASS" if gate else "❌ FAIL"
                print(
                    f"[{run_n:>3}/{total_runs}] "
                    f"sw={swing:>2} tp={tp} sl={sl} {tf} × {form_name:<22} "
                    f"p5={p5:+.4f} p95={p95:+.4f} tr={result.pooled_trade_count:>5} "
                    f"{gate_str}  ⏱{elapsed:.1f}s"
                )

    elapsed_total = time.time() - t_sweep_start

    # Write final consolidated JSON for easy loading
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_json = report_dir / f"config_sweep_final_{stamp}.json"
    payload = {
        "experiment":       "config_sweep",
        "symbol":           safe,
        "profile":          "control",
        "n_expected":       total_runs,
        "n_completed":      len(rows),
        "elapsed_seconds":  round(elapsed_total, 1),
        "exclusion_rules": {
            "tp_lt_sl":           "all combos where tp < sl (structural RR<1)",
            "neutral_gt_50pct":   "per (swing,tp,sl,tf) independently",
            "asymmetric_exclusions": [
                "swing=5,tp=1.5,sl=1.5 excluded in 1h only (neutral=50.9%)"
            ],
        },
        "results": rows,
    }
    with open(final_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n💾 JSONL (append log): {jsonl_path}")
    print(f"💾 Final consolidated JSON: {final_json}")
    print(f"📊 Rows written: {len(rows)} / {total_runs} expected")
    print(f"⏱ Total: {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing JSONL, skipping already-completed runs.",
    )
    args = parser.parse_args()
    main(resume=args.resume)
