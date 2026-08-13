"""Shared walk-forward experiment defaults for the multi-asset pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.helpers import (
    BINARY_HOMERUN_THRESHOLD_GRID,
    MULTICLASS_3_THRESHOLD_GRID,
    train_predict_binary_homerun,
    train_predict_multiclass_3,
)

# Pre-registered baseline screening gate: pooled ΔPF p5 > 0.0 (model vs naive_long)
MIN_BASELINE_DELTA_P5: float = 0.0

DEFAULT_TIMEFRAMES: tuple[str, ...] = ("4h", "1h")

FORMULATIONS: tuple[tuple[str, object, tuple[float, float, float]], ...] = (
    ("binary_homerun", train_predict_binary_homerun, BINARY_HOMERUN_THRESHOLD_GRID),
    ("multiclass_3", train_predict_multiclass_3, MULTICLASS_3_THRESHOLD_GRID),
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Walk-forward parameters shared across baseline and A/B experiments."""

    swing_period: int = 10
    tp_multi: float = 1.5
    sl_multi: float = 1.0
    window_months: int = 6
    step_months: int = 6
    fee_rate: float = 0.0
    slippage: float = 0.0
    n_bootstrap: int = 1000
    n_blocks: int = 8
    random_state: int = 42

    def walk_forward_kwargs(self) -> dict:
        """Keyword arguments forwarded to ``run_walk_forward``."""
        return {
            "tp_multi": self.tp_multi,
            "sl_multi": self.sl_multi,
            "swing_period": self.swing_period,
            "window_months": self.window_months,
            "step_months": self.step_months,
            "fee_rate": self.fee_rate,
            "slippage": self.slippage,
            "n_bootstrap": self.n_bootstrap,
            "n_blocks": self.n_blocks,
            "random_state": self.random_state,
        }
