"""Tests for out-of-sample walk-forward validation module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.utils.data_splits import (
    compute_dynamic_split,
    compute_split_boundaries,
    compute_train_val_split,
)
from src.utils.helpers import train_predict_binary_homerun
from src.utils.oos_validation import MIN_POOLED_TRADES, run_walk_forward


class TestWalkForwardValidation:
    @staticmethod
    def create_synthetic_data(n_bars: int = 10000, edge: bool = False) -> pd.DataFrame:
        """Generate synthetic price, indicator, and target data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="4h")

        df = pd.DataFrame(index=dates)
        df["close"] = np.cumprod(1 + np.random.normal(0, 0.001, n_bars)) * 10000.0
        df["atr_14"] = df["close"] * 0.02

        # Ternary target: -1 (SL), 0 (timeout), 1 (TP)
        df["target"] = np.random.choice([-1, 0, 1], size=n_bars, p=[0.4, 0.2, 0.4])

        if edge:
            # Predictor strongly correlated with class 1 target
            df["feature_1"] = np.where(df["target"] == 1, 1.0, -1.0) + np.random.normal(
                0, 0.1, n_bars
            )
        else:
            # Pure white noise predictor
            df["feature_1"] = np.random.normal(0, 1, n_bars)

        return df

    def test_walk_forward_with_edge(self) -> None:
        """Validate that a dataset with genuine edge passes the gate."""
        df = self.create_synthetic_data(n_bars=10000, edge=True)

        res = run_walk_forward(
            df_raw=df,
            symbol="BTC_USDT",
            timeframe="4h",
            train_predict_fn=train_predict_binary_homerun,
            tp_multi=1.5,
            sl_multi=1.0,
            swing_period=10,
            features=["feature_1"],
            window_months=6,
            step_months=6,
            n_bootstrap=100,
        )

        assert res.pooled_trade_count >= MIN_POOLED_TRADES
        assert res.pooled_delta_bootstrap[0] > 0.0
        assert res.passes_gate is True

    def test_walk_forward_pure_noise(self) -> None:
        """Validate that a dataset with pure noise fails the gate."""
        df = self.create_synthetic_data(n_bars=10000, edge=False)

        res = run_walk_forward(
            df_raw=df,
            symbol="BTC_USDT",
            timeframe="4h",
            train_predict_fn=train_predict_binary_homerun,
            tp_multi=1.5,
            sl_multi=1.0,
            swing_period=10,
            features=["feature_1"],
            window_months=6,
            step_months=6,
            n_bootstrap=100,
        )

        assert res.passes_gate is False
        assert res.pooled_delta_bootstrap[0] <= 0.0 or np.isnan(
            res.pooled_delta_bootstrap[0]
        )

    def test_walk_forward_low_trade_count(self) -> None:
        """Validate that low trade count fails the gate even if delta is positive."""
        df = self.create_synthetic_data(n_bars=3500, edge=True)

        res = run_walk_forward(
            df_raw=df,
            symbol="BTC_USDT",
            timeframe="4h",
            train_predict_fn=train_predict_binary_homerun,
            tp_multi=1.5,
            sl_multi=1.0,
            swing_period=10,
            features=["feature_1"],
            window_months=6,
            step_months=6,
            n_bootstrap=10,
        )

        assert res.pooled_trade_count < MIN_POOLED_TRADES
        assert res.passes_gate is False

    def test_walk_forward_skip_insufficient_prior(self) -> None:
        """Validate that windows with insufficient prior data are skipped smoothly."""
        df = self.create_synthetic_data(n_bars=500, edge=False)

        res = run_walk_forward(
            df_raw=df,
            symbol="BTC_USDT",
            timeframe="4h",
            train_predict_fn=train_predict_binary_homerun,
            tp_multi=1.5,
            sl_multi=1.0,
            swing_period=10,
            features=["feature_1"],
            window_months=1,
            step_months=1,
            n_bootstrap=10,
        )

        skipped = [
            w for w in res.windows if w.skipped_reason == "insufficient_prior_data"
        ]
        assert len(skipped) > 0, "No windows were skipped. Check minimum data threshold."
        assert isinstance(res.passes_gate, bool)

    def test_walk_forward_uses_train_val_split_not_3way(self) -> None:
        """Integration: run_walk_forward must call compute_train_val_split.

        Confirms the patched code path is actually exercised by a real
        run_walk_forward invocation, not just that the function exists.

        Also asserts compute_dynamic_split (3-way) is never called by the
        walk-forward engine. compute_dynamic_split lives in data_splits and
        is no longer imported by oos_validation after fix #1, so we patch
        at the data_splits level.
        """
        df = self.create_synthetic_data(n_bars=5000, edge=True)

        with patch(
            "src.utils.oos_validation.compute_train_val_split",
            wraps=compute_train_val_split,
        ) as spy_tv, patch(
            "src.utils.data_splits.compute_dynamic_split",
            wraps=compute_dynamic_split,
        ) as spy_3way_ds:
            _ = run_walk_forward(
                df_raw=df,
                symbol="BTC_USDT",
                timeframe="4h",
                train_predict_fn=train_predict_binary_homerun,
                tp_multi=1.5,
                sl_multi=1.0,
                swing_period=10,
                features=["feature_1"],
                window_months=2,
                step_months=2,
                n_bootstrap=10,
            )

            assert spy_tv.call_count > 0, (
                "run_walk_forward must call compute_train_val_split "
                "(2-part split, fix #1) — not the 3-way compute_dynamic_split."
            )
            assert spy_3way_ds.call_count == 0, (
                "run_walk_forward must NOT call compute_dynamic_split at all "
                "any more (patched at data_splits level). The 3-way split "
                "reserves an unused test block inside prior_data and starves train."
            )


class TestSplitBoundariesTwoPart:
    """Isolated checks on compute_split_boundaries when called with n_test=0.

    run_walk_forward now calls compute_split_boundaries(n_train, n_val, 0,
    embargo_days=...) directly (fix #2). This class does NOT go through
    run_walk_forward or compute_train_val_split — it calls
    compute_split_boundaries in isolation with hand-picked numbers, so a
    failure here points unambiguously at compute_split_boundaries itself,
    not at anything upstream.

    compute_split_boundaries was originally designed assuming 3 real
    blocks (train/val/test), each separated by an embargo gap. These tests
    exist because that assumption may not degrade gracefully to n_test=0 —
    e.g. an extra embargo could be appended after val "for the test block"
    even though no test block exists, silently shrinking val or wasting
    bars that should be available.
    """

    @pytest.mark.parametrize(
        "n_train,n_val,embargo_days",
        [
            (1000, 200, 10),
            (5000, 500, 25),
            (300, 50, 5),
        ],
    )
    def test_val_slice_matches_hand_computed_boundaries(
        self, n_train: int, n_val: int, embargo_days: int
    ) -> None:
        """val_slice must start exactly at n_train + embargo_days.

        With n_test=0 there is no trailing block, so nothing after val
        should consume extra bars — val_slice.stop should be exactly
        n_train + embargo_days + n_val, not that plus another embargo.
        """
        train_slice, val_slice, test_slice = compute_split_boundaries(
            n_train, n_val, 0, embargo_days=embargo_days,
        )

        expected_train_start = 0
        expected_train_stop = n_train
        expected_val_start = n_train + embargo_days
        expected_val_stop = expected_val_start + n_val

        assert train_slice.start == expected_train_start
        assert train_slice.stop == expected_train_stop, (
            f"train_slice.stop={train_slice.stop} != expected "
            f"{expected_train_stop}. If this fails, compute_split_boundaries "
            f"is consuming train bars for a phantom test block."
        )
        assert val_slice.start == expected_val_start, (
            f"val_slice.start={val_slice.start} != expected "
            f"{expected_val_start} (n_train + embargo_days). A mismatch "
            f"here means the embargo before val is wrong when n_test=0."
        )
        assert val_slice.stop == expected_val_stop, (
            f"val_slice.stop={val_slice.stop} != expected {expected_val_stop} "
            f"(n_train + embargo_days + n_val). If val_slice.stop is larger "
            f"than expected, an extra embargo is being appended after val "
            f"for a test block that does not exist (n_test=0) — this is "
            f"the exact off-by-one risk flagged in review."
        )

        # test_slice should be empty / degenerate when n_test=0.
        test_len = (test_slice.stop or 0) - (test_slice.start or 0)
        assert test_len <= 0, (
            f"test_slice should be empty when n_test=0, got "
            f"start={test_slice.start} stop={test_slice.stop}"
        )

    def test_val_slice_does_not_overrun_available_bars(self) -> None:
        """val_slice must not exceed n_train + embargo + n_val bars total.

        Guards against compute_split_boundaries silently padding val with
        extra bars that were meant for a test block that no longer exists.
        """
        n_train, n_val, embargo_days = 2000, 300, 15
        _, val_slice, _ = compute_split_boundaries(
            n_train, n_val, 0, embargo_days=embargo_days,
        )
        max_expected_bars_consumed = n_train + embargo_days + n_val
        assert val_slice.stop <= max_expected_bars_consumed, (
            f"val_slice.stop={val_slice.stop} exceeds the maximum expected "
            f"bar consumption ({max_expected_bars_consumed}) for a 2-part "
            f"split with n_test=0. This means bars are being reserved for "
            f"a test block that will never be used — exactly the bug fix #1 "
            f"was meant to eliminate."
        )