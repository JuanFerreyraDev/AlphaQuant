"""Tests for src.brain.features."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.brain.features import (
    compute_all_technicals,
    compute_momentum,
    compute_trend,
    compute_volatility,
    compute_volume,
    add_sentiment,
)


class TestComputeMomentum:
    def test_adds_rsi(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_momentum(ohlcv_df)
        assert "rsi_14" in result.columns

    def test_adds_macd(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_momentum(ohlcv_df)
        assert "macd" in result.columns
        assert "macd_hist" in result.columns


class TestComputeTrend:
    def test_adds_ema_and_adx(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_trend(ohlcv_df)
        assert "ema_50" in result.columns
        assert "dist_ema_50" in result.columns
        assert "adx_14" in result.columns


class TestComputeVolatility:
    def test_adds_atr_and_bb(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_volatility(ohlcv_df)
        assert "atr_14" in result.columns
        assert "bb_width" in result.columns
        assert "bb_pos" in result.columns


class TestComputeVolume:
    def test_adds_obv_and_rel_volume(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_volume(ohlcv_df)
        assert "obv" in result.columns
        assert "rel_volume" in result.columns


class TestComputeAllTechnicals:
    def test_adds_all_indicators(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_all_technicals(ohlcv_df)
        expected = [
            "rsi_14",
            "macd",
            "macd_hist",
            "stoch_k",
            "ema_50",
            "adx_14",
            "atr_14",
            "bb_width",
            "bb_pos",
            "obv",
            "rel_volume",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_lose_existing_columns(self, ohlcv_df: pd.DataFrame) -> None:
        """Original OHLCV columns are preserved."""
        original_cols = set(ohlcv_df.columns)

        result = compute_all_technicals(ohlcv_df)

        for col in original_cols:
            assert col in result.columns


class TestComputeMomentumEdgeCases:
    def test_with_insufficient_data(self) -> None:
        """With fewer than 14 rows, RSI produces NaN but does not fail."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 99.0],
                "high": [102.0, 103.0, 101.0],
                "low": [98.0, 99.0, 97.0],
                "volume": [1000.0, 1100.0, 900.0],
            }
        )

        result = compute_momentum(df)

        assert "rsi_14" in result.columns
        assert result["rsi_14"].isna().any()

    @patch("src.brain.features.ta.macd", return_value=None)
    def test_handles_none_macd(self, _mock_macd, ohlcv_df: pd.DataFrame) -> None:
        """If ta.macd returns None, macd/macd_hist columns are not created."""
        result = compute_momentum(ohlcv_df)

        assert "stoch_k" in result.columns

    @patch("src.brain.features.ta.stoch", return_value=None)
    def test_handles_none_stochastic_defaults_to_zero(
        self, _mock_stoch, ohlcv_df: pd.DataFrame
    ) -> None:
        """If ta.stoch returns None, stoch_k = 0."""
        result = compute_momentum(ohlcv_df)

        assert (result["stoch_k"] == 0).all()

    def test_with_constant_prices(self) -> None:
        """Constant prices (zero variance) do not cause errors."""
        n = 50
        df = pd.DataFrame(
            {
                "close": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "volume": [1000.0] * n,
            }
        )

        result = compute_momentum(df)
        assert "rsi_14" in result.columns


class TestComputeTrendEdgeCases:
    @patch("src.brain.features.ta.adx", return_value=None)
    def test_handles_none_adx(self, _mock_adx, ohlcv_df: pd.DataFrame) -> None:
        """If ta.adx returns None, adx_14 is not added but does not fail."""
        result = compute_trend(ohlcv_df)

        assert "ema_50" in result.columns
        assert "dist_ema_50" in result.columns

    def test_zero_ema_produces_inf_in_dist_ema(self) -> None:
        """If ema_50 is 0, dist_ema_50 can be inf (does not fail)."""
        n = 60
        close = np.concatenate([np.zeros(50), np.ones(10)])
        df = pd.DataFrame(
            {
                "close": close,
                "high": close + 1,
                "low": close - 0.5,
                "volume": [1000.0] * n,
            }
        )

        result = compute_trend(df)

        assert "dist_ema_50" in result.columns


class TestComputeVolatilityEdgeCases:
    @patch("src.brain.features.ta.bbands", return_value=None)
    def test_handles_none_bollinger_bands(
        self, _mock_bb, ohlcv_df: pd.DataFrame
    ) -> None:
        """If ta.bbands returns None, bb_width and bb_pos = 0."""
        result = compute_volatility(ohlcv_df)

        assert (result["bb_width"] == 0).all()
        assert (result["bb_pos"] == 0).all()


class TestComputeVolumeEdgeCases:
    def test_zero_sma_produces_inf_in_rel_volume(self) -> None:
        """If vol_sma_20 is 0, rel_volume will be inf but does not fail."""
        n = 25
        volume = np.concatenate([np.zeros(20), np.ones(5) * 1000])
        df = pd.DataFrame(
            {
                "close": np.random.uniform(100, 200, n),
                "high": np.random.uniform(200, 300, n),
                "low": np.random.uniform(50, 100, n),
                "volume": volume,
            }
        )

        result = compute_volume(df)

        assert "rel_volume" in result.columns


class TestAddSentiment:
    def test_returns_false_when_fng_df_is_empty(self) -> None:
        """If the FNG DataFrame is empty, has_sentiment=False."""
        df = pd.DataFrame({"close": [100.0, 110.0]})
        df_fg = pd.DataFrame()

        result_df, has_sentiment = add_sentiment(df, df_fg)

        assert has_sentiment is False
        assert "fng_value" not in result_df.columns

    def test_adds_rolling_columns_on_success(self) -> None:
        """With valid FNG data, fng_sma_14 and fng_vol_14 are added."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame({"close": np.random.uniform(100, 200, 20)}, index=dates)
        df_fg = pd.DataFrame({"fng_value": np.random.uniform(20, 80, 20)}, index=dates)

        result_df, has_sentiment = add_sentiment(df, df_fg)

        assert has_sentiment is True
        assert "fng_value" in result_df.columns
        assert "fng_sma_14" in result_df.columns
        assert "fng_vol_14" in result_df.columns


class TestAddSentimentSubDaily:
    """Regression tests for the dynamic-timeframe bug where joining daily
    Fear & Greed data onto a sub-daily (e.g. 4h) index produced all-NaN
    ``fng_sma_14``/``fng_vol_14`` columns, which ``cleanup_columns``'s
    ``dropna()`` then turned into total row loss. See Issue 1 of the
    dynamic-timeframe regression audit.
    """

    def test_add_sentiment_4h_no_all_nan_fng_sma(self) -> None:
        """4h candles (midnight-aligned) must inherit a non-null daily FNG
        rolling average instead of ending up all-NaN."""
        dates_4h = pd.date_range("2023-01-01", periods=100, freq="4h")
        df = pd.DataFrame(
            {"close": np.random.uniform(100, 200, 100)}, index=dates_4h
        )

        dates_daily = pd.date_range("2022-12-01", periods=40, freq="D")
        df_fg = pd.DataFrame(
            {"fng_value": np.random.uniform(20, 80, 40)}, index=dates_daily
        )

        result_df, has_sentiment = add_sentiment(df, df_fg)

        assert has_sentiment is True
        assert not result_df["fng_sma_14"].isna().all()
        assert not result_df["fng_vol_14"].isna().all()
        # Most rows (all but the very first daily warmup window) should
        # have a valid rolling value once joined onto the 4h grid.
        assert result_df["fng_sma_14"].notna().sum() > 0

    def test_add_sentiment_daily_unchanged(self) -> None:
        """Daily data behavior is unchanged by the merge_asof rewrite —
        each daily candle still matches its own FNG entry exactly."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        fng_values = np.random.uniform(20, 80, 30)
        df = pd.DataFrame({"close": np.random.uniform(100, 200, 30)}, index=dates)
        df_fg = pd.DataFrame({"fng_value": fng_values}, index=dates)

        result_df, has_sentiment = add_sentiment(df, df_fg)

        assert has_sentiment is True
        np.testing.assert_array_equal(
            result_df["fng_value"].to_numpy(), fng_values
        )
        expected_sma = pd.Series(fng_values).rolling(14).mean().to_numpy()
        np.testing.assert_array_almost_equal(
            result_df["fng_sma_14"].to_numpy(), expected_sma
        )

    def test_add_sentiment_non_midnight_aligned_grid(self) -> None:
        """Sub-daily candles that never land on 00:00:00 must still inherit
        a non-null daily FNG rolling average.

        This is the case an exact-index ``.join()`` cannot handle even with
        pre-computed rolling stats: since no candle timestamp equals a daily
        FNG timestamp, every row would join to NaN. ``merge_asof`` (backward)
        picks up the most recent known daily value regardless of alignment,
        so this test fails under the old join-based implementation and
        passes with the as-of merge.
        """
        dates_offset = pd.date_range(
            "2023-01-01 02:00:00", periods=100, freq="4h"
        )
        df = pd.DataFrame(
            {"close": np.random.uniform(100, 200, 100)}, index=dates_offset
        )
        assert not (dates_offset.hour == 0).any()

        dates_daily = pd.date_range("2022-12-01", periods=40, freq="D")
        df_fg = pd.DataFrame(
            {"fng_value": np.random.uniform(20, 80, 40)}, index=dates_daily
        )

        result_df, has_sentiment = add_sentiment(df, df_fg)

        assert has_sentiment is True
        assert not result_df["fng_sma_14"].isna().all()
        assert not result_df["fng_vol_14"].isna().all()
