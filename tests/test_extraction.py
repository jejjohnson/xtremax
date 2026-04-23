"""Smoke tests for extraction utilities (block maxima, thresholds, decluster)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xtremax.extraction import (
    constant_threshold,
    decluster_runs,
    decluster_separation,
    quantile_threshold,
    rolling_threshold,
    seasonal_threshold,
    temporal_block_maxima,
    temporal_threshold,
)


@pytest.fixture
def daily_series():
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
    values = rng.standard_normal(len(time)) + 0.1 * np.arange(len(time)) / len(time)
    return xr.DataArray(values, dims="time", coords={"time": time})


class TestBlockMaxima:
    def test_annual_block_maxima(self, daily_series):
        annual = temporal_block_maxima(daily_series, freq="YS")
        assert "time" in annual.dims
        assert annual.sizes["time"] == 3  # 3 years
        assert bool(np.all(np.isfinite(annual.values)))


class TestThreshold:
    def test_constant(self, daily_series):
        assert constant_threshold(daily_series, 1.5) == 1.5

    def test_quantile(self, daily_series):
        u = quantile_threshold(daily_series, 0.95)
        empirical = float(np.quantile(daily_series.values, 0.95))
        assert np.isclose(float(u), empirical, rtol=1e-5)

    def test_temporal(self, daily_series):
        u = temporal_threshold(daily_series, 0.95, groupby="time.year")
        assert "year" in u.dims or "time" in u.dims

    def test_rolling(self, daily_series):
        u = rolling_threshold(daily_series, 0.95, window_size=30)
        assert u.sizes["time"] == daily_series.sizes["time"]

    def test_seasonal(self, daily_series):
        u = seasonal_threshold(daily_series, 0.95)
        # seasonal threshold returns one value per season
        assert "season" in u.dims or "time" in u.dims


class TestDecluster:
    def test_runs(self, daily_series):
        out = decluster_runs(daily_series, threshold=1.5, reduction="max")
        assert isinstance(out, xr.DataArray)

    def test_separation_measures_original_time_steps(self):
        """Regression: min_separation must count original time steps, not peaks apart.

        Construct a sparse series with three isolated exceedance peaks at
        positions [10, 25, 30] (values 5, 4, 3). With min_separation=10 and
        the old buggy code (peak_times = 0..n_peaks-1), every peak is 1 apart
        so only the largest would be kept. The correct behaviour keeps the
        peak at 10 (value 5) and the one at 25 (5 > 10 steps away); 30 is
        within 5 of 25 so gets dropped.
        """
        values = np.zeros(40, dtype=float)
        # Each peak flanked by zeros so the "local max above threshold"
        # detector picks only the peak indices.
        values[10] = 5.0
        values[25] = 4.0
        values[30] = 3.0
        time = pd.date_range("2020-01-01", periods=40, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=0.5, min_separation=10)

        assert out.sizes["time"] == 2
        assert set(out.values.tolist()) == {5.0, 4.0}

    def test_separation_keeps_only_largest_when_all_too_close(self):
        values = np.zeros(20, dtype=float)
        values[5], values[8], values[11] = 3.0, 2.0, 4.0
        time = pd.date_range("2020-01-01", periods=20, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=0.5, min_separation=10)

        assert out.sizes["time"] == 1
        assert float(out.values[0]) == 4.0


# Quantile-regression threshold selection needs scikit-learn (optional
# `[threshold]` extra). Skip when sklearn isn't installed.
pytest.importorskip("sklearn")
from xtremax.extraction.quantile_regression import (
    XarrayQuantileRegressor,
    quantile_regression_threshold,
)


class TestQuantileRegression:
    def test_regressor_fits(self, daily_series):
        reg = XarrayQuantileRegressor(quantile=0.95)
        # Use time as covariate
        t = xr.DataArray(
            np.arange(daily_series.sizes["time"], dtype=float),
            dims="time",
            coords={"time": daily_series["time"]},
        )
        reg.fit(t.expand_dims("feature", axis=-1), daily_series)
        preds = reg.predict(t.expand_dims("feature", axis=-1))
        assert preds.shape == daily_series.shape

    def test_threshold_function(self, daily_series):
        u = quantile_regression_threshold(daily_series, quantile=0.95, time_dim="time")
        assert u.shape == daily_series.shape
