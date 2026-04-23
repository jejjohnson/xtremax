"""Smoke tests for extraction utilities (block maxima, thresholds, decluster)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xtremax.extraction import (
    XarrayQuantileRegressor,
    constant_threshold,
    decluster_runs,
    quantile_regression_threshold,
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
