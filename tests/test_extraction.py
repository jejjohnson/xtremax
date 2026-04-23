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
    declustered_block_maxima,
    quantile_threshold,
    r_largest_block_maxima,
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

    def test_r_largest_with_int_block_size_runs(self, daily_series):
        """Regression: fixed-size (int) block_size branch used to call
        `.values` on numpy sub-arrays yielded by reshape+iterate, raising
        AttributeError before producing any output.
        """
        out = r_largest_block_maxima(daily_series, block_size=30, r=3)
        assert "block" in out.dims and "order" in out.dims
        assert out.sizes["order"] == 3
        assert bool(np.all(np.isfinite(out.values)))
        # Order-1 ≥ order-2 ≥ order-3 within each block (descending).
        assert bool(np.all(np.diff(out.values, axis=1) <= 0))


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

    def test_separation_vectorises_over_extra_dims(self):
        """Regression: `np.flatnonzero(is_peak.values)` used linear indices
        across all dims, so for multi-D arrays positions didn't correspond
        to offsets along `dim`. `apply_ufunc` now applies the 1-D
        selector per batch row.
        """
        time = pd.date_range("2020-01-01", periods=30, freq="D")
        # Two sites with different peak layouts; each has peaks at
        # positions with differing separations.
        values = np.zeros((30, 2), dtype=float)
        values[5, 0], values[20, 0] = 5.0, 3.0  # 15 apart
        values[10, 1], values[14, 1] = 4.0, 2.0  # 4 apart → one kept
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )

        out = decluster_separation(da, threshold=0.5, min_separation=10)

        site0_vals = out.sel(site=0).dropna("time").values.tolist()
        site1_vals = out.sel(site=1).dropna("time").values.tolist()
        assert set(site0_vals) == {5.0, 3.0}  # both kept (15 apart)
        assert set(site1_vals) == {4.0}  # only the larger (too close)

    def test_declustered_block_maxima_separation_applies_min_separation(self):
        """Regression: the `method='separation'` branch of
        declustered_block_maxima previously returned all peaks unchanged.
        It must now enforce min_separation (via decluster_separation).
        """
        values = np.zeros(40, dtype=float)
        values[10], values[25], values[30] = 5.0, 4.0, 3.0
        time = pd.date_range("2020-01-01", periods=40, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = declustered_block_maxima(
            da, threshold=0.5, min_separation=10, method="separation"
        )

        assert out.sizes["time"] == 2
        assert set(out.values.tolist()) == {5.0, 4.0}


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
