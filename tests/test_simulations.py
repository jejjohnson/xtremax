"""Smoke tests for simulation generators (temporal, spatial, extremes)."""

from __future__ import annotations

import numpy as np
import pytest

from xtremax.simulations import (
    compute_climate_signal,
    generate_fractal_terrain,
    generate_gmst_trajectory,
    generate_iberia_mask,
    generate_physical_gmst,
    generate_spatial_field,
    simulate_precip_extremes,
    simulate_temp_extremes,
    simulate_wind_extremes,
)


class TestTemporal:
    @pytest.mark.parametrize("trend", ["linear", "exponential", "logistic"])
    def test_gmst_trajectory_shape(self, trend):
        n_years = 50
        da = generate_gmst_trajectory(
            n_years=n_years, start_year=1970, trend_type=trend, seed=0
        )
        assert da.dims == ("year",)
        assert da.sizes["year"] == n_years
        assert np.all(np.isfinite(da.values))

    @pytest.mark.parametrize("n_years", [5, 9])
    def test_physical_gmst_short_runs(self, n_years):
        """Regression: `rng.uniform(5, n_years - 5, ...)` raised ValueError
        for n_years < 10 because `high < low`. Short runs must now return
        a valid dataset with an empty eruption schedule.
        """
        ds = generate_physical_gmst(n_years=n_years, seed=0)
        assert "gmst" in ds.data_vars
        assert np.all(np.isfinite(ds["gmst"].values))

    @pytest.mark.parametrize("n_years", [10, 20, 50])
    def test_physical_gmst_year_count(self, n_years):
        """Regression: `np.linspace(0, n_years, n_years * 12)` included
        `t = n_years`, so the floor(t) grouping created an extra final bin
        with a single sample and produced `n_years + 1` annual rows
        instead of `n_years`.
        """
        ds = generate_physical_gmst(n_years=n_years, seed=0)
        assert ds.sizes["year"] == n_years


class TestSpatial:
    def test_iberia_mask(self):
        lat = np.linspace(36.0, 44.0, 20)
        lon = np.linspace(-9.0, 3.0, 30)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        mask = generate_iberia_mask(lat_grid, lon_grid)
        assert mask.shape == lat_grid.shape
        assert mask.dtype == bool
        assert mask.any()

    def test_fractal_terrain(self):
        terrain = generate_fractal_terrain(shape=(32, 32), seed=0)
        assert terrain.shape == (32, 32)
        assert np.all(np.isfinite(terrain))


class TestExtremes:
    def test_spatial_field(self):
        ds = generate_spatial_field(n_sites=20, seed=0)
        assert ds.sizes["site"] == 20
        for name in ("lon", "lat", "elevation"):
            assert name in ds.data_vars

    def test_temp_extremes(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        mu = compute_climate_signal(
            space,
            gmst,
            base_val=20.0,
            coeffs={"elevation": -6.5, "lat": -0.5, "gmst": 1.0, "interaction": 0.1},
        )
        ds = simulate_temp_extremes(mu, scale=1.5, shape=-0.1, seed=0)
        assert "tmax" in ds.data_vars
        assert ds["tmax"].shape == (10, 5)

    def test_precip_extremes(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        ds = simulate_precip_extremes(space, gmst, seed=0)
        assert "rx1day" in ds.data_vars
        assert "cwd" in ds.data_vars

    def test_wind_extremes(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        ds = simulate_wind_extremes(space, gmst, seed=0)
        assert "wind_max" in ds.data_vars
