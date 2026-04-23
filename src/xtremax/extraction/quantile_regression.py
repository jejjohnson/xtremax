"""
Quantile regression utilities for extremes thresholding.

Provides two layers:

1. ``XarrayQuantileRegressor`` — a thin sklearn-compatible wrapper around
   ``sklearn.linear_model.QuantileRegressor`` that accepts and returns
   ``xr.DataArray`` objects, preserving coordinates through fit → predict.

2. ``quantile_regression_threshold`` — domain-specific function that builds
   the feature matrix (numeric time, optional covariates) and delegates to
   the regressor, returning a time-varying threshold suitable for
   peak-over-threshold analysis.

Example
-------
>>> import xarray as xr, numpy as np, pandas as pd
>>> da = xr.DataArray(
...     np.random.default_rng(0).standard_normal(200),
...     dims="time",
...     coords={"time": pd.date_range("2000", periods=200, freq="D")},
... )
>>> threshold = quantile_regression_threshold(da, quantile=0.95, time_dim="time")
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor as _SklearnQuantileRegressor
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Layer 1: sklearn-compatible regressor that operates on xr.DataArray
# ---------------------------------------------------------------------------


class XarrayQuantileRegressor(BaseEstimator, RegressorMixin):
    """Quantile regressor that operates on :class:`xr.DataArray` inputs.

    Thin wrapper around :class:`sklearn.linear_model.QuantileRegressor`.
    Both ``fit`` and ``predict`` expect ``DataArray`` objects; coordinate
    metadata on the leading dimension is preserved through the round-trip.

    Parameters
    ----------
    quantile : float
        Target quantile ∈ (0, 1).
    alpha : float
        L1 regularisation strength.  ``0`` gives the classical
        unregularised linear quantile regression (matching the default
        behaviour of ``statsmodels.QuantReg``).
    solver : str
        Solver forwarded to ``QuantileRegressor``.
        ``"highs"`` (the default) is robust for most problem sizes.
    fit_intercept : bool
        Whether the sklearn estimator should fit its own intercept.
        Set to ``False`` if you supply a constant column yourself.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        alpha: float = 0.0,
        solver: str = "highs",
        fit_intercept: bool = True,
    ):
        self.quantile = quantile
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept

    # -- sklearn API --------------------------------------------------------

    def fit(self, X: xr.DataArray, y: xr.DataArray) -> XarrayQuantileRegressor:
        """Fit the quantile regression model.

        Parameters
        ----------
        X : xr.DataArray, shape (n_samples,) or (n_samples, n_features)
            Feature matrix.
        y : xr.DataArray, shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X_np = X.values
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        self.model_ = _SklearnQuantileRegressor(
            quantile=self.quantile,
            alpha=self.alpha,
            solver=self.solver,
            fit_intercept=self.fit_intercept,
        )
        self.model_.fit(X_np, y.values.ravel())
        self.n_features_in_ = X_np.shape[1]
        return self

    def predict(self, X: xr.DataArray) -> xr.DataArray:
        """Predict quantile values.

        Parameters
        ----------
        X : xr.DataArray
            Feature matrix (same schema as training *X*).

        Returns
        -------
        xr.DataArray
            Predicted quantile values with the leading dimension's
            coordinates preserved from *X*.
        """
        check_is_fitted(self, "model_")

        first_dim = X.dims[0]
        X_np = X.values
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        y_hat = self.model_.predict(X_np)
        return xr.DataArray(
            y_hat,
            dims=[first_dim],
            coords={first_dim: X.coords[first_dim]},
        )

    # -- introspection ------------------------------------------------------

    @property
    def coef_(self) -> np.ndarray:
        """Regression coefficients (delegates to the inner model)."""
        check_is_fitted(self, "model_")
        return self.model_.coef_

    @property
    def intercept_(self) -> float:
        """Intercept (delegates to the inner model)."""
        check_is_fitted(self, "model_")
        return self.model_.intercept_


# ---------------------------------------------------------------------------
# Layer 2: domain wrapper for peak-over-threshold analysis
# ---------------------------------------------------------------------------


def _time_to_numeric(time_values: np.ndarray) -> np.ndarray:
    """Convert datetime-like values to float days since first observation.

    Handles ``np.datetime64`` and already-numeric arrays.
    """
    if np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
        t0 = time_values[0]
        return (time_values - t0) / np.timedelta64(1, "D")
    return np.asarray(time_values, dtype=np.float64)


def _build_feature_matrix(
    da: xr.DataArray,
    time_dim: str,
    covariates: xr.DataArray | None = None,
) -> xr.DataArray:
    """Assemble the (n_samples, n_features) design matrix as a DataArray.

    Features are ``[t, cov_1, ..., cov_k]`` where *t* is numeric time
    and ``cov_i`` are optional covariate columns.  An intercept is *not*
    added here — the regressor handles that via ``fit_intercept=True``.

    Returns
    -------
    xr.DataArray
        Shape ``(n_time, n_features)`` with a ``"feature"`` dimension.
    """
    time_vals = da.coords[time_dim].values
    t_numeric = _time_to_numeric(time_vals)

    columns = [t_numeric]
    feature_names = ["time"]

    if covariates is not None:
        cov_np = covariates.values
        if cov_np.ndim == 1:
            columns.append(cov_np)
            feature_names.append("covariate_0")
        else:
            for i in range(cov_np.shape[1]):
                columns.append(cov_np[:, i])
                feature_names.append(f"covariate_{i}")

    X = np.column_stack(columns)
    return xr.DataArray(
        X,
        dims=[time_dim, "feature"],
        coords={time_dim: da.coords[time_dim], "feature": feature_names},
    )


def quantile_regression_threshold(
    da: xr.DataArray,
    quantile: float,
    time_dim: str,
    covariates: xr.DataArray | None = None,
    *,
    alpha: float = 0.0,
    solver: str = "highs",
) -> xr.DataArray:
    """Estimate a time-varying threshold via linear quantile regression.

    Fits the model

    .. math::

        Q_{\\tau}(y \\mid t, \\mathbf{z}) = \\beta_0
            + \\beta_1 t + \\boldsymbol{\\beta}_z^\\top \\mathbf{z}

    where *t* is numeric time and **z** are optional covariates, then
    returns the fitted quantile surface evaluated at the observed times.
    The result is suitable as a non-stationary threshold for
    peak-over-threshold (POT) extreme value analysis.

    Parameters
    ----------
    da : xr.DataArray
        Response variable (1-D along *time_dim*).
    quantile : float
        Target quantile τ ∈ (0, 1), e.g. 0.95 for the 95th percentile.
    time_dim : str
        Name of the time dimension in *da*.
    covariates : xr.DataArray, optional
        Additional covariates aligned along *time_dim*.
    alpha : float
        L1 regularisation passed to the regressor (0 = unregularised).
    solver : str
        LP solver, forwarded to ``QuantileRegressor``.

    Returns
    -------
    xr.DataArray
        Fitted threshold values with the same coords as *da*.

    Examples
    --------
    >>> # 95th-percentile linear-trend threshold
    >>> threshold = quantile_regression_threshold(da, 0.95, "time")

    >>> # With a covariate (e.g., sea-surface temperature)
    >>> threshold = quantile_regression_threshold(
    ...     da, 0.99, "time", covariates=sst
    ... )
    """
    X = _build_feature_matrix(da, time_dim, covariates)

    reg = XarrayQuantileRegressor(
        quantile=quantile,
        alpha=alpha,
        solver=solver,
        fit_intercept=True,
    )
    reg.fit(X, da)
    return reg.predict(X)


# ---------------------------------------------------------------------------
# Example: non-stationary POT on synthetic daily temperature
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    # -- 1. Build a synthetic daily temperature series ----------------------
    #
    # Model:  T(t) = seasonal(t) + trend(t) + noise(t)
    #
    #   seasonal : 15 °C amplitude cosine with period 365.25 days
    #   trend    : +2 °C over 20 years (mimics warming)
    #   noise    : Gaussian bulk  +  occasional heavy-tailed spikes
    #              (the spikes are the "extremes" we want to capture)

    rng = np.random.default_rng(42)
    n_years = 20
    time = pd.date_range("2000-01-01", periods=365 * n_years, freq="D")
    n = len(time)

    day_of_year = np.arange(n)
    seasonal = 20.0 + 15.0 * np.cos(2 * np.pi * day_of_year / 365.25)
    trend = np.linspace(0, 2.0, n)  # +2 °C over record
    bulk_noise = rng.normal(scale=3.0, size=n)
    spikes = rng.exponential(scale=2.0, size=n) * rng.binomial(1, 0.05, n)
    temperature = seasonal + trend + bulk_noise + spikes

    da = xr.DataArray(
        temperature,
        dims="time",
        coords={"time": time},
        attrs={"units": "°C", "long_name": "synthetic daily temperature"},
    )

    # -- 2. Fit a 95th-percentile threshold with linear trend ---------------

    tau = 0.95
    threshold = quantile_regression_threshold(da, quantile=tau, time_dim="time")

    # -- 3. Extract exceedances (peak-over-threshold) -----------------------

    exceedances = da.where(da > threshold, drop=True)
    excess_values = (da - threshold).where(da > threshold, drop=True)

    print("=" * 60)
    print("Non-stationary Peak-Over-Threshold example")
    print("=" * 60)
    print(f"Series length     : {n} days ({n_years} years)")
    print(f"Quantile (τ)      : {tau}")
    print(
        f"Threshold range   : {threshold.values[0]:.2f} → {threshold.values[-1]:.2f} °C"
    )
    print(
        f"N exceedances     : {exceedances.sizes['time']}  "
        f"({exceedances.sizes['time'] / n * 100:.1f}% of record)"
    )
    print(f"Mean excess       : {excess_values.mean().item():.2f} °C")
    print(f"Max excess        : {excess_values.max().item():.2f} °C")
    print()

    # -- 4. Inspect the fitted regressor directly ---------------------------

    X = _build_feature_matrix(da, "time")
    reg = XarrayQuantileRegressor(quantile=tau)
    reg.fit(X, da)

    print("Regressor coefficients:")
    print(f"  intercept : {reg.intercept_:.4f} °C")
    print(
        f"  β_time    : {reg.coef_[0]:.6f} °C/day  "
        f"({reg.coef_[0] * 365.25:.3f} °C/year)"
    )
    print()

    # -- 5. (Optional) quick plot if matplotlib is available ----------------

    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.plot(da.time, da, linewidth=0.3, color="0.6", label="temperature")
        ax1.plot(
            threshold.time,
            threshold,
            color="C3",
            linewidth=1.5,
            label=f"Q({tau}) threshold",
        )
        ax1.scatter(
            exceedances.time,
            exceedances,
            s=4,
            color="C3",
            zorder=3,
            label=f"exceedances (n={exceedances.sizes['time']})",
        )
        ax1.set_ylabel("Temperature [°C]")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_title("Non-stationary POT: quantile-regression threshold")

        ax2.stem(
            excess_values.time.values,
            excess_values.values,
            linefmt="C3-",
            markerfmt="C3o",
            basefmt="k-",
        )
        ax2.set_ylabel("Excess [°C]")
        ax2.set_xlabel("Time")

        plt.tight_layout()
        plt.savefig("pot_example.png", dpi=150)
        print("Plot saved to pot_example.png")
    except ImportError:
        print("(matplotlib not available — skipping plot)")
