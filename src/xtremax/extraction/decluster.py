from __future__ import annotations

import numpy as np
import xarray as xr


def decluster_runs(
    da: xr.DataArray,
    threshold: float | xr.DataArray,
    dim: str = "time",
    reduction: str = "max",
) -> xr.DataArray:
    """
    Decluster exceedances using the runs method.

    Identifies continuous runs of exceedances and applies a reduction
    function to each run. This ensures temporal independence by treating
    each cluster as a single event.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float or xr.DataArray
        Threshold value(s)
    dim : str, default 'time'
        Dimension along which to identify runs
    reduction : str, default 'max'
        Reduction function to apply to each run: 'max', 'mean', 'sum', 'min'

    Returns
    -------
    xr.DataArray
        Declustered values, one per run

    Examples
    --------
    >>> # Maximum value from each exceedance cluster
    >>> declustered = decluster_runs(da, threshold=100, reduction='max')

    >>> # Mean value during each exceedance period
    >>> declustered = decluster_runs(da, threshold=100, reduction='mean')

    Notes
    -----
    The runs method identifies continuous periods where values exceed
    the threshold. Each continuous period (run) is treated as one
    cluster, and we extract one representative value per cluster.
    """
    # Identify exceedances
    exceedances = da > threshold

    # Label contiguous runs
    # When exceedance status changes, start a new run
    run_boundaries = exceedances != exceedances.shift({dim: 1})
    run_ids = run_boundaries.cumsum(dim=dim)

    # Only keep run IDs where there are exceedances
    run_ids = run_ids.where(exceedances)

    # Apply reduction function to each run
    if reduction == "max":
        result = da.groupby(run_ids).max(dim=dim, skipna=True)
    elif reduction == "mean":
        result = da.groupby(run_ids).mean(dim=dim, skipna=True)
    elif reduction == "sum":
        result = da.groupby(run_ids).sum(dim=dim, skipna=True)
    elif reduction == "min":
        result = da.groupby(run_ids).min(dim=dim, skipna=True)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return result


def decluster_separation(
    da: xr.DataArray,
    threshold: float | xr.DataArray,
    min_separation: int,
    dim: str = "time",
) -> xr.DataArray:
    """
    Decluster by enforcing minimum separation between peaks.

    Identifies peaks above threshold and ensures they are separated by
    at least min_separation time steps. When peaks are too close, keeps
    the larger one.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float or xr.DataArray
        Threshold value(s)
    min_separation : int
        Minimum separation (in time steps) between independent peaks
    dim : str, default 'time'
        Dimension along which to enforce separation

    Returns
    -------
    xr.DataArray
        Declustered peak values

    Examples
    --------
    >>> # Peaks separated by at least 3 days
    >>> declustered = decluster_separation(da, threshold=100, min_separation=3)

    >>> # Storm events with 7-day minimum separation
    >>> storms = decluster_separation(da, threshold=95, min_separation=7)

    Notes
    -----
    This method first identifies all local maxima above threshold,
    then iteratively removes peaks that are too close to larger peaks.
    """
    # Identify local maxima above threshold
    exceedances = da > threshold
    is_peak = (da > da.shift({dim: 1})) & (da > da.shift({dim: -1})) & exceedances

    # Peak positions along the original time axis — these are what
    # `min_separation` is measured against (as time steps, per the docstring),
    # not the index of a peak within the reduced peaks array.
    peak_positions = np.flatnonzero(is_peak.values)
    if peak_positions.size == 0:
        return da.isel({dim: peak_positions})

    peak_values = da.values[peak_positions]

    # Sort peaks by value (descending) and greedily keep peaks that are far
    # enough, in original time steps, from every peak already selected.
    order = np.argsort(peak_values)[::-1]
    sorted_positions = peak_positions[order]

    selected_positions: list[int] = []
    for pos in sorted_positions:
        if all(abs(int(pos) - s) >= min_separation for s in selected_positions):
            selected_positions.append(int(pos))

    selected_positions.sort()
    return da.isel({dim: np.asarray(selected_positions, dtype=int)})


def estimate_extremal_index(
    da: xr.DataArray,
    threshold: float | xr.DataArray,
    dim: str = "time",
    method: str = "runs",
) -> float:
    """
    Estimate the extremal index (theta), a measure of clustering.

    The extremal index θ ∈ [0, 1] measures the degree of clustering in extremes:
    - θ = 1: no clustering (independent extremes)
    - θ < 1: clustering present (dependent extremes)
    - θ = 0: complete clustering (perfect dependence)

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float or xr.DataArray
        Threshold value for identifying extremes
    dim : str, default 'time'
        Dimension along which to estimate clustering
    method : str, default 'runs'
        Estimation method: 'runs' or 'intervals'

    Returns
    -------
    float
        Extremal index estimate (between 0 and 1)

    Examples
    --------
    >>> # Estimate clustering strength
    >>> theta = estimate_extremal_index(da, threshold=100)
    >>> print(f"Extremal index: {theta:.3f}")
    >>> # theta ≈ 0.7 suggests moderate clustering

    >>> # Use high threshold (99th percentile)
    >>> from extremes.threshold import quantile_threshold
    >>> thresh = quantile_threshold(da, 0.99)
    >>> theta = estimate_extremal_index(da, thresh)

    Notes
    -----
    The runs estimator uses: θ = (number of clusters) / (number of exceedances)
    A smaller θ indicates more clustering.
    """
    # Count total exceedances
    exceedances = da > threshold
    n_exceedances = int(exceedances.sum().values)

    if n_exceedances == 0:
        return np.nan

    if method == "runs":
        # Count number of runs (clusters)
        run_boundaries = exceedances != exceedances.shift({dim: 1})
        run_ids = run_boundaries.cumsum(dim=dim)
        run_ids_masked = run_ids.where(exceedances)

        # Number of unique runs
        n_runs = len(np.unique(run_ids_masked.values[~np.isnan(run_ids_masked.values)]))

        # Extremal index estimate
        theta = n_runs / n_exceedances

        return theta

    elif method == "intervals":
        # Intervals estimator (more sophisticated, placeholder)
        # Would need inter-exceedance times distribution
        raise NotImplementedError("Intervals method not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")
