from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr


def temporal_block_maxima(
    da: xr.DataArray, freq: str, time_dim: str = "time", min_periods: int | None = None
) -> xr.DataArray:
    """
    Extract temporal block maximum values from an xarray DataArray.

    Uses time-based resampling to compute maximum values over temporal blocks.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    freq : str
        Frequency string for temporal grouping (e.g., '1Y' for annual,
        '1M' for monthly, '7D' for weekly, '3H' for 3-hourly)
    time_dim : str, default 'time'
        Name of the time dimension
    min_periods : int, optional
        Minimum number of observations required for a valid maximum.
        If None, all blocks are used regardless of missing data.

    Returns
    -------
    xr.DataArray
        Temporal block maximum values

    Examples
    --------
    >>> # Annual maxima
    >>> annual_max = temporal_block_maxima(da, '1Y')

    >>> # Monthly maxima with at least 20 observations per month
    >>> monthly_max = temporal_block_maxima(da, '1M', min_periods=20)

    >>> # Seasonal maxima (3-month blocks)
    >>> seasonal_max = temporal_block_maxima(da, '3M')
    """
    resampler = da.resample({time_dim: freq})

    if min_periods is not None:
        # Count valid observations in each block
        counts = resampler.count(dim=time_dim)
        maxima = resampler.max(dim=time_dim, keep_attrs=True)
        # Mask blocks with insufficient data
        maxima = maxima.where(counts >= min_periods)
        return maxima
    else:
        return resampler.max(dim=time_dim, keep_attrs=True)


def spatial_block_maxima(
    da: xr.DataArray,
    block_size: int | dict[Hashable, int],
    dims: Hashable | Sequence[Hashable] = None,
    min_periods: int | None = None,
    boundary: str = "trim",
) -> xr.DataArray:
    """
    Extract spatial block maximum values from an xarray DataArray.

    Uses coarsening to compute maximum values over spatial blocks of fixed size.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    block_size : int or dict
        Size of spatial blocks. If int, uses same size for all specified dims.
        If dict, maps dimension names to block sizes (e.g., {'x': 10, 'y': 10})
    dims : Hashable or Sequence[Hashable], optional
        Dimension(s) to coarsen. If None and block_size is int,
        raises error. If None and block_size is dict, uses dict keys.
    min_periods : int, optional
        Minimum number of valid observations required within each block.
        If None, computes maximum even with missing data.
    boundary : str, default 'trim'
        How to handle boundaries. Options: 'trim', 'pad', 'exact'

    Returns
    -------
    xr.DataArray
        Spatial block maximum values with reduced resolution

    Examples
    --------
    >>> # 10x10 spatial blocks
    >>> spatial_max = spatial_block_maxima(da, {'x': 10, 'y': 10})

    >>> # 5x5 blocks along lat/lon dimensions
    >>> spatial_max = spatial_block_maxima(da, 5, dims=['lat', 'lon'])

    >>> # Require at least 50 valid points per block
    >>> spatial_max = spatial_block_maxima(
    ...     da, {'x': 10, 'y': 10}, min_periods=50
    ... )
    """
    # Handle block_size specification
    if isinstance(block_size, dict):
        coarsen_kwargs = block_size
    else:
        if dims is None:
            raise ValueError("Must specify 'dims' when block_size is an integer")
        if isinstance(dims, str):
            dims = [dims]
        coarsen_kwargs = {dim: block_size for dim in dims}

    # Create coarsen object
    coarsened = da.coarsen(dim=coarsen_kwargs, boundary=boundary)

    if min_periods is not None:
        # Count valid observations in each block
        counts = coarsened.count()
        maxima = coarsened.max(keep_attrs=True)
        # Mask blocks with insufficient data
        maxima = maxima.where(counts >= min_periods)
        return maxima
    else:
        return coarsened.max(keep_attrs=True)


def sliding_block_maxima(
    da: xr.DataArray,
    window_size: int,
    dim: str = "time",
    stride: int = 1,
    min_periods: int | None = None,
    center: bool = False,
) -> xr.DataArray:
    """
    Extract block maxima using a sliding window approach.

    Sliding windows provide overlapping blocks, which helps:
    - Avoid sensitivity to arbitrary block boundaries
    - Increase the number of block maxima for better statistical estimation
    - Capture extreme events that might span fixed block boundaries

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    window_size : int
        Size of the sliding window (number of observations)
    dim : str, default 'time'
        Dimension along which to slide the window
    stride : int, default 1
        Step size between windows. stride=1 gives fully overlapping windows,
        stride=window_size gives non-overlapping blocks (equivalent to coarsen)
    min_periods : int, optional
        Minimum number of valid observations required for a valid maximum.
        If None, defaults to window_size.
    center : bool, default False
        If True, set the window labels at the center of the window

    Returns
    -------
    xr.DataArray
        Sliding block maximum values

    Examples
    --------
    >>> # 365-day sliding window with daily stride
    >>> sliding_max = sliding_block_maxima(da, window_size=365, stride=1)

    >>> # Non-overlapping 30-day blocks (equivalent to fixed blocks)
    >>> block_max = sliding_block_maxima(da, window_size=30, stride=30)

    >>> # 90-day window with 7-day stride (weekly updates)
    >>> weekly_max = sliding_block_maxima(da, window_size=90, stride=7)
    """
    if min_periods is None:
        min_periods = window_size

    # Use rolling window
    rolling = da.rolling({dim: window_size}, center=center, min_periods=min_periods)
    maxima = rolling.max()

    # Apply stride by subsampling
    if stride > 1:
        maxima = maxima.isel({dim: slice(None, None, stride)})

    return maxima


def declustered_block_maxima(
    da: xr.DataArray,
    threshold: float,
    min_separation: int,
    dim: str = "time",
    method: str = "runs",
) -> xr.DataArray:
    """
    Extract declustered block maxima by identifying independent extreme events.

    Declustering removes temporal dependence in extremes by ensuring sufficient
    separation between events. This is crucial for extreme value theory which
    assumes independence.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float
        Threshold value for identifying extreme events
    min_separation : int
        Minimum separation (in time steps) between independent events
    dim : str, default 'time'
        Dimension along which to decluster
    method : str, default 'runs'
        Declustering method:
        - 'runs': Select maximum from each exceedance run
        - 'separation': Enforce minimum time separation between peaks

    Returns
    -------
    xr.DataArray
        Declustered block maxima with reduced temporal dependence

    Examples
    --------
    >>> # Extract independent storm maxima (3-day separation)
    >>> storm_max = declustered_block_maxima(
    ...     da, threshold=100, min_separation=3, method='runs'
    ... )

    >>> # Heat wave maxima with 7-day separation
    >>> heatwave_max = declustered_block_maxima(
    ...     da, threshold=35, min_separation=7, method='separation'
    ... )

    Notes
    -----
    The runs method identifies continuous periods above threshold and takes
    the maximum from each run. The separation method ensures peaks are
    separated by at least min_separation time steps.
    """
    # Identify exceedances
    exceedances = da > threshold

    if method == "runs":
        # Label contiguous runs above threshold
        # Note: This is a simplified version - full implementation would use
        # scipy.ndimage.label or similar for multi-dimensional arrays
        runs = (exceedances != exceedances.shift({dim: 1})).cumsum(dim=dim)
        runs = runs.where(exceedances)

        # Get maximum value from each run
        maxima = da.groupby(runs).max(dim=dim, skipna=True)
        return maxima

    elif method == "separation":
        # Delegate to the separation-based declustering helper, which
        # actually applies `min_separation` in units of original time steps.
        from xtremax.extraction.decluster import decluster_separation

        return decluster_separation(
            da, threshold=threshold, min_separation=min_separation, dim=dim
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'runs' or 'separation'")


def r_largest_block_maxima(
    da: xr.DataArray,
    block_size: int | str,
    r: int = 3,
    dim: str = "time",
    min_periods: int | None = None,
) -> xr.DataArray:
    """
    Extract r-largest order statistics from each block.

    Instead of just the maximum, extract the r largest values from each block.
    This provides more information for extreme value analysis and can improve
    statistical efficiency.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    block_size : int or str
        Size of blocks. If int, number of elements. If str, frequency string.
    r : int, default 3
        Number of largest values to extract from each block
    dim : str, default 'time'
        Dimension along which to compute order statistics
    min_periods : int, optional
        Minimum number of observations required in a block

    Returns
    -------
    xr.DataArray
        Array with r-largest values from each block, with new dimension 'order'

    Examples
    --------
    >>> # 3 largest values from each year
    >>> annual_r_largest = r_largest_block_maxima(da, '1Y', r=3)

    >>> # Top 5 values from 100-element blocks
    >>> block_r_largest = r_largest_block_maxima(da, 100, r=5)

    Notes
    -----
    The r-largest order statistics model is useful when you want to use
    more information than just the block maximum for fitting extreme
    value distributions.
    """

    def get_r_largest(block):
        """Extract r largest values from a block.

        Accepts both xarray DataArrays (groupby path) and plain numpy
        arrays (fixed-size reshape path), since ``np.asarray`` on a
        DataArray materializes its values.
        """
        sorted_block = np.sort(np.asarray(block).flatten())
        # Remove NaNs
        sorted_block = sorted_block[~np.isnan(sorted_block)]

        if min_periods is not None and len(sorted_block) < min_periods:
            return np.full(r, np.nan)

        if len(sorted_block) < r:
            # Pad with NaN if insufficient data
            result = np.full(r, np.nan)
            result[: len(sorted_block)] = sorted_block[-len(sorted_block) :][::-1]
            return result

        return sorted_block[-r:][::-1]  # Return in descending order

    if isinstance(block_size, str):
        # Time-based resampling
        groups = da.resample({dim: block_size})

        # Apply r-largest extraction to each group
        r_largest_list = []
        group_labels = []

        for label, group in groups:
            r_vals = get_r_largest(group)
            r_largest_list.append(r_vals)
            group_labels.append(label)

        # Construct output array
        result = xr.DataArray(
            np.array(r_largest_list),
            dims=[dim, "order"],
            coords={dim: group_labels, "order": np.arange(1, r + 1)},
            attrs=da.attrs,
        )
        return result
    else:
        # Fixed-size blocks - use reshape and apply
        # This is a simplified version for 1D case
        n_blocks = len(da[dim]) // block_size
        trimmed_length = n_blocks * block_size
        trimmed = da.isel({dim: slice(0, trimmed_length)})

        # Reshape into blocks
        new_shape = (n_blocks, block_size, *trimmed.shape[1:])
        reshaped = trimmed.values.reshape(new_shape)

        # Get r-largest from each block
        r_largest = np.array([get_r_largest(block) for block in reshaped])

        # Create coordinate for block centers
        block_coords = da[dim].values[:trimmed_length:block_size]

        result = xr.DataArray(
            r_largest,
            dims=["block", "order"],
            coords={"block": block_coords, "order": np.arange(1, r + 1)},
            attrs=da.attrs,
        )
        return result
