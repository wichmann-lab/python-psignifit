# -*- coding: utf-8 -*-
"""
Confidence intervals and region for parameters

Two methods are implemented:
      'project' -> project the confidence region down each axis
  'percentiles' -> find alpha/2 and 1-alpha/2 percentiles
                   (alpha = 1-confP)
"""
from typing import Sequence

import numpy as np
from scipy import stats


def confidence_intervals(probability_mass: np.ndarray, grid_values: np.ndarray,
                         p_values: Sequence[float], mode: str) -> np.ndarray:
    """ Confidence intervals on probability grid.

    Supports two methods:

        - 'project', projects the confidence region down each axis.
          Implemented using :func:`grid_hdi`.
        - 'percentiles', finds alpha/2 and 1-alpha/2 percentiles (alpha = 1-p_value).
          Implemented using :func:`percentile_intervals`.

    Args:
        probability_mass: Probability mass at each grid point, shape (n_points, n_points, ...)
        grid_values: Parameter values along grid axis in the same order as zerocentered_normal_mass dimensions,
                     shape (n_dims, n_points)
        p_values: Probabilities of confidence in the intervals.
        mode: Either 'project' or 'percentiles'.
    Returns:
        Start and end grid-values for the confidence interval
        per dimension and p_value, shape (n_dims, n_p_values, 2)
    Raises:
        ValueError for unsupported mode or sum(probability_mass) != 1.
     """
    CI_METHODS = {'project': grid_hdi, 'percentiles': percentile_intervals}
    if mode in CI_METHODS:
        calc_ci = CI_METHODS[mode]
    elif mode == 'stripes':
        raise ValueError('Confidence mode "stripes" is not supported anymore. "'
                         f'"Use one of {list(CI_METHODS.keys())}.')
    else:
        raise ValueError(f'Expects mode as one of {list(CI_METHODS.keys())}, got {mode}')

    if not np.isclose(probability_mass.sum(), 1):
        raise ValueError(f'Expects sum(probability_mass) to be 1., got {probability_mass.sum():.4f}')
    intervals = np.empty((probability_mass.ndim, len(p_values), 2))
    for p_ix, p_value in enumerate(p_values):
        intervals[:, p_ix, :] = calc_ci(probability_mass, grid_values, p_value)

    return intervals


def grid_hdi(probability_mass: np.ndarray, grid_values: np.ndarray, credible_mass: float) -> np.ndarray:
    """ Highest density intervals (hdi) on a grid.

    Calculates the grid region, in which the probability mass
    is at least the credible mass (highest density region, hdr).
    This region is projected to the individual dimensions.
    The intervals range from the first to the last grid entry
    in the projected hdr.

    See `stats.stackexchange` for an explanation of the method.

    Args:
        probability_mass: Probability mass at each grid point, shape (n_points, n_points, ...)
        grid_values: Parameter values along grid axis, shape (n_dims, n_points)
        credible_mass: Minimal mass in highest density region
    Returns:
        Grid value at interval per dimension, shape (n_dims, 2)

    .. _stats.stackexchange: https://stats.stackexchange.com/questions/148439/what-is-a-highest-density-region-hdr
    """
    decreasing_mass = np.sort(probability_mass.reshape(-1))[::-1]
    atleast_height_ix = np.argwhere(decreasing_mass.cumsum() >= credible_mass)
    if len(atleast_height_ix) == 0:
        raise ValueError(f"Expects credible mass < sum(mass), got {credible_mass} > {np.sum(probability_mass)}")
    hd_height_ix = atleast_height_ix.min()
    hd_region = probability_mass >= decreasing_mass[hd_height_ix]

    dims = np.arange(hd_region.ndim)
    intervals = np.empty((hd_region.ndim, 2))
    for d in dims:
        projected_region = hd_region.any(axis=tuple(dims[dims != d]))
        interval_ix = projected_region.reshape(-1).nonzero()[0][[0, -1]]
        intervals[d, :] = grid_values[d][interval_ix]
    return intervals


def percentile_intervals(probability_mass: np.ndarray, grid_values: np.ndarray, p_value: float):
    """ Percentile intervals on a grid.

    Find alpha/2 and 1-alpha/2 percentiles on marginal probability mass (alpha = 1 - p_value).

    Args:
        probability_mass: Probability mass at each grid point, shape (n_points, n_points, ...)
        grid_values: Parameter values along grid axis, shape (n_dims, n_points)
        p_value: Probability mass within the returned confidence bounds.
    Returns:
        Grid value at interval per dimension, shape (n_dims, 2)
    """
    mass_margins = stats.contingency.margins(probability_mass)
    intervals = np.empty((len(mass_margins), 2))
    alpha = 1 - p_value
    for d, mass in enumerate(mass_margins):
        intervals[d, :] = np.interp([alpha / 2, 1 - alpha / 2], mass.cumsum(), grid_values[d])
    return intervals
