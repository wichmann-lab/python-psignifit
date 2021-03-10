# -*- coding: utf-8 -*-
"""
get confidence intervals and region for parameters
function [conf_Intervals, confRegion]=getConfRegion(result)
This function returns the conf_intervals for all parameters and a
confidence region on the whole parameter space.

Useage
pass the result obtained from psignifit
additionally in confP the confidence/ the p-value for the interval is required
finally you can specify in CImethod how to compute the intervals
      'project' -> project the confidence region down each axis
      'stripes' -> find a threshold with (1-alpha) above it
  'percentiles' -> find alpha/2 and 1-alpha/2 percentiles
                   (alpha = 1-confP)

confP may also be a vector of confidence levels. The returned CIs
are a 5x2xN array then containting the confidence intervals at the different
confidence levels. (any shape of confP will be interpreted as a vector)

"""
# THIS FILE IS LEGACY AND WILL BE REMOVED. NO STYLE CHECKING HERE
# flake8: noqa

####TODO: 'project' should be default, is a HDI (highest density interval) method
####      'percentiles' should be an option
####      'stripes' can be removed
### R implementation of project:
# HDIofGrid = function( probMassVec , credMass=0.95 ) {
# # Arguments:
# #   probMassVec is a vector of probability masses at each grid point.
# #   credMass is the desired mass of the HDI region.
# # Return value:
# #   A list with components:
# #   indices is a vector of indices that are in the HDI
# #   mass is the total mass of the included indices
# #   height is the smallest component probability mass in the HDI
# # Example of use: For determining HDI of a beta(30,12) distribution
# #   approximated on a grid:
# #   > probDensityVec = dbeta( seq(0,1,length=201) , 30 , 12 )
# #   > probMassVec = probDensityVec / sum( probDensityVec )
# #   > HDIinfo = HDIofGrid( probMassVec )
# #   > show( HDIinfo )
# sortedProbMass = sort( probMassVec , decreasing=TRUE )
# HDIheightIdx = min( which( cumsum( sortedProbMass ) >= credMass ) )
# HDIheight = sortedProbMass[ HDIheightIdx ]
# HDImass = sum( probMassVec[ probMassVec >= HDIheight ] )
# return( list( indices = which( probMassVec >= HDIheight ) ,
# mass = HDImass , height = HDIheight ) )
# }
from typing import Sequence

import numpy as np
from scipy import stats

from .marginalize import marginalize


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
    CI_METHODS = { 'project': grid_hdi, 'percentiles': percentile_intervals }
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
        interval_ix = projected_region.reshape(-1).nonzero()[[0, -1]]
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


def getConfRegion(result):
    mode = result['options']['CImethod']
    d = len(result['X1D'])
    ''' get confidence intervals for each parameter --> marginalize'''
    conf_Intervals = np.zeros((d, 2, len(result['options']['confP'])))
    confRegion = 0
    i = 0
    for iConfP in result['options']['confP']:

        if mode == 'project':
            order = np.array(result['Posterior'][:]).argsort()[::-1]
            Mass = result['Posterior'] * result['weight']
            Mass = np.cumsum(Mass[order])

            confRegion = np.reshape(
                np.array([True] * np.size(result['Posterior']),
                         result['Posterior'].shape))
            confRegion[order[Mass >= iConfP]] = False
            for idx in range(0, d):
                confRegionM = confRegion
                for idx2 in range(0, d):
                    if idx != idx2:
                        confRegionM = np.any(confRegionM, idx2)
                start = result['X1D'][idx].flatten().nonzero()[0][0]
                stop = result['X1D'][idx].flatten().nonzero()[0][-1]
                conf_Intervals[idx, :, i] = [start, stop]
        elif mode == 'stripes':
            for idx in range(0, d):
                (margin, x, weight1D) = marginalize(result, idx)
                order = np.array(margin).argsort()[::-1]

                Mass = margin * weight1D
                MassSort = np.cumsum(Mass[order])
                # find smallest possible percentage above confP
                confP1 = min(MassSort[MassSort > iConfP])
                confRegionM = np.reshape(
                    np.array([True] * np.size(margin), np.shape(margin)))
                confRegionM[order[MassSort > confP1]] = False
                '''
                Now we have the confidence regions
                put the bounds between the nearest contained and the first
                not contained point

                we move in from the outer points to collect the half of the
                leftover confidence from each side
                '''
                startIndex = confRegionM.flatten().nonzero()[0][0]
                pleft = confP1 - iConfP
                if startIndex > 1:
                    start = (x[startIndex] + x[startIndex - 1]) / 2
                    start += pleft / 2 / margin[startIndex]
                else:
                    start = x[startIndex]
                    pleft *= 2

                stopIndex = confRegionM.flatten().nonzero()[0][-1]
                if stopIndex < len(x):
                    stop = (x[stopIndex] + x[stopIndex + 1]) / 2
                    stop -= pleft / 2 / margin[stopIndex]
                else:
                    stop = x[stopIndex]
                    if startIndex > 1:
                        start += pleft / 2 / margin[startIndex]
                conf_Intervals[idx, :, i] = [start, stop]
        elif mode == 'percentiles':
            for idx in range(0, d):
                (margin, x, weight1D) = marginalize(result, idx)
                if len(x) == 1:
                    start = x[0]
                    stop = x[0]
                else:
                    Mass = margin * weight1D
                    cumMass = np.cumsum(Mass)

                    confRegionM = np.logical_and(
                        cumMass > (1 - iConfP) / 2, cumMass <
                        (1 - (1 - iConfP) / 2))
                    if any(confRegionM):
                        alpha = (1 - iConfP) / 2
                        startIndex = confRegionM.flatten().nonzero()[0][0]
                        if startIndex > 0:
                            start = (x[startIndex - 1] + x[startIndex]) / 2 + (alpha - cumMass[startIndex - 1]) / \
                                    margin[startIndex]
                        else:
                            start = x[startIndex] + alpha / margin[startIndex]

                        stopIndex = confRegionM.flatten().nonzero()[0][-1]
                        if stopIndex < len(x):
                            stop = (x[stopIndex] + x[stopIndex + 1]) / 2 + (
                                1 - alpha -
                                cumMass[stopIndex]) / margin[stopIndex + 1]
                        else:
                            stop = x[stopIndex] - alpha / margin[stopIndex]
                    else:
                        cumMass_greq_iConfP = np.array(
                            cumMass > (1 - iConfP) / 2)
                        index = cumMass_greq_iConfP.flatten().nonzero()[0][0]
                        MMid = cumMass[index] - Mass[index] / 2
                        start = x[index] + (
                            (1 - iConfP) / 2 - MMid) / margin[index]
                        stop = x[index] + (
                            (1 - (1 - iConfP) / 2) - MMid) / margin[index]
                conf_Intervals[idx, :, i] = np.array([start, stop])

        else:
            raise ValueError('You specified an invalid mode')

        i += 1
    return conf_Intervals


if __name__ == "__main__":
    import sys

    getConfRegion(sys.argv[1], sys.argv[2])
