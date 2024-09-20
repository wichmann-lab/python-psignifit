# -*- coding: utf-8 -*-
from typing import Dict, Tuple

import numpy as np
import scipy.special as sp
from scipy import optimize

from ._utils import fp_error_handler, PsignifitException
from ._typing import Prior, ParameterGrid
from .sigmoids import Sigmoid


def integral_weights(grid):
    """Calculate integral of multivariate function using composite trapezoidal rule

    Input parameters:
       -  `func` is an array of dimensions n_1 x n_2 x ... x n_m
       - `grid` is a tuple (s_1, s_2, ..., s_m), where `s_i` are the points
          on dimension `i` along which `func` has been evaluated

    Outputs;
       - integral is a number
       - `deltas` is the grid of deltas used for the integration, for each
         dimension these are:
         (x1-x0)/2, x1-x0, x2-x1, ..., x(m-1)-x(m-2), (xm-x(m-1))/2
         `weights` has the same shape as `func`
    """
    deltas = []
    for steps in grid:
        # handle singleton dimensions
        if steps is None or len(steps) <= 1:
            deltas.append(1)
        else:
            delta = np.empty_like(steps, dtype=float)
            delta[1:] = np.diff(steps)
            # delta weight is half at the bounds of the integration interval
            delta[0] = delta[1] / 2
            delta[-1] = delta[-1] / 2
            deltas.append(delta)

    # create a meshgrid for each dimension
    weights = 1
    for param_weights in np.meshgrid(*deltas, copy=False, sparse=True, indexing='ij'):
        weights = weights * param_weights
    return weights


def posterior_grid(data, sigmoid: Sigmoid, priors: Dict[str, Prior],
                   grid: ParameterGrid) -> Tuple[np.ndarray, Dict[str, float]]:
    """ Finds the parameters which maximize the log posterior_grid in grid of parameter values.

    The objective is described in :func:`psignifit.likelihood.log_posterior:.

    Args:
        data: Numpy array with three columns for the stimulus levels, the correct trials, and the total trials.
        param_init: Initial value per parameter to optimize.
        param_fixed: Fixed value per parameter not to optimize.
        sigmoid: Sigmoid function
        priors: Prior function per parameter
        Grid: Dictionary of parameter name and numpy array with possible parameter values.
    Returns:
        p: Numpy array with a posterior value for each entry in the grid.
           The axis correspond to the the grid entries in alphabetic order.
           If the grid entry is None there is no corresponding axis.
        grid_max: Dictionary of parameter names with the parameter values,
                  which maximize the posterior_grid.
    """
    p = log_posterior(data, sigmoid=sigmoid, priors=priors, grid=grid)

    ind_max = np.unravel_index(p.argmax(), p.shape)
    p = np.exp(p - p[ind_max])

    grid_max = dict()
    for index, (name, grid_values) in zip(ind_max, sorted(grid.items())):
        if grid_values is None:
            grid_max[name] = None
        else:
            grid_max[name] = grid_values[index]

    # normalize the posterior_grid
    weights = integral_weights([grid_value for _, grid_value in sorted(grid.items())])
    posterior_volumes = p * weights
    posterior_integral = posterior_volumes.sum()
    posterior_mass = posterior_volumes / posterior_integral
    return posterior_mass, grid_max


@fp_error_handler(divide='ignore')  # noqa: C901, function is too complex
def log_posterior(data: np.ndarray, sigmoid: Sigmoid, priors: Dict[str, Prior], grid: ParameterGrid) -> np.ndarray:
    """ Estimate the logarithmic posterior_grid probability of sigmoid parameters.

    The function estimates the log-posterior_grid for a grid of sigmoid parameters and adds the corresponding
    log-priors. This is described in formulas A.4 and A.5 of [Schuett2016]_.
    The grid entry for gamma can be None to constraint lambda = gamma as in equal asymptote experiments.

    Args:
        data: numpy array with three columns for the stimulus levels, the correct trials, and the total trials.
        sigmoid: sigmoid function with three input parameter, see :module:`~psignifit.sigmoids`.
        priors: dictionary of parameter names and corresponding prior function, see :module:`~psignifit.priors`.
        grid: dictionary of parameter name and numpy array with parameter values.
    Returns:
        Numpy array with the log-posterior_grid for each parameter combination in grid.
        The axis correspond to the the grid entries in alphabetic order.
        If the grid entry is None there is no corresponding axis.

    .. [Schuett2016] Schütt, H. H., Harmeling, S., Macke, J. H. and Wichmann, F. A. (2016).
          Painfree and accurate Bayesian estimation of psychometric functions for (potentially) overdispersed data.
          Vision Research 122, 105-123.
    """
    thres = grid['threshold']
    width = grid['width']
    lambd = grid['lambda']
    gamma = grid['gamma']
    eta_std = grid['eta']

    if np.allclose(eta_std, [0]):
        eta_prime = None
    else:
        eta = eta_std ** 2  # use variance instead of standard deviation
        eta_prime = 1 / eta[eta > 1e-09] - 1
        eta_binom = (eta <= 1e-09).sum()  # for small variance we use the binomial model

    levels = data[:, 0]
    ncorrect = data[:, 1]
    ntrials = data[:, 2]

    if gamma is None and eta_prime is None:
        lambd, thres, width = np.meshgrid(lambd, thres, width, copy=False, sparse=True, indexing='ij')
    elif gamma is None:
        eta_prime, lambd, thres, width = np.meshgrid(eta_prime, lambd, thres, width,
                                                     copy=False, sparse=True, indexing='ij')
    elif eta_prime is None:
        gamma, lambd, thres, width = np.meshgrid(gamma, lambd, thres, width, copy=False, sparse=True, indexing='ij')
    else:
        eta_prime, gamma, lambd, thres, width = np.meshgrid(eta_prime, gamma, lambd, thres, width,
                                                            copy=False, sparse=True, indexing='ij')

    if gamma is None:
        gamma = lambd
    scale = 1 - gamma - lambd

    pbin = 0
    p = 0
    # we could get rid of the loop by playing crazy fancy-indexing tricks
    # with the arrays here, but that version would be quite more unreadable.
    # as we don't expect huge number of levels, we should not lose too much
    # performance by using an explicit, but no profiling has been done to prove
    # this.
    for level, trials, correct_trials in zip(levels, ntrials, ncorrect):
        if trials == 0:
            continue

        psi = sigmoid(level, thres, width) * scale + gamma

        # Separate cases to avoid warnings problems with np.log(...)
        # for correct_trials == 0 or (trials - correct_trials) == 0
        if correct_trials == 0:
            pbin += (trials - correct_trials) * np.log(1 - psi)
        elif correct_trials == trials:
            pbin += correct_trials * np.log(psi)
        elif correct_trials < trials:
            pbin += correct_trials * np.log(psi) + (trials - correct_trials) * np.log(1 - psi)
        else:  # we should never land here: we can't more ncorrect than ntrials
            raise PsignifitException('ncorrect %d > ntrials %d!' % (correct_trials, trials))

        if eta_prime is not None:
            a = psi * eta_prime
            b = (1 - psi) * eta_prime
            # Separate cases to avoid numerical problems with sp.gammaln(...) == inf
            # for correct_trials == 0 or (trials - correct_trials) == 0
            if correct_trials == 0:
                p += (sp.gammaln(trials + b) - sp.gammaln(trials + eta_prime) - sp.gammaln(b) + sp.gammaln(eta_prime))
            elif correct_trials == trials:
                p += (sp.gammaln(correct_trials + a) - sp.gammaln(trials + eta_prime) - sp.gammaln(a) +
                      sp.gammaln(eta_prime))
            elif correct_trials < trials:
                p += (sp.gammaln(correct_trials + a) + sp.gammaln(trials - correct_trials + b) -
                      sp.gammaln(trials + eta_prime) - sp.gammaln(a) - sp.gammaln(b) +
                      sp.gammaln(eta_prime))
            else:  # we should never land here: we can't more ncorrect than ntrials
                raise PsignifitException('ncorrect %d > ntrials %d!' % (correct_trials, trials))

    if eta_prime is None:
        p = pbin
    else:
        # eta_prime is on axis=0
        pbin = np.broadcast_to(pbin, (eta_binom,) + pbin.shape[1:])
        p = np.concatenate((pbin, p), axis=0)

    # add priors on the corresponding axis
    p += np.log(priors['threshold'](thres))
    p += np.log(priors['width'](width))
    p += np.log(priors['lambda'](lambd))
    p += np.log(priors['gamma'](gamma))
    if eta_prime is None:
        p = np.expand_dims(p, axis=0)
    else:
        p += np.log(priors['eta'](eta_std.reshape(-1, *eta_prime.shape[1:])))
    if grid['gamma'] is None:
        p = np.expand_dims(p, axis=1)
    return p


def maximize_posterior(data, param_init: Dict[str, float], param_fixed: Dict[str, float],
                       sigmoid: Sigmoid, priors: Dict[str, Prior]) -> Dict[str, float]:
    """ Finds the parameters which maximize the log posterior_grid using hill climbing.

     Parameters with None or a single entry in `grid` are treated as fixed.
     The objective is described in :func:`psignifit.likelihood.log_posterior:.
     It is optimized by :func:`scipy.optimize.fmin` with default settings,
     using the downhill simplex algorithm.

     Args:
         data: numpy array with three columns for the stimulus levels, the correct trials, and the total trials.
         param_init: Initial value per parameter to optimize.
         param_fixed: Fixed value per parameter not to optimize.
         sigmoid: Sigmoid function
         priors: Prior function per parameter
    Returns:
        Dictionary of parameter names with the parameter values,
        which maximize the posterior_grid.
    """
    for name in param_fixed.keys():
        # do not optimize fixed parameter
        if name in param_init:
            param_init.pop(name)
    for name, value in param_init.items():
        if not np.isscalar(value):
            raise PsignifitException(f'Expects scalar number as initialization of {name}, got {value}.')

    def objective(x):
        optimized_param = dict(zip(sorted(param_init.keys()), x))
        _grid = {**param_fixed, **optimized_param}
        return -log_posterior(data, sigmoid=sigmoid, priors=priors, grid=_grid)

    init_values = [value for name, value in sorted(param_init.items())]
    optimized_values = optimize.fmin(objective, init_values, disp=False)
    optimized_param = dict(zip(sorted(param_init.keys()), optimized_values))
    return {**param_fixed, **optimized_param}


def marginalize_posterior(parameter_grid: ParameterGrid, posterior_mass: np.ndarray) -> Dict[str, np.ndarray]:
    marginals = dict()
    for i, (param, grid) in enumerate(sorted(parameter_grid.items())):
        if grid is None or len(grid)==1:
            marginals[param] = None
        else:
            axis = tuple(range(0, i)) + tuple(range(i + 1, len(parameter_grid)))
            # we get first the unnormalized marginal, and then we scale it
            nmarginal = np.squeeze(posterior_mass.sum(axis))
            integral = np.trapz(nmarginal, x=grid)
            marginals[param] = nmarginal / integral

    return marginals
