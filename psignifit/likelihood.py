# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import scipy.special as sp

from .utils import fp_error_handler, PsignifitException
from .typing import Sigmoid, Prior, ParameterGrid

# this is the order in which the parameters are stored in the big
# likelihood 5 dimensional matrix
PARM_ORDER = ('threshold', 'width', 'lambda', 'gamma', 'eta')


def likelihood(data, sigmoid=None, priors=None, grid=None):
    p = log_posterior(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # locate the maximum
    ind_max = np.unravel_index(p.argmax(), p.shape)
    logmax = p[ind_max]
    p -= logmax
    p = np.exp(p)
    # get the value of the parameters on the grid corresponding to the maximum
    grid_max = [grid[parm][i] for (i, parm) in zip(ind_max, PARM_ORDER)]
    return p, logmax, ind_max, grid_max


@fp_error_handler(divide='ignore')
def log_posterior(data: np.ndarray, sigmoid: Sigmoid, priors: Dict[str, Prior], grid: ParameterGrid) -> np.ndarray:
    """ Estimate the logarithmic posterior probability of sigmoid parameters.

    The function estimates the log-likelihood for a grid of sigmoid parameters and adds the corresponding
    log-priors. This is described in formulas A.4 and A.5 of [Schuett2016]_.

    Args:
        data: numpy array with three columns for the stimulus levels, the correct trials, and the total trials.
        sigmoid: sigmoid function with three input parameter, see :module:`~psignifit.sigmoids`.
        priors: dictionary of parameter names and corresponding prior function, see :module:`~psignifit.priors`.
        grid: dictionary of parameter name and numpy array with parameter values.
    Returns:
        Numpy array with the log-posterior for each parameter combination in grid.

    .. [Schuett2016] Schütt, H. H., Harmeling, S., Macke, J. H. and Wichmann, F. A. (2016).
          Painfree and accurate Bayesian estimation of psychometric functions for (potentially) overdispersed data.
          Vision Research 122, 105-123.
    """
    thres = grid['threshold']
    width = grid['width']
    lambd = grid['lambda']
    gamma = grid['gamma']
    eta_std = grid['eta']
    if eta_std is None:
        eta_prime = None
    else:
        eta = eta_std ** 2  # use variance instead of standard deviation
        eta_prime = 1 / eta[eta > 1e-09] - 1
        eta_binom = (eta <= 1e-09).sum() # for small variance we use the binomial model

    levels = data[:, 0]
    ncorrect = data[:, 1]
    ntrials = data[:, 2]

    ###FIXME: fix treatment of no 'eta' configurations,
    # best choice is probably having grid['eta'] = [0]
    # then rewrite below so that we don't test for v -> None but for
    # v -> [] instead, then rearrange the p, pbin broadcasting and
    # concatenating?
    # fixing this would also fix the case with the curried likelihood
    # below, that currently does work for grid['eta'] -> None
    if gamma is None and eta_prime is None:
        thres, width, lambd = np.meshgrid(thres, width, lambd, copy=False, sparse=True, indexing='ij')
    elif gamma is None:
        thres, width, lambd, eta_prime = np.meshgrid(thres, width, lambd, eta_prime,
                                                     copy=False, sparse=True, indexing='ij')
    elif eta_prime is None:
        thres, width, lambd, gamma = np.meshgrid(thres, width, lambd, gamma, copy=False, sparse=True, indexing='ij')
    else:
        thres, width, lambd, gamma, eta_prime = np.meshgrid(thres, width, lambd, gamma, eta_prime,
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
        pbin += correct_trials * np.nan_to_num(np.log(psi)) + (trials - correct_trials) * np.nan_to_num(np.log(1 - psi))
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
        else:
            # we should never land here: we can't more ncorrect than ntrials
            raise PsignifitException('ncorrect %d > ntrials %d!' % (correct_trials, trials))

    if eta_prime is None:
        p = pbin
    else:
        print(pbin.shape, p.shape, pbin.shape[:-1] + (eta_binom,))
        pbin = np.broadcast_to(pbin, pbin.shape[:-1] + (eta_binom,))
        p = np.concatenate((pbin, p), axis=-1)

    # add priors on the corresponding axis
    p += np.log(priors['threshold'](thres))
    p += np.log(priors['width'](width))
    p += np.log(priors['lambda'](lambd))
    p += np.log(priors['gamma'](gamma))
    if eta_prime is not None:
        p += np.log(priors['eta'](eta_std))
    return p


def get_optm_llh(data, sigmoid=None, priors=None, grid=None):
    ###FIXME### does not work when grid[parm] = None!!!
    # return a curried version of log_posterior to be used in the
    # optimization routine
    # - go through the grid and detect "fixed" parameters that we shouldn't
    #   try to optimize
    fixed_parms = [parm for parm, steps in grid.items() if len(steps) == 1]
    # create a list of (fixed_parm_index, fixed_parm_value) tuples
    fixed = [(PARM_ORDER.index(parm), grid[parm][0]) for parm in fixed_parms]

    def cllh(x):
        # convert to list for efficient insertion
        y = list(x)
        for fidx, fval in fixed:
            # insert value of fixed parameter at the right position
            y.insert(fidx, fval)
        # create a temporary grid with only one step (all params are fixed)
        lgrid = {parm: y[idx] for idx, parm in enumerate(PARM_ORDER)}
        return -log_posterior(data, sigmoid=sigmoid, priors=priors, grid=lgrid)

    return cllh, fixed
