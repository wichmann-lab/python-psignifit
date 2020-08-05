# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp

from .utils import fp_error_handler, PsignifitException

# this is the order in which the parameters are stored in the big
# likelihood 5 dimensional matrix
PARM_ORDER = ('threshold', 'width', 'lambda', 'gamma', 'eta')


def likelihood(data, sigmoid=None, priors=None, grid=None):
    p = log_likelihood(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # locate the maximum
    ind_max = np.unravel_index(p.argmax(), p.shape)
    logmax = p[ind_max]
    p -= logmax
    p = np.exp(p)
    # get the value of the parameters on the grid corresponding to the maximum
    grid_max = [grid[parm][i] for (i, parm) in zip(ind_max, PARM_ORDER)]
    return p, logmax, ind_max, grid_max


@fp_error_handler(divide='ignore')
def log_likelihood(data, sigmoid=None, priors=None, grid=None):
    thres = grid['threshold']
    width = grid['width']
    lambd = grid['lambda']
    gamma = grid['gamma']

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
    if grid['eta'] is None:
        eta_prime = None
        thres, width, lambd, gamma = np.meshgrid(thres,
                                                 width,
                                                 lambd,
                                                 gamma,
                                                 copy=False,
                                                 sparse=True,
                                                 indexing='ij')
    else:
        eta_std = grid['eta']
        eta = eta_std ** 2  # use variance instead of standard deviation
        eta_prime = 1 / eta[eta > 1e-09] - 1

        # for smaller variance we use the binomial model
        vbinom = (eta <= 1e-09).sum()
        thres, width, lambd, gamma, eta_prime = np.meshgrid(thres,
                                                            width,
                                                            lambd,
                                                            gamma,
                                                            eta_prime,
                                                            copy=False,
                                                            sparse=True,
                                                            indexing='ij')

    scale = 1 - gamma - lambd
    ###FIXME: handle the case with equal_asymptote

    pbin = 0
    p = 0
    # we could get rid of the loop by playing crazy fancy-indexing tricks
    # with the arrays here, but that version would be quite more unreadable.
    # as we don't expect huge number of levels, we should not lose too much
    # performance by using an explicit, but no profiling has been done to prove
    # this.
    for level, trials, correct_trials in zip(levels, ntrials, ncorrect):
        # Notation in paper: n=trials, k=correct_trials, x=level
        if trials == 0:
            continue

        # average predicted probability of correct
        psi = sigmoid(level, thres, width) * scale + gamma
        pbin += correct_trials * np.log(psi) + (trials - correct_trials) * np.log(1 - psi)
        if eta_prime is not None:
            a = psi * eta_prime
            b = (1 - psi) * eta_prime
            p += -sp.gammaln(trials + eta_prime) + sp.gammaln(eta_prime)

            if correct_trials == 0:
                p += (sp.gammaln(trials + b) - sp.gammaln(b))
            elif correct_trials == trials:
                p += (sp.gammaln(correct_trials + a) - sp.gammaln(a))
            elif correct_trials < trials:
                p += (sp.gammaln(correct_trials + a) + sp.gammaln(trials - correct_trials + b)
                      - sp.gammaln(a) - sp.gammaln(b))
            else: # we should never land here: we can't more ncorrect than ntrials
                raise PsignifitException('ncorrect %d > ntrials %d!' % (correct_trials, trials))

    if eta_prime is None:
        p = pbin
    else:
        pbin = np.broadcast_to(pbin, pbin.shape[:4] + (vbinom,))
        p = np.concatenate((pbin, p), axis=4)

    # add priors on the corresponding axis (they get added on the right axis
    # because of the meshgrid
    p += np.log(priors['threshold'](thres))
    p += np.log(priors['width'](width))
    p += np.log(priors['lambda'](lambd))
    ## FIXME equal asymptote case not contemplated here
    p += np.log(priors['gamma'](gamma))
    if eta_prime is not None:
        p += np.log(priors['eta'](eta_std))
    return p


def get_optm_llh(data, sigmoid=None, priors=None, grid=None):
    ###FIXME### does not work when grid[parm] = None!!!
    # return a curried version of log_likelihood to be used in the
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
        return -log_likelihood(data, sigmoid=sigmoid, priors=priors, grid=lgrid)

    return cllh, fixed
