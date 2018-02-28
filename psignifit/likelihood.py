# -*- coding: utf-8 -*-
import functools
import numpy as np
import scipy.special as sp

from . import sigmoids
from .utils import fp_error_handler

@fp_error_handler(divide='ignore')
def log_likelihood(data, sigmoid=None, priors=None, grid=None):

    thres = grid['threshold']
    width = grid['width']
    lambd = grid['lambda']
    gamma = grid['gamma']
    parm_order = ('threshold', 'width', 'lambda', 'gamma')
    if grid['eta'] is None:
        v = None
        thres, width, lambd, gamma = np.meshgrid(thres, width, lambd, gamma,
                                                 copy=False, sparse=True)
    else:
        parm_order = parm_order + ('eta', )
        eta_std = grid['eta']
        eta = eta_std**2 # use variance instead of standard deviation
        v = 1/eta[eta > 1e-09] - 1

        # for smaller variance we use the binomial model
        vbinom = (eta <= 1e-09).sum()
        thres, width, lambd, gamma, v = np.meshgrid(thres, width, lambd, gamma,
                                                    v, copy=False, sparse=True)

    levels = data[:,0]
    ncorrect = data[:,1]
    ntrials = data[:,2]

    scale = 1 - gamma - lambd
    ###FIXME: handle the case with equal_asymptote

    pbin = 0
    p = 0
    for i in range(len(levels)):
        x = levels[i]
        # average predicted probability of correct
        psi = sigmoid(x, thres, width)*scale + gamma
        n = ntrials[i]
        k = ncorrect[i]
        if k == 0:
            pbin += pbin + n * np.log(1-psi)
            if v is not None:
                b = (1-psi)*v
                p += (sp.gammaln(n+b) - sp.gammaln(n+v) - sp.gammaln(b) +
                      sp.gammaln(v))
        elif k == n:
            pbin += k * np.log(psi)
            if v is not None:
                a = psi*v
                p += (sp.gammaln(k+a) - sp.gammaln(n+v) - sp.gammaln(a) +
                      sp.gammaln(v))
        elif k < n:
            psi_r = 1-psi
            pbin += k*np.log(psi) + (n-k)*np.log(psi_r)
            if v is not None:
                a = psi*v
                b = (1-psi)*v
                p += ( sp.gammaln(k+a) + sp.gammaln(n-k+b) - sp.gammaln(n+v) -
                        sp.gammaln(a) - sp.gammaln(b) + sp.gammaln(v))
        else:
            # this is n==0, i.e. no trials done at this stimulus level
            pass

    pbin = np.broadcast_to(pbin, pbin.shape[:4]+(vbinom,))
    p = np.concatenate((pbin, p), axis=4)

    # add priors on the corresponding axis (they get added on the right axis
    # because of the meshgrid
    p += np.log(priors['threshold'](thres))
    p += np.log(priors['width'](width))
    p += np.log(priors['lambda'](lambd))
    ## FIXME equal asymptote case not contemplated here
    p += np.log(priors['gamma'](gamma))
    if grid['eta'] is not None:
        p += np.log(priors['eta'](eta_std))
    return p, parm_order

def likelihood(data, sigmoid=None, priors=None, grid=None):
    p, p_order = log_likelihood(data, sigmoid=sigmoid, priors=priors, grid=grid)
    logmax = np.max(p)
    p -= logmax
    p = np.exp(p)
    return p, logmax, p_order

