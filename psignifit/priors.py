# -*- coding: utf-8 -*-
"""
"""
import warnings

import numpy as np
import scipy.stats

from .utils import norminv


def pthreshold(x, st_range):
    """Default prior for the threshold parameter

    A uniform prior over the range `st_range` of the data with a cosine fall off
    to 0 over half the range of the data.

    This prior expresses the belief that the threshold is anywhere in the range
    of the tested stimulus levels with equal probability and may be up to 50% of
    the spread of the data outside the range with decreasing probability"""
    # spread
    sp = st_range[1] - st_range[0]
    s0 = st_range[0]
    s1 = st_range[1]
    p = np.zeros_like(x)
    p[(s0 < x) & (x < s1)] = 1.
    left = ((s0 - sp / 2) <= x) & (x <= s0)
    p[left] = (1 + np.cos(2 * np.pi * (s0 - x[left]) / sp)) / 2
    right = (s1 <= x) & (x <= (s1 + sp / 2))
    p[right] = (1 + np.cos(2 * np.pi * (x[right] - s1) / sp)) / 2
    return p


def pwidth(x, alpha, wmin, wmax):
    """Default prior for the width parameter

    A uniform prior between two times the minimal distance of two tested stimulus
    levels and the range of the stimulus levels with cosine fall offs to 0 at
    the minimal difference of two stimulus levels and at 3 times the range of the
    tested stimulus levels"""
    # rescaling for alpha
    y = x * (norminv(.95) - norminv(.05)) / (norminv(1 - alpha) -
                                             norminv(alpha))
    p = np.zeros_like(x)
    p[((2 * wmin) < y) & (y < wmax)] = 1.
    left = (wmin <= y) & (y <= (2 * wmin))
    p[left] = (1 - np.cos(np.pi * (y[left] - wmin) / wmin)) / 2
    right = (wmax <= y) & (y <= (3 * wmax))
    p[right] = (1 + np.cos(np.pi / 2 * (y[right] - wmax) / wmax)) / 2
    return p


def plambda(x):
    """Default prior for the lapse rate

    A Beta distribution, wit parameters 1 and 10."""
    return scipy.stats.beta.pdf(x, 1, 10)


def pgamma(x):
    """Default prior for the guess rate

    A Beta distribution, wit parameters 1 and 10."""
    return scipy.stats.beta.pdf(x, 1, 10)


def peta(x, k):
    """Default prior for overdispersion

    A Beta distribution, wit parameters 1 and k."""
    return scipy.stats.beta.pdf(x, 1, k)


def checkPriors(data, options):
    """
    this runs a short test whether the provided priors are functional
     function checkPriors(data,options)
     concretely the priors are evaluated for a 25 values on each dimension and
     a warning is issued for zeros and a error for nan and infs and negative
     values

    """

    if options['logspace']:
        data[:, 0] = np.log(data[:, 0])
    """ on threshold
    values chosen according to standard boarders
    at the bounds it may be 0 -> a little inwards """
    data_min = np.min(data[:, 0])
    data_max = np.max(data[:, 0])
    dataspread = data_max - data_min
    testValues = np.linspace(data_min - .4 * dataspread,
                             data_max + .4 * dataspread, 25)

    testResult = options['priors'][0](testValues)

    testForWarnings(testResult, "the threshold")
    """ on width
    values according to standard priors
    """
    testValues = np.linspace(
        1.1 * np.min(np.diff(np.sort(np.unique(data[:, 0])))), 2.9 * dataspread,
        25)
    testResult = options['priors'][1](testValues)

    testForWarnings(testResult, "the width")
    """ on lambda
    values 0 to .9
    """
    testValues = np.linspace(0.0001, .9, 25)
    testResult = options['priors'][2](testValues)

    testForWarnings(testResult, "lambda")
    """ on gamma
    values 0 to .9
    """
    testValues = np.linspace(0.0001, .9, 25)
    testResult = options['priors'][3](testValues)

    testForWarnings(testResult, "gamma")
    """ on eta
    values 0 to .9
    """
    testValues = np.linspace(0, .9, 25)
    testResult = options['priors'][4](testValues)

    testForWarnings(testResult, "eta")


def testForWarnings(testResult, parameter):
    assert all(
        np.isfinite(testResult)
    ), "the prior you provided for %s returns non-finite values" % parameter
    assert all(
        testResult >= 0
    ), "the prior you provided for %s returns negative values" % parameter

    if any(testResult == 0):
        warnings.warn("the prior you provided for %s returns zeros" % parameter)
