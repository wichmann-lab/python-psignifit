# -*- coding: utf-8 -*-
"""
"""
import warnings
from typing import Tuple, Dict
from functools import partial

import numpy as np
import scipy.stats

from .utils import norminv, PsignifitException
from .typing import Prior


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


def default_priors(stimulus_range: Tuple[float, float], width_min: float,
                   width_alpha: float, beta: float, thresh_PC: float = 0.5) -> Dict[str, Prior]:
    if not np.isclose(thresh_PC, 0.5):
        raise ValueError("Default prior 'threshold' expects thresh_PC=0.5, got {thresh_PC = }")

    return {
        'threshold': partial(pthreshold, st_range=stimulus_range),
        'width': partial(pwidth, wmin=width_min,
                         wmax=stimulus_range[1] - stimulus_range[0],
                         alpha=width_alpha),
        'lambda': plambda,
        'gamma': pgamma,
        'eta': partial(peta, k=beta),
    }


def check_priors(priors: Dict[str, Prior], stimulus_range: Tuple[float, float],
                 width_min: float, n_test_values: int = 25):
    """ Checks basic properties of the prior results.

    Each prior function is evaluated on a small number of
    artificial data points, created using stimulus_range and width_min.

    Args:
        priors: Dictionary of prior functions.
        stimulus_range: Minimum and maximum stimulus values.
        width_min: Parameter adjusting the lower bound of the width prior
        n_test_values: Number of data points used to test the priors.
    Raises:
        AssertionError if prior returns non-finite or non-positive values.
    """
    data_min, data_max = stimulus_range
    dataspread = data_max - data_min

    test_values = np.linspace(data_min - .4 * dataspread, data_max + .4 * dataspread, n_test_values)
    _check_prior(priors, "threshold", test_values)

    test_values = np.linspace( 1.1 * width_min, 2.9 * dataspread, n_test_values)
    _check_prior(priors, "width", test_values)

    test_values = np.linspace(0.0001, .9, n_test_values)
    _check_prior(priors, "lambda", test_values)
    _check_prior(priors, "gamma", test_values)
    _check_prior(priors, "eta", test_values)


def _check_prior(priors, name, values):
    result = priors[name](values)
    assert np.all(np.isfinite(result)), f"Prior '{name}' returns non-finite values."
    assert np.all(result >= 0), f"Prior '{name}' returns negative values."
    assert np.all(result != 0), f"Prior '{name}' returns zeros."
