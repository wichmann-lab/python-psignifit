# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple, Dict
import warnings

import numpy as np
import scipy.stats

from ._typing import Prior

# accomodate numpy versions < 2
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

def threshold_prior(x, stimulus_range: Tuple[float, float]):
    """Default prior for the threshold parameter

    A uniform prior over the range `st_range` of the data with a cosine fall off
    to 0 over half the range of the data.

    This prior expresses the belief that the threshold is anywhere in the range
    of the tested stimulus levels with equal probability and may be up to 50% of
    the spread of the data outside the range with decreasing probability"""
    # spread
    sp = stimulus_range[1] - stimulus_range[0]
    s0 = stimulus_range[0]
    s1 = stimulus_range[1]
    p = np.zeros_like(x)
    p[(s0 < x) & (x < s1)] = 1.
    left = ((s0 - sp / 2) <= x) & (x <= s0)
    p[left] = (1 + np.cos(2 * np.pi * (s0 - x[left]) / sp)) / 2
    right = (s1 <= x) & (x <= (s1 + sp / 2))
    p[right] = (1 + np.cos(2 * np.pi * (x[right] - s1) / sp)) / 2
    return p


def width_prior(x, alpha: np.ndarray, wmin: np.ndarray, wmax: np.ndarray):
    """Default prior for the width parameter

    A uniform prior between two times the minimal distance of two tested stimulus
    levels and the range of the stimulus levels with cosine fall offs to 0 at
    the minimal difference of two stimulus levels and at 3 times the range of the
    tested stimulus levels"""
    # rescaling for alpha
    norminv = scipy.stats.norm(loc=0, scale=1).ppf
    y = x * (norminv(.95) - norminv(.05)) / (norminv(1 - alpha) -
                                             norminv(alpha))
    p = np.zeros_like(x)
    p[((2 * wmin) < y) & (y < wmax)] = 1.
    left = (wmin <= y) & (y <= (2 * wmin))
    p[left] = (1 - np.cos(np.pi * (y[left] - wmin) / wmin)) / 2
    right = (wmax <= y) & (y <= (3 * wmax))
    p[right] = (1 + np.cos(np.pi / 2 * (y[right] - wmax) / wmax)) / 2
    return p


def lambda_prior(x):
    """Default prior for the lapse rate

    A Beta distribution, with parameters 1 and 10."""
    return scipy.stats.beta.pdf(x, 1, 10)


def gamma_prior(x):
    """Default prior for the guess rate

    A Beta distribution, wit parameters 1 and 10."""
    return scipy.stats.beta.pdf(x, 1, 10)


def eta_prior(x, k):
    """Default prior for overdispersion

    A Beta distribution, wit parameters 1 and k."""
    return scipy.stats.beta.pdf(x, 1, k)


def default_prior(parameter: str, stimulus_range: Tuple[float, float], width_min: float,
                   width_alpha: float, beta: float, threshold_percent_correct: float = 0.5) -> Dict[str, Prior]:
    if not np.isclose(threshold_percent_correct, 0.5):
        warnings.warn(f"The `thresh_PC` parameter is set to {threshold_percent_correct}, not the default 0.5. "
                      f"Be aware that the default prior over threshold assumes that the experimental stimulus range "
                      f"covers the range where the threshold likely falls. If this doesn't match your setup, you'll "
                      f"need a custom prior. See the documentation for guidance.")

    if parameter == 'threshold':
        prior = partial(threshold_prior, stimulus_range=stimulus_range)
    elif parameter == 'width':
        prior = partial(width_prior, wmin=width_min,
                         wmax=stimulus_range[1] - stimulus_range[0],
                         alpha=width_alpha)
    elif parameter == 'lambda':
        prior = lambda_prior
    elif parameter == 'gamma':
        prior = gamma_prior
    elif parameter == 'eta':
        prior = partial(eta_prior, k=beta)
    else:
        raise ValueError(f"Unknown parameter '{parameter}'")
    return prior


def _check_prior(priors, name, values):
    if name not in priors:
        return
    result = priors[name](values)
    assert np.all(np.isfinite(result)), f"Prior '{name}' returns non-finite values."
    assert np.all(result >= 0), f"Prior '{name}' returns negative values."
    assert np.any(result > 0), f"Prior '{name}' returns zeros."


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

    test_values = np.linspace(1.1 * width_min, 2.9 * dataspread, n_test_values)
    _check_prior(priors, "width", test_values)

    test_values = np.linspace(0.0001, .9, n_test_values)
    _check_prior(priors, "lambda", test_values)
    _check_prior(priors, "gamma", test_values)
    _check_prior(priors, "eta", test_values)


def normalize_prior(func: Prior, interval: Tuple[float, float], steps: int = 10000) -> Prior:
    """ Normalize the prior function to have integral 1 within the interval.

    Integration is done using the composite trapezoidal rule.

    Args:
        func: is a vectorized function that takes one single argument
        interval: is a tuple (lo, hi)

    Returns:
        The normalized prior function.
    """
    if np.isclose(interval[0], interval[1]):
        def nfunc(y):
            return np.ones_like(y)
    else:
        x = np.linspace(interval[0], interval[1], steps)
        integral = trapezoid(func(x), x=x)

        def nfunc(y):
            return func(y) / integral

    return nfunc


def setup_priors(custom_priors, bounds, stimulus_range, width_min, width_alpha, beta_prior, threshold_prop_correct):
    priors = {}
    for parameter in bounds:
        priors[parameter] = default_prior(parameter, stimulus_range, width_min, width_alpha, beta_prior, threshold_prop_correct)

    if custom_priors is not None:
        priors.update(custom_priors)
    check_priors(priors, stimulus_range, width_min)

    for parameter, prior in priors.items():
        priors[parameter] = normalize_prior(prior, bounds[parameter])
    return priors


