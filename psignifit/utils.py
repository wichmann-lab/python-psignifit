# -*- coding: utf-8 -*-
"""
Utils class capsulating all custom made probabilistic functions
"""
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.stats

from .typing import ParameterBounds, Prior

# Alias common statistical distribution to be reused all over the place.

# - Normal distribution:
#   - This one is useful when we want mean=0, std=1
#     Percent point function -> inverse of cumulative normal distribution function
#     returns percentiles
norminv = scipy.stats.norm(loc=0, scale=1).ppf
#   - also instantiate a generic version
norminvg = scipy.stats.norm.ppf
#   - Cumulative normal distribution function
normcdf = scipy.stats.norm.cdf

# T-Student with df=1
t1cdf = scipy.stats.t(1).cdf
t1icdf = scipy.stats.t(1).ppf


# our own Exception class
class PsignifitException(Exception):
    pass


# create a decorator from the numpy errstate contextmanager, used to handle
# floating point errors. In our case divide-by-zero errors are usually harmless,
# because we are working in log space and log(0)==-inf is a valid result
class fp_error_handler(np.errstate):
    pass


def get_grid(bounds: ParameterBounds, steps: Dict[str, int]) -> Dict[str, Optional[np.ndarray]]:
    """Return uniformely spaced grid within given bounds.

    If the bound start and end values are close, a fixed value is assumed and the grid entry contains
    only the start.
    If the bound is None, the grid entry will be None.

    Args:
       bounds: a dictionary {parameter : (min_val, max_val)}
       steps: a dictionary {parameter : nsteps} where `nsteps` is the number of steps in the grid.

    Returns:
        grid:  a dictionary {parameter: (min_val, val1, val2, ..., max_val)}
    """
    grid = {}
    for param, bound in bounds.items():
        if bound is None:
            grid[param] = None
        elif np.isclose(bound[0], bound[1]):
            grid[param] = np.array([bound[0]])
        else:
            grid[param] = np.linspace(*bound, num=steps[param])
    return grid


def normalize(func: Prior, interval: Tuple[float, float], steps: int = 10000) -> Prior:
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
        integral = np.trapz(func(x), x=x)

        def nfunc(y):
            return func(y) / integral

    return nfunc


def strToDim(string):
    """
    Finds the number corresponding to a dim/parameter given as a string.
    """
    s = string.lower()
    if s in ['threshold', 'thresh', 'm', 't', 'alpha', '0']:
        return 0, 'Threshold'
    elif s in ['width', 'w', 'beta', '1']:
        return 1, 'Width'
    elif s in [
            'lapse', 'lambda', 'lapserate', 'lapse rate', 'lapse-rate',
            'upper asymptote', 'l', '2'
    ]:
        return 2, r'$\lambda$'
    elif s in [
            'gamma', 'guess', 'guessrate', 'guess rate', 'guess-rate',
            'lower asymptote', 'g', '3'
    ]:
        return 3, r'$\gamma$'
    elif s in ['sigma', 'std', 's', 'eta', 'e', '4']:
        return 4, r'$\eta$'


def check_data(data: np.ndarray, logspace: Optional[bool] = None) -> np.ndarray:
    """ Check data format, type and range.

    Args:
        data: The data matrix with columns levels, number of correct and number of trials
        logspace: Data should be used logarithmically. If None, no test on logspace is performed.
    Returns:
        data as float numpy array
    Raises:
        PsignifitException: if checks fail.
    """
    data = np.asarray(data, dtype=float)
    if len(data.shape) != 2 and data.shape[1] != 3:
        raise PsignifitException("Expects data to be two dimensional with three columns, got {data.shape = }")
    levels, ncorrect, ntrials = data[:, 0], data[:, 1], data[:, 2]

    # levels should show some variance
    if levels.max() == levels.min():
        raise PsignifitException('Your stimulus levels are all identical.'
                                 ' They can not be fitted by a sigmoid!')
    # ncorrect and ntrials should be integers
    if not np.allclose(ncorrect, ncorrect.astype(int)):
        raise PsignifitException(
            'The number correct column contains non integer'
            ' numbers!')
    if not np.allclose(ntrials, ntrials.astype(int)):
        raise PsignifitException('The number of trials column contains non'
                                 ' integer numbers!')
    if logspace is True and levels.min() <= 0:
        raise PsignifitException(f'Sigmoid {data.sigmoid} expects positive stimulus level data.')

    return data
