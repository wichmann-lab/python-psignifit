# -*- coding: utf-8 -*-
"""
Utils class capsulating all custom made probabilistic functions
"""
import numpy as np
import scipy.stats

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


def get_grid(bounds, steps):
    """Return uniformely spaced grid within given bounds.

    Input Parameters:

       bounds  - a dictionary {parameter : (min_val, max_val)}
       steps    - a dictionary {parameter : nsteps}

    where `nsteps` is the number of steps in the grid.

    Returns:

        grid    - a dictionary {parameter: (min_val, val1, val2, ..., max_val)}
    """
    grid = {}
    for param in bounds:
        if bounds[param] is None:
            grid[param] = None
        else:
            grid[param] = np.linspace(*bounds[param], num=steps[param])
    return grid


def normalize(func, interval, steps=10000):
    """
    Return a normalized function, which has integral 1 within the interval.

    - `func` is a vectorized function that takes one single argument
    - `interval` is a tuple (lo, hi)

    Intregration is done using the composite trapezoidal rule.
    """
    if interval[0] == interval[1]:

        def nfunc(y):
            return np.ones_like(y)
    else:
        x = np.linspace(interval[0], interval[1], steps)
        integral = np.trapz(func(x), x=x)

        def nfunc(y):
            return func(y) / integral

    return nfunc


def nd_integrate(func, grid):
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
    M = len(grid)
    deltas = []
    # loop over all dimensions and gather the deltas (dx_1, dx_2, ..., dx_m)
    for steps in grid:
        # handle singleton dimensions
        if len(steps) == 1:
            deltas.append(1)
        else:
            delta = np.empty_like(steps)
            delta[1:] = np.diff(steps)
            # delta weight is half at the bounds of the integration interval
            delta[0] = delta[1] / 2
            delta[-1] = delta[-1] / 2
            deltas.append(delta)
    # create a meshgrid for each dimension
    mesh_grids = np.meshgrid(*deltas, copy=False, sparse=True, indexing='ij')
    weights = np.prod(mesh_grids, axis=0)
    return (func * weights).sum(), weights


def pool_data(data, xtol=0, max_gap=np.inf, max_length=np.inf):
    """
    Pool trials together which differ at maximum pool_xtol from the first one
    it finds, are separated by maximally pool_max_gap trials of other levels and
    at max pool_max_length trials appart in general.
    """
    ndata = data.shape[0]
    seen = [False] * ndata
    cum_ntrials = [0] + list(data[:, 2].cumsum())

    pool = []
    for i in range(ndata):
        if not seen[i]:
            current = data[i, 0]
            block = []
            gap = 0
            for j in range(i, ndata):
                if (cum_ntrials[j + 1] -
                        cum_ntrials[i]) > max_length or gap > max_gap:
                    break
                level, ncorrect, ntrials = data[j, :]
                if abs(level - current) <= xtol and not seen[j]:
                    seen[j] = True
                    gap = 0
                    block.append((level * ntrials, ncorrect, ntrials))
                else:
                    gap += ntrials

            level, ncorrect, ntrials = np.sum(block, axis=0)
            pool.append((level / ntrials, ncorrect, ntrials))

    return np.array(pool)


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
