""" Tests for confidence intervals.

We use a multi-variate normal distribution
and compare the confidence intervals to known properties
of the distribution.
"""
from scipy import stats
import numpy as np
from numpy.testing import assert_allclose
import pytest

from psignifit._confidence import confidence_intervals
from psignifit._confidence import grid_hdi
from psignifit._confidence import percentile_intervals

N = 100
GRID_RANGE = (-3, 3)
ATOL = (max(GRID_RANGE) - min(GRID_RANGE)) / N

# Use rather small STD or increase the grid range
# such that probabilities at the grid margin are close to zero
# and the empirical CDF approximation error is small.
STD = np.array([1, 0.5, 0.75])
STD_CI = np.c_[-STD, STD]
NORMAL = stats.multivariate_normal(mean=np.zeros_like(STD), cov=np.diag(STD**2))


@pytest.fixture
def zerocentered_normal_mass():
    x = np.linspace(*GRID_RANGE, N)
    X_mesh = np.meshgrid(*[x for __ in range(len(STD))])
    X = np.concatenate([X[..., np.newaxis] for X in X_mesh], axis=-1)
    X = np.swapaxes(X, 0, 1)

    # Setup probability using multi-dimensional Gaussian with variances 1 and 2.
    probability = NORMAL.pdf(X)
    return probability / probability.sum()


@pytest.fixture
def grid_values():
    x = np.linspace(*GRID_RANGE, N)
    return np.array([x, x, x])


def test_confidence_intervals(zerocentered_normal_mass, grid_values):
    p_values = [0.05, 0.5, 0.95]

    intervals = confidence_intervals(zerocentered_normal_mass, grid_values, p_values, mode='project')
    assert intervals.shape == (len(grid_values), len(p_values), 2)
    intervals = confidence_intervals(zerocentered_normal_mass, grid_values, p_values, mode='percentiles')
    assert intervals.shape == (len(grid_values), len(p_values), 2)

    with pytest.raises(ValueError):
        confidence_intervals(zerocentered_normal_mass, grid_values, p_values, mode='stripes')
    with pytest.raises(ValueError):
        confidence_intervals(zerocentered_normal_mass, grid_values, p_values, mode='foobar')
    with pytest.raises(ValueError):
        confidence_intervals(zerocentered_normal_mass * 3, grid_values, p_values, mode='project')


def test_grid_hdi(zerocentered_normal_mass, grid_values):
    """ Test grid hdi CIs reflect the relative difference in the axis' std. dev. of a multivariate normal.

    The grid hdi method estimates the area (or hypercube), such that the confidence intervals
    cannot directly be tested on the (marginal) normal distributions.
    This test is more indirect, and asserts that the relative difference in marginal standard deviation
    is present in the confidence intervals. In addition, the CI has to be larger compared to the percentile interval
    method (as this is estimating probability of ellipsis instead of square in the probability grid).
    """
    assert_allclose(0 * STD_CI, grid_hdi(zerocentered_normal_mass, grid_values, 0), atol=ATOL)
    assert_allclose(1.85 * STD_CI, grid_hdi(zerocentered_normal_mass, grid_values, 0.6827), atol=ATOL)
    assert_allclose(2.76 * STD_CI, grid_hdi(zerocentered_normal_mass, grid_values, 0.9545), atol=ATOL)
    assert_allclose(np.clip(4.31 * STD_CI, *GRID_RANGE),
                    grid_hdi(zerocentered_normal_mass, grid_values, 0.9999), atol=ATOL)


def test_percentile_intervals(zerocentered_normal_mass, grid_values):
    """ Test if percentile_intervals on multivatiate normal distribution, using the `68-95-99.7 rule`_.

    The percentile method uses only the marginal probabilities.
    For the non-correlated multivariate normal, the marginals are normal distributions.
    This means, the confidence intervals are integer multiples of the standard deviation
    for probabilities 0.683, 0.955, and 0.997.

    As this method is based on the empirical CDF, it becomes very inaccurate if
    probabilites at the grid margin are not close to zero.


    .. _`68-95-99.7 rule`: https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    """
    assert_allclose(0 * STD_CI, percentile_intervals(zerocentered_normal_mass, grid_values, 0), atol=ATOL)
    assert_allclose(1 * STD_CI, percentile_intervals(zerocentered_normal_mass, grid_values, 0.6827), atol=2 * ATOL)
    assert_allclose(2 * STD_CI, percentile_intervals(zerocentered_normal_mass, grid_values, 0.9545), atol=4 * ATOL)
    assert_allclose(3 * STD_CI, percentile_intervals(zerocentered_normal_mass, grid_values, 0.9973), atol=4 * ATOL)
