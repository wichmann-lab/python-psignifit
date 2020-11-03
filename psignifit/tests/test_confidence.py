from scipy import stats
import numpy as np
from numpy.testing import assert_allclose
import pytest

from psignifit.getConfRegion import grid_hdi
from psignifit.getConfRegion import percentile_intervals

N = 100

@pytest.fixture
def probability_mass():
    x = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(x, x)
    XY = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=-1)

    # Setup probability using 2-dimensional Gaussian with variances 1 and 2.
    probability = stats.multivariate_normal.pdf(XY, mean=[0, 0], cov=np.diag([1, 2]))
    return probability / probability.sum()


@pytest.fixture
def grid_values():
    x = np.arange(N)
    return np.array([x, x])


def test_grid_hdi(probability_mass, grid_values):
    # Intervals should be minimal / maximal and centered for extreme credible mass.
    assert_allclose([[49, 50], [49, 50]], grid_hdi(probability_mass, grid_values, 0))
    assert_allclose([[0, 99], [0, 99]], grid_hdi(probability_mass, grid_values, 0.99999))
    # Intervals should reflect the variance differences of the 2-d Gaussian.
    assert_allclose([[17, 82], [26, 73]], grid_hdi(probability_mass, grid_values, 0.66))


def test_percentile_intervals(probability_mass, grid_values):
    # Intervals should be minimal / maximal and centered for extreme credible mass.
    assert_allclose([[49, 49], [49, 49]], percentile_intervals(probability_mass, grid_values, 0))
    assert_allclose([[0, 99], [0, 99]], percentile_intervals(probability_mass, grid_values, 0.99999), atol=0.02)
    # Intervals should reflect the variance differences of the 2-d Gaussian.
    assert_allclose([[8.2, 89.8], [17.0, 81.0]], percentile_intervals(probability_mass, grid_values, 0.95), atol=0.02)