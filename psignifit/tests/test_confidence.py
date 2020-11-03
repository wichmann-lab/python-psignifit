from scipy import stats
import numpy as np
from numpy.testing import assert_equal

from psignifit.getConfRegion import grid_hdi

def test_grid_hdi():
    x = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, x)
    XY = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=-1)

    # Setup probability using 2-dimensional Gaussian with variances 1 and 2.
    probability = stats.multivariate_normal.pdf(XY, mean=[0, 0], cov=np.diag([1, 2]))
    probability_mass = probability / probability.sum()

    # Intervals should be minimal / maximal and centered for extreme credible mass.
    assert_equal([[49, 50], [49, 50]], grid_hdi(probability_mass, 0))
    assert_equal([[0, 99], [0, 99]], grid_hdi(probability_mass, 0.99999))
    # Intervals should reflect the variance differences of the 2-d Gaussian.
    assert_equal([[17, 82], [26, 73]], grid_hdi(probability_mass, 0.66))
