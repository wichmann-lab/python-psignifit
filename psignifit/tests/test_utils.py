import numpy as np

from psignifit import utils


def test_normalize():
    # a constant function
    func = lambda x: np.ones_like(x)
    # the integral is length of x, so the normalized function should return 1/len(x)
    x = np.arange(11)
    norm = utils.normalize(func, (0, 10))
    assert np.allclose(1. / 10, norm(x))

    # For a fixed value, the integral should be one
    norm = utils.normalize(func, (1, 1))
    assert np.allclose([1], norm(1))


def test_normalize_sin():
    # for sin the integral in (0, pi/2) is 1, so the norm(sin) == sin
    x = np.linspace(0, np.pi / 2, 100)
    norm = utils.normalize(np.sin, (0, np.pi / 2))
    assert np.allclose(np.sin(x), norm(x))

    # For a fixed value, the integral should be one
    norm = utils.normalize(np.sin, (1, 1))
    assert np.allclose([1], norm(1))


def test_get_grid():
    bounds = {
        'none': None,
        'fixed': (0.5, 0.5),
        'normal': (0, 1),
    }
    steps = {
        'none': 3,
        'normal': 15,
    }

    grid = utils.get_grid(bounds, steps)
    assert grid['none'] is None
    assert grid['normal'].shape == (15,)
    np.testing.assert_equal(grid['fixed'], [0.5])




