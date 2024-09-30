import numpy as np
import pytest

from psignifit import sigmoids


def test_ALL_SIGMOID_NAMES():
    TEST_SIGS = (
        'norm', 'gauss', 'neg_norm', 'neg_gauss', 'logistic', 'neg_logistic',
        'gumbel', 'neg_gumbel', 'rgumbel', 'neg_rgumbel',
        'weibull', 'neg_weibull',
        'tdist', 'student', 'heavytail', 'neg_tdist', 'neg_student', 'neg_heavytail')
    for name in TEST_SIGS:
        assert name in sigmoids.ALL_SIGMOID_NAMES


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_by_name(sigmoid_name):
    s = sigmoids.sigmoid_by_name(sigmoid_name)
    assert isinstance(s, sigmoids.Sigmoid)

    s = sigmoids.sigmoid_by_name(sigmoid_name.upper())
    assert isinstance(s, sigmoids.Sigmoid)

    s = sigmoids.sigmoid_by_name(sigmoid_name, PC=0.2, alpha=0.132)
    assert isinstance(s, sigmoids.Sigmoid)
    assert s.PC == 0.2
    assert s.alpha == 0.132

    assert sigmoid_name.startswith('neg_') == s.negative


@pytest.mark.parametrize(
    'subclass, expected_y',
    {
        # Computed by hand in Switzerland
        sigmoids.Gaussian: np.array([0.43099688, 0.6, 0.93759564]),
        sigmoids.Logistic: np.array([0.4189846, 0.6, 0.93103448]),
        sigmoids.Gumbel: np.array([0.42189253, 0.6, 0.98620617]),
        sigmoids.ReverseGumbel: np.array([0.42564916, 0.6, 0.89648769]),
        sigmoids.Student: np.array([0.30539173, 0.6, 0.90901293]),
    }.items()
)
def test_sigmoid_values(subclass, expected_y):
    sigmoid = subclass(PC=0.6, alpha=0.1)
    x = np.array([9.5, 10.0, 11.5])
    y = sigmoid(x, threshold=10, width=3)
    np.testing.assert_allclose(y, expected_y, atol=1e-6)


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_inverse(sigmoid_name):
    pc = 0.7
    alpha = 0.12
    threshold = 0.6
    width = 0.6

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=pc, alpha=alpha)
    x = np.linspace(0.1, 0.9, 10)
    y = sigmoid(x, threshold, width)
    reverse_x = sigmoid.inverse(y, threshold, width)
    np.testing.assert_allclose(x, reverse_x, atol=1e-6)


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_slope(sigmoid_name):
    pc = 0.7
    alpha = 0.12
    threshold = 0.6
    width = 0.6

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=pc, alpha=alpha)
    x = 0.4
    slope = sigmoid.slope(x, threshold, width)

    delta = 0.00001
    numerical_slope = (
            (sigmoid(x+delta, threshold, width)
             - sigmoid(x-delta, threshold, width))
            / (2 * delta)
    )
    np.testing.assert_allclose(slope, numerical_slope, atol=1e-6)


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_sanity_check(sigmoid_name):
    """ Basic sanity checks for sigmoids.

    These sanity checks test some basic relations between the parameters
    as well as rule of thumbs which can be derived from visual inspection
    of the sigmoid functions.
    """
    # fixed parameters for simple sigmoid sanity checks
    PC = 0.45
    # Threshold computed by hand to correspond to PC (for negative sigmoids, we'll compare this
    # threshold with the value at 1 - PC)
    threshold = 0.54
    alpha = 0.083

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=alpha)
    sigmoids.assert_sigmoid_sanity_checks(
        sigmoid, n_samples=10000, threshold=threshold, width=0.7,
    )
