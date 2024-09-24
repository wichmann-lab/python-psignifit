import numpy as np
import pytest

from psignifit import sigmoids


# fixed parameters for simple sigmoid sanity checks
X = np.linspace(1e-12, 1 - 1e-12, num=10000)
THRESHOLD_PARAM = 0.5
WIDTH_PARAM = 0.9
PC = 0.5
ALPHA = 0.05


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

    assert sigmoid_name.startswith('neg_') == s.negative


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_sanity_check(sigmoid_name):
    """ Basic sanity checks for sigmoids.

    These sanity checks test some basic relations between the parameters
    as well as rule of thumbs which can be derived from visual inspection
    of the sigmoid functions.
    """

    # fixed parameters for simple sigmoid sanity checks
    PC = 0.4
    threshold = 0.460172162722971  # Computed by hand to correspond to PC
    alpha = 0.083

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=alpha)

    assert_sanity_checks(
        sigmoid,
        n_samples=10000,
        threshold=threshold,
    )


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_roundtrip(sigmoid_name):
    pc = 0.7
    alpha = 0.12
    threshold = 0.6
    width = 0.6

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=pc, alpha=alpha)
    for x in np.linspace(0.1, 0.9, 10):
        y = sigmoid(x, threshold, width)
        reverse_x = sigmoid.inverse(y, threshold, width)
        assert np.isclose(x, reverse_x, atol=1e-6)
