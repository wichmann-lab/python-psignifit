import numpy as np
import pytest

from psignifit import _sigmoids


# fixed parameters for simple sigmoid sanity checks
X = np.linspace(1e-12, 1 - 1e-12, num=10000)
THRESHOLD_PARAM = 0.5
WIDTH_PARAM = 0.9
PC = 0.5
ALPHA = 0.05


# list of all sigmoids (after having removed aliases)
LOG_SIGS = ('weibull', 'logn', 'neg_weibull', 'neg_logn')


def test_ALL_SIGMOID_NAMES():
    TEST_SIGS = (
        'norm', 'gauss', 'neg_norm', 'neg_gauss', 'logistic', 'neg_logistic',
        'gumbel', 'neg_gumbel', 'rgumbel', 'neg_rgumbel',
        'logn', 'neg_logn', 'weibull', 'neg_weibull',
        'tdist', 'student', 'heavytail', 'neg_tdist', 'neg_student', 'neg_heavytail')
    for name in TEST_SIGS:
        assert name in _sigmoids.ALL_SIGMOID_NAMES


@pytest.mark.parametrize('sigmoid_name', _sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_by_name(sigmoid_name):
    s = _sigmoids.sigmoid_by_name(sigmoid_name)
    assert isinstance(s, _sigmoids.Sigmoid)

    s = _sigmoids.sigmoid_by_name(sigmoid_name.upper())
    assert isinstance(s, _sigmoids.Sigmoid)

    s = _sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    assert isinstance(s, _sigmoids.Sigmoid)

    assert sigmoid_name.startswith('neg_') == s.negative


@pytest.mark.parametrize('sigmoid_name', _sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_sanity_check(sigmoid_name):
    """ Basic sanity checks for sigmoids.

    These sanity checks test some basic relations between the parameters
    as well as rule of thumbs which can be derived from visual inspection
    of the sigmoid functions.
    """
    sigmoid = _sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    sigmoid.assert_sanity_checks(n_samples=100,
                                 threshold=THRESHOLD_PARAM,
                                 width=WIDTH_PARAM)


@pytest.mark.parametrize('sigmoid_name', _sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_roundtrip(sigmoid_name):
    sigmoid = _sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    x = 0.5
    y = sigmoid(x, THRESHOLD_PARAM, WIDTH_PARAM)
    reverse_x = sigmoid.inverse(y, THRESHOLD_PARAM, WIDTH_PARAM)
    assert np.isclose(x, reverse_x, atol=1e-6)

