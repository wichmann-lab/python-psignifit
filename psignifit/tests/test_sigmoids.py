import numpy as np
import pytest

from psignifit import sigmoids


# fixed parameters for simple sigmoid sanity checks
X = np.linspace(1e-12, 1-1e-12, num=10000)
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
        assert name in sigmoids.ALL_SIGMOID_NAMES

@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_by_name(sigmoid_name):
    s = sigmoids.sigmoid_by_name(sigmoid_name)
    assert isinstance(s, sigmoids.Sigmoid)

    s = sigmoids.sigmoid_by_name(sigmoid_name.upper())
    assert isinstance(s, sigmoids.Sigmoid)

    s = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    assert isinstance(s, sigmoids.Sigmoid)

    assert (sigmoid_name in LOG_SIGS) == s.logspace
    assert sigmoid_name.startswith('neg_') == s.negative


@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_sanity_check(sigmoid_name):
    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    x_threshold = THRESHOLD_PARAM
    x = np.linspace(1e-8, 1, 100)
    if sigmoid.negative:
        x = 1 - x
    if sigmoid.logspace:
        x_threshold = np.exp(x_threshold)
        x = np.exp(x)

    # sigmoid(M) == PC
    np.testing.assert_allclose(sigmoid(x_threshold, THRESHOLD_PARAM, WIDTH_PARAM), PC)

    # |X_L - X_R| == WIDTH, with
    # with sigmoid(X_L) == ALPHA
    # and  sigmoid(X_R) == 1 - ALPHA
    s = sigmoid(x, THRESHOLD_PARAM, WIDTH_PARAM)
    idx_alpha, idx_nalpha =  np.abs(s - ALPHA).argmin(), np.abs(s - (1 - ALPHA)).argmin()
    np.testing.assert_allclose(s[idx_nalpha] - s[idx_alpha], WIDTH_PARAM, atol=0.02)

    t = sigmoid.inverse(PC, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM)
    np.testing.assert_allclose(t, x_threshold)
    t = sigmoid.inverse(PC, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM, gamma=0, lambd=0)
    np.testing.assert_allclose(t, x_threshold)
