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
    """ Basic sanity checks for sigmoids.

    These sanity checks test some basic relations between the parameters
    as well as rule of thumbs which can be derived from visual inspection
    of the sigmoid functions.
    """
    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    threshold_stimulus_level = THRESHOLD_PARAM
    stimulus_levels = np.linspace(1e-8, 1, 100)
    if sigmoid.negative:
        stimulus_levels = 1 - stimulus_levels
    if sigmoid.logspace:
        threshold_stimulus_level = np.exp(threshold_stimulus_level)
        stimulus_levels = np.exp(stimulus_levels)

    # sigmoid(threshold_stimulus_level) == threshold_percent_correct
    np.testing.assert_allclose(sigmoid(threshold_stimulus_level, THRESHOLD_PARAM, WIDTH_PARAM), PC)

    # |X_L - X_R| == WIDTH, with
    # with sigmoid(X_L) == ALPHA
    # and  sigmoid(X_R) == 1 - ALPHA
    perc_correct = sigmoid(stimulus_levels, THRESHOLD_PARAM, WIDTH_PARAM)
    idx_alpha, idx_nalpha =  np.abs(perc_correct - ALPHA).argmin(), np.abs(perc_correct - (1 - ALPHA)).argmin()
    np.testing.assert_allclose(perc_correct[idx_nalpha] - perc_correct[idx_alpha], WIDTH_PARAM, atol=0.02)

    # Inverse sigmoid at threshold percentage correct (y-axis)
    # Expects the threshold stimulus level (x-axis).
    stimulus_level_from_inverse = sigmoid.inverse(PC, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM)
    np.testing.assert_allclose(stimulus_level_from_inverse, threshold_stimulus_level)
    stimulus_level_from_inverse = sigmoid.inverse(PC, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM, gamma=0, lambd=0)
    np.testing.assert_allclose(stimulus_level_from_inverse, threshold_stimulus_level)
    # Expects inverse(value(x)) == x
    y = sigmoid(stimulus_levels, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM)
    np.testing.assert_allclose(stimulus_levels, sigmoid.inverse(y, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM), atol=1e-9)

    slope = sigmoid.slope(stimulus_levels, threshold=THRESHOLD_PARAM, width=WIDTH_PARAM, gamma=0, lambd=0)
    # Expects maximal slope at a medium stimulus level
    assert 0.4 * len(slope) < np.argmax(np.abs(slope)) < 0.6 * len(slope)
    # Expects slope to be all positive/negative for standard/decreasing sigmoid
    if sigmoid.negative:
        assert np.all(slope < 0)
    else:
        assert np.all(slope > 0)

