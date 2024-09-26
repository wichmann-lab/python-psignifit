import numpy as np
import pytest

from psignifit import sigmoids


def assert_sanity_checks(sigmoid, threshold: float):
    """ Assert multiple sanity checks on this sigmoid implementation.

    These checks cannot completely assure the correct implementation of a sigmoid,
    but try to catch common and obvious mistakes.
    We recommend to use these for custom sigmoid subclasses.

    The checks are performed on linear spaced stimulus levels between 0 and 1
    and the provided sigmoid parameters.

    Two checks for relations between parameters:
      - `sigmoid(threshold_stimulus_level) == threshold_percent_correct`
      - `|X_L - X_R| == width`
        with `sigmoid(X_L) == alpha`
        and  `sigmoid(X_R) == 1 - alpha`

    Two checks for the inverse:
      - `inverse(PC) == threshold_stimulus_level`
      - `inverse(inverse(stimulus_levels) == stimulus_levels`

    Two checks for the slope:
      - `maximum(|slope(stimulus_levels)|)` close to `|slope(0.5)|`
      - `slope(stimulus_levels) > 0`  (or < 0 for negative sigmoid)

    Args:
        sigmoid : The Sigmoid subclass to test
        n_samples: Number of stimulus levels between 0 (exclusive) and 1 for tests
        threshold: Parameter value for threshold at PC
        width: Width of the sigmoid
    Raises:
        AssertionError if a sanity check fails.
    """
    stimulus_levels = np.linspace(-0.3, 1.3, 10000)
    width = 1 - sigmoid.alpha * 2

    # Test that sigmoid(threshold_stimulus_level) == threshold_percent_correct
    threshold_stimulus_level = threshold
    # The threshold passed to the test was computed  by hand to correspond to PC on regular
    # sigmoids. For negative sigmoids, we'll compare this threshold with the value at 1 - PC.
    threshold_prop_correct = sigmoid.PC
    if sigmoid.negative:
        threshold_prop_correct = 1 - sigmoid.PC
    prop_correct = sigmoid(threshold_stimulus_level, threshold, width)
    np.testing.assert_allclose(prop_correct, threshold_prop_correct)

    # Test that the width is equal to the
    # |X_L - X_R| == WIDTH, with
    # with sigmoid(X_L) == ALPHA
    # and  sigmoid(X_R) == 1 - ALPHA
    prop_correct = sigmoid(stimulus_levels, threshold, width)
    # When the sigmoid is negative, it is decreasing, so we compute the width on 1-prop_correct
    # (Alternatively, we could have used `argmax` and swapped the indices)
    if sigmoid.negative:
        prop_correct = 1 - prop_correct
    idx_alpha, idx_nalpha = ((prop_correct < sigmoid.alpha).argmin(),
                             (prop_correct < (1 - sigmoid.alpha)).argmin())
    np.testing.assert_allclose(prop_correct[idx_nalpha] - prop_correct[idx_alpha], width,
                               atol=0.02)

    # Inverse sigmoid at threshold proportion correct (y-axis)
    # Expects the threshold stimulus level (x-axis).
    stimulus_level_from_inverse = sigmoid.inverse(threshold_prop_correct,
                                                  threshold=threshold,
                                                  width=width)
    np.testing.assert_allclose(stimulus_level_from_inverse, threshold_stimulus_level)

    # Expects maximal slope at a medium stimulus level
    slope = sigmoid.slope(stimulus_levels, threshold=threshold, width=width, gamma=0, lambd=0)
    assert 0.4 * len(slope) < np.argmax(np.abs(slope)) < 0.6 * len(slope)
    # Expects slope to be all positive/negative for standard/decreasing sigmoid
    if sigmoid.negative:
        assert np.all(slope < 0)
    else:
        assert np.all(slope > 0)


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


@pytest.mark.skip("still WIP")
@pytest.mark.parametrize('sigmoid_name', sigmoids.ALL_SIGMOID_NAMES)
def test_sigmoid_sanity_check(sigmoid_name):
    """ Basic sanity checks for sigmoids.

    These sanity checks test some basic relations between the parameters
    as well as rule of thumbs which can be derived from visual inspection
    of the sigmoid functions.
    """
    # fixed parameters for simple sigmoid sanity checks
    PC = 0.4
    # Threshold computed by hand to correspond to PC (for negative sigmoids, we'll compare this
    # threshold with the value at 1 - PC)
    threshold = 0.460172162722971
    alpha = 0.083

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=alpha)
    assert_sanity_checks(sigmoid, threshold=threshold)


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
def test_sigmoid_roundtrip(sigmoid_name):
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
