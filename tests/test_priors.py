import numpy as np
import pytest

import psignifit._priors
from psignifit import _priors


def test_check_priors():
    stimulus_range = (0., 1.)
    width_min = 0.1
    alpha = 0.13
    prior_dict = {}
    for parameter in ['threshold', 'width', 'lambda', 'gamma', 'eta']:
        prior_dict[parameter] = _priors.default_prior(parameter, stimulus_range, width_min, width_alpha=alpha, beta=10)

    # should not fail for default priors
    _priors.check_priors(prior_dict, stimulus_range, width_min)

    # should fail for parameters so far outside prior that the prior over the stimulus range  is 0
    with pytest.raises(AssertionError):
        _priors.check_priors(prior_dict, (10 + stimulus_range[0], 10 + stimulus_range[1]), width_min)

    # should fail for non-positive, non-finite, or missing priors
    with pytest.raises(AssertionError):
        prior_dict['threshold'] = lambda x: np.zeros_like(x)
        _priors.check_priors(prior_dict, stimulus_range, width_min)

    with pytest.raises(AssertionError):
        prior_dict['threshold'] = lambda x: np.full_like(x, np.inf)
        _priors.check_priors(prior_dict, stimulus_range, width_min)

    del prior_dict['threshold']
    _priors.check_priors(prior_dict, stimulus_range, width_min)


def test_normalize_sin():
    # for sin the integral in (0, pi/2) is 1, so the norm(sin) == sin
    x = np.linspace(0, np.pi / 2, 100)
    norm = psignifit._priors.normalize_prior(np.sin, (0, np.pi / 2))
    assert np.allclose(np.sin(x), norm(x))

    # For a fixed value, the integral should be one
    norm = psignifit._priors.normalize_prior(np.sin, (1, 1))
    assert np.allclose([1], norm(1))


def test_normalize():
    # a constant function

    def func(x):
        return np.ones_like(x)

    # the integral is length of x, so the normalized function should return 1/len(x)
    x = np.arange(11)
    norm = psignifit._priors.normalize_prior(func, (0, 10))
    assert np.allclose(1. / 10, norm(x))

    # For a fixed value, the integral should be one
    norm = psignifit._priors.normalize_prior(func, (1, 1))
    assert np.allclose([1], norm(1))
