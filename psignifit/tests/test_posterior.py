import numpy as np
import pytest

import psignifit._parameter
import psignifit._posterior
import psignifit._priors
from psignifit import sigmoids
from psignifit import Configuration
from psignifit._priors import default_priors
from psignifit._parameter import parameter_bounds
from psignifit import _posterior
from psignifit import _utils

from .data import DATA


def setup_experiment(**kwargs):
    conf = Configuration(**kwargs)

    stimulus_levels = DATA[:, 0]
    stimulus_range = (stimulus_levels.min(), stimulus_levels.max())
    width_min = np.diff(np.unique(stimulus_levels)).min()
    bounds = parameter_bounds(min_width=width_min, experiment_type=conf.experiment_type, stimulus_range=stimulus_range,
                              alpha=conf.width_alpha, nafc_choices=conf.experiment_choices)

    sigmoid = sigmoids.sigmoid_by_name(conf.sigmoid, PC=conf.thresh_PC, alpha=conf.width_alpha)

    priors = default_priors(stimulus_range, width_min, conf.width_alpha, conf.beta_prior)
    for parameter, prior in priors.items():
        if bounds[parameter]:
            priors[parameter] = psignifit._priors.normalize_prior(prior, bounds[parameter])
    grid = psignifit._parameter.parameter_grid(bounds, conf.steps_moving_bounds)
    return DATA, sigmoid, priors, grid


@pytest.mark.parametrize(
    "experiment_type,result_shape,result_max",
    [
        ("yes/no", (15, 10, 10, 25, 30), -557.5108),
        ("3AFC", (20, 1, 10, 30, 40), -560.0022),
        ("equal asymptote", (20, 1, 10, 30, 40), -560.8881),  # gamma is none in grid
    ]
)
def test_log_posterior(experiment_type, result_shape, result_max):
    data, sigmoid, priors, grid = setup_experiment(experiment_type=experiment_type)
    if experiment_type == "equal asymptote":
        assert grid['gamma'] is None

    log_pp = _posterior.log_posterior(data, sigmoid, priors, grid)

    np.testing.assert_equal(log_pp.shape, result_shape)
    np.testing.assert_allclose(log_pp.max(), result_max)


def test_log_posterior_zero_eta():
    data, sigmoid, priors, grid = setup_experiment(experiment_type='yes/no')
    grid['eta'] = np.array([0])

    log_pp = _posterior.log_posterior(data, sigmoid, priors, grid)

    np.testing.assert_equal(log_pp.shape, (1, 10, 10, 25, 30))
    np.testing.assert_allclose(log_pp.max(), -559.8134)


MAX = .097163


@pytest.mark.parametrize(
    "experiment_type,result_shape,result_max",
    [
        ("yes/no", (15, 10, 10, 25, 30), .102878),
        ("3AFC", (20, 1, 10, 30, 40), .080135),
        ("equal asymptote", (20, 1, 10, 30, 40), MAX),  # gamma is none in grid
    ]
)
def test_posterior_grid(experiment_type, result_shape, result_max):
    data, sigmoid, priors, grid = setup_experiment(experiment_type=experiment_type)

    posterior, max_grid = _posterior.posterior_grid(data, sigmoid, priors, grid)

    np.testing.assert_equal(posterior.shape, result_shape)
    np.testing.assert_allclose(posterior.max(), result_max, atol=0.001)
    if experiment_type == "equal asymptote":
        assert grid['gamma'] is None
        assert max_grid['gamma'] is None


@pytest.mark.parametrize(
    "experiment_type,result_shape,result_max",
    [
        ("yes/no", (15, 10, 10, 25, 30), MAX),
        ("3AFC", (20, 1, 10, 30, 40), MAX),
        ("equal asymptote", (20, 1, 10, 30, 40), MAX),  # gamma is none in grid
    ]
)
def test_max_posterior(experiment_type, result_shape, result_max):
    data, sigmoid, priors, grid = setup_experiment(experiment_type=experiment_type)
    __, init_param = _posterior.posterior_grid(data, sigmoid, priors, grid)

    fixed_param = {'threshold': 0.5}
    if experiment_type == "equal asymptote":
        assert grid['gamma'] is None
        with pytest.raises(_utils.PsignifitException):
            _posterior.maximize_posterior(data, init_param, fixed_param, sigmoid, priors)

    fixed_param = {'threshold': 0.5, 'gamma': None}
    max_param = _posterior.maximize_posterior(data, init_param, fixed_param, sigmoid, priors)

    print(max_param)
    for key, fixed_value in fixed_param.items():
        np.testing.assert_equal(max_param[key], fixed_value)

    if experiment_type == "equal asymptote":
        assert max_param['gamma'] is None


def test_integral_weights():
    # Simple case
    weights = psignifit._posterior.integral_weights([[0, 1], [0, 1], [0, 1]])
    assert weights.sum() == 1.
    assert weights.shape == (2, 2, 2)
    np.testing.assert_equal(weights, np.full((2, 2, 2), 1 / 8))

    # Various differences
    weights = psignifit._posterior.integral_weights([[0, 1], [0, 2], [2, 6]])
    assert weights.sum() == 8
    assert weights.shape == (2, 2, 2)
    np.testing.assert_equal(weights, np.full((2, 2, 2), 1))

    # Various number of steps and None entries
    weights = psignifit._posterior.integral_weights([[0, 1], [5, 6, 9], None, [5]])
    assert weights.sum() == 3.
    assert weights.shape == (2, 3, 1, 1)
