import numpy as np
import pytest
from functools import partial

from psignifit import likelihood
from psignifit import sigmoids
from psignifit.priors import default_priors
from psignifit.bounds import parameter_bounds
from psignifit.configuration import Configuration
from psignifit import utils

from .data import DATA


def setup_experiment(**kwargs):
    conf = Configuration(**kwargs)

    stimulus_levels = DATA[:, 0]
    stimulus_range = (stimulus_levels.min(), stimulus_levels.max())
    width_min = np.diff(np.unique(stimulus_levels)).min()
    bounds = parameter_bounds(wmin=width_min, etype=conf.experiment_type, srange=stimulus_range,
                              alpha=conf.width_alpha, echoices=conf.experiment_choices)

    sigmoid = sigmoids.sigmoid_by_name(conf.sigmoid, PC=conf.thresh_PC, alpha=conf.width_alpha)

    priors = default_priors(stimulus_range, width_min, conf.width_alpha, conf.beta_prior)
    for parameter, prior in priors.items():
        if bounds[parameter]:
            priors[parameter] = utils.normalize(prior, bounds[parameter])
    grid = utils.get_grid(bounds, conf.steps_moving_bounds)
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

    log_pp = likelihood.log_posterior(data, sigmoid, priors, grid)

    np.testing.assert_equal(log_pp.shape, result_shape)
    np.testing.assert_allclose(log_pp.max(), result_max)


def test_log_posterior_zero_eta():
    data, sigmoid, priors, grid = setup_experiment(experiment_type='yes/no')
    grid['eta'] = np.array([0])

    log_pp = likelihood.log_posterior(data, sigmoid, priors, grid)

    np.testing.assert_equal(log_pp.shape, (1, 10, 10, 25, 30))
    np.testing.assert_allclose(log_pp.max(), -559.8134)


@pytest.mark.parametrize(
    "experiment_type,result_shape,result_max",
    [
        ("yes/no", (15, 10, 10, 25, 30), 1.),
        ("3AFC", (20, 1, 10, 30, 40), 1.),
        ("equal asymptote", (20, 1, 10, 30, 40), 1.),  # gamma is none in grid
    ]
)
def test_posterior_grid(experiment_type, result_shape, result_max):
    data, sigmoid, priors, grid = setup_experiment(experiment_type=experiment_type)

    posterior, max_grid = likelihood.posterior_grid(data, sigmoid, priors, grid)

    np.testing.assert_equal(posterior.shape, result_shape)
    np.testing.assert_allclose(posterior.max(), result_max)
    if experiment_type == "equal asymptote":
        assert grid['gamma'] is None
        assert max_grid['gamma'] is None


@pytest.mark.parametrize(
    "experiment_type,result_shape,result_max",
    [
        ("yes/no", (15, 10, 10, 25, 30), 1.),
        ("3AFC", (20, 1, 10, 30, 40), 1.),
        ("equal asymptote", (20, 1, 10, 30, 40), 1.),  # gamma is none in grid
    ]
)
def test_max_posterior(experiment_type, result_shape, result_max):
    data, sigmoid, priors, grid = setup_experiment(experiment_type=experiment_type)
    __, init_param = likelihood.posterior_grid(data, sigmoid, priors, grid)

    fixed_param = {'threshold': 0.5}
    if experiment_type == "equal asymptote":
        assert grid['gamma'] is None
        with pytest.raises(utils.PsignifitException):
            likelihood.max_posterior(data, init_param, fixed_param, sigmoid, priors)

    fixed_param = {'threshold': 0.5, 'gamma': None}
    max_param = likelihood.max_posterior(data, init_param, fixed_param, sigmoid, priors)

    print(max_param)
    for key, fixed_value in fixed_param.items():
        np.testing.assert_equal(max_param[key], fixed_value)

    if experiment_type == "equal asymptote":
        assert max_param['gamma'] is None
