import numpy as np
import pytest
from functools import partial

from psignifit import likelihood
from psignifit import sigmoids
from psignifit.priors import default_priors
from psignifit.bounds import parameter_bounds
from psignifit.conf import Conf
from psignifit import utils

from .data import DATA


def setup_experiment(**kwargs):
    conf = Conf(**kwargs)

    stimulus_levels = DATA[:, 0]
    stimulus_range = (stimulus_levels.min(), stimulus_levels.max())
    width_min = np.diff(np.unique(stimulus_levels)).min()
    bounds = parameter_bounds(wmin=width_min, etype=conf.experiment_type, srange=stimulus_range,
                              alpha=conf.width_alpha, echoices=conf.experiment_choices)

    sigmoid = getattr(sigmoids, conf.sigmoid)
    sigmoid = partial(sigmoid, PC=conf.thresh_PC, alpha=conf.width_alpha)

    priors = default_priors(stimulus_range, width_min, conf.width_alpha, conf.beta_prior)
    for parameter, prior in priors.items():
        if bounds[parameter]:
            priors[parameter] = utils.normalize(prior, bounds[parameter])
    grid = utils.get_grid(bounds, conf.steps_moving_bounds)
    return DATA, sigmoid, priors, grid

@pytest.mark.parametrize(
   "experiment_type,result_shape,result_max",
    [
       ("yes/no", (30, 40, 10, 1, 20), -560.8871),
       ("3AFC", (30, 40, 10, 1, 20), -560.0022),
       ("equal asymptote", (30, 40, 10, 20), -560.8881),  # gamma dimension is None
    ]
)
def test_log_posterior(experiment_type, result_shape, result_max):
    data, sigmoid, priors, grid = setup_experiment(experiment_type=experiment_type)
    log_pp = likelihood.log_posterior(data, sigmoid, priors, grid)

    np.testing.assert_equal(log_pp.shape, result_shape)
    np.testing.assert_allclose(log_pp.max(), result_max)
