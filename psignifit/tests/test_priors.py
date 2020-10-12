import numpy as np
import pytest
from psignifit import priors


def test_check_priors():
    stimulus_range = (0., 1.)
    width_min = 0.1
    prior_dict = priors.default_priors(stimulus_range=stimulus_range,
                                       width_min=width_min, width_alpha=0.05, beta=10)

    # should not fail for default priors
    priors.check_priors(prior_dict, stimulus_range, width_min)

    # should fail for parameters not matching default priors
    with pytest.raises(AssertionError):
        priors.check_priors(prior_dict, stimulus_range, 10 * width_min)
        priors.check_priors(prior_dict, (1 + stimulus_range[0], 1 + stimulus_range[1]), width_min)

    # should fail for non-positive, non-finite, or missing priors
    with pytest.raises(AssertionError):
        prior_dict['threshold'] = lambda x: np.zeros_like(x)
        priors.check_priors(prior_dict, stimulus_range, width_min)

        prior_dict['threshold'] = lambda x: np.fill_like(x, np.inf)
        priors.check_priors(prior_dict, stimulus_range, width_min)

        del prior_dict['threshold']
        priors.check_priors(prior_dict, stimulus_range, width_min)
