import numpy as np
import pytest

from psignifit import psignifit
from psignifit._sigmoids import Gaussian, ALL_SIGMOID_NAMES, sigmoid_by_name


def psychometric(stimulus_level, threshold, width, gamma, lambda_,
                 sigmoid_name):
    """ Psychometric function aka percent correct function.

    Generates percent correct values for a range of stimulus levels given a
    sigmoid.
    Implementation of Eq 1 in Schuett, Harmeling, Macke and Wichmann (2016)

    Parameters:
        stimulus_level: array
          Values of the stimulus value
        threshold: float
            Threshold of the psychometric function
        width: float
            Width of the psychometric function
        gamma: float
            Guess rate
        lambda_: float
            Lapse rate
        sigmoid: callable
            Sigmoid function to use. Default is Gaussian

    Returns:
        psi: array
            Percent correct values for each stimulus level

    """
    # we use the defaults for pc and alpha in the sigmoids:
    # pc = 0.5
    # alpha = 0.05
    sigmoid = sigmoid_by_name(sigmoid_name)
    sigmoid_values = sigmoid(stimulus_level, threshold=threshold, width=width)
    psi = gamma + (1.0 - lambda_ - gamma) * sigmoid_values
    return psi


@pytest.mark.parametrize("sigmoid", list(ALL_SIGMOID_NAMES))
def test_parameter_recovery_2afc(sigmoid):
    width = 0.3
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1]/3
    gamma = 0.5  # 2-AFC
    lambda_ = 0.0232

    nsteps = 50
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric(stimulus_level, threshold, width, gamma, lambda_,
                            sigmoid)
    ntrials = np.ones(nsteps) * 9000000
    hits = (perccorr * ntrials).astype(int)
    # data
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': lambda_,
                                   'gamma': gamma}
    options["stimulus_range"] = stim_range

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate['lambda'], lambda_)
    assert np.isclose(res.parameter_estimate['gamma'], gamma)
    assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate['width'], width, atol=1e-4)






    # TODO: Also check for warnings
    # TODO: check different fixed params and different options
    # TODO: add simulation test for Y/N paradigm
    # TODO: simulation test with different ETAs?
    
    
    
    # 1 block per stimulus level
    # number of trials no should be (rule of thumb) 40-50 trials. No more than 200, since it assymptotes
    
