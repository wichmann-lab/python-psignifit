from psignifit._sigmoids import Gaussian
import numpy as np
import pytest
from psignifit import psignifit

def test_parameter_recovery():
    stim_range = [0.001, 0.01]
    nsteps = 19
    threshold = stim_range[1]/2
    # we want the threshold to be in the middle of the range, but I dont know
    # how to get the scaled sigmoid directly.
    PC = 0.5
    # alpha is related to the width: width = X_(1-alpha) - X_(alpha)
    # i dont understand why we are passing both alpha and width then...
    alpha = 0.05
    width = 0.01

    # levels column
    stimulus_level = np.linspace(stim_range[0],stim_range[1],nsteps)
    # hits and ntrials columns
    g = Gaussian(PC=PC, alpha=alpha, negative=False, logspace=False)
    # the +1 and /2 are to convert the range from 0 to 1 to 0.5 to 1
    perccorr = (g(stimulus_level, threshold=threshold, width=width) + 1) / 2
    ntrials = np.ones(19)*9000000
    hits = (perccorr * ntrials).astype(int)
    # data
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = 'norm'  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': 0,
                                   'gamma': 0.5}
    options["stimulus_range"] = [0,0.01]

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate['lambda'], 0)
    assert np.isclose(res.parameter_estimate['gamma'], 0.5)
    assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate['width'], width, atol=1e-4)
    
    # TODO: Also check for warnings
    
