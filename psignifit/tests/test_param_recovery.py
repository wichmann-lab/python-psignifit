import numpy as np
import pytest
from scipy import stats

from psignifit import psignifit
from psignifit.sigmoids import ALL_SIGMOID_NAMES
from psignifit.tools import psychometric

RANDOMSTATE = np.random.RandomState(837400)


@pytest.mark.parametrize("sigmoid", list(ALL_SIGMOID_NAMES))
def test_parameter_recovery_2afc(sigmoid):
    width = 0.3
    stim_range = [0.01, 0.01 + width * 1.1]
    threshold = stim_range[1] / 2.5
    gamma = 0.5  # 2-AFC
    lambda_ = 0.0532

    nsteps = 10
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric(stimulus_level, threshold, width, gamma, lambda_, sigmoid)
    ntrials = np.ones(nsteps) * 9000000
    hits = (perccorr * ntrials).astype(int)
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


@pytest.mark.parametrize("eta", [0.1, 0.2, 0.3])
def test_parameter_recovery_2afc_eta(eta):
    sigmoid = "norm"
    width = 0.1
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1] / 2.5
    gamma = 0.5  # 2-AFC
    lambda_ = 0.0232

    nsteps = 100
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_,
                            sigmoid, eta=eta, random_state=RANDOMSTATE)

    ntrials = np.ones(nsteps) * 500
    hits = (perccorr * ntrials).astype(int)
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
    assert np.isclose(res.parameter_estimate['eta'], eta, atol=0.05)
    assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=0.01)
    assert np.isclose(res.parameter_estimate['width'], width, atol=0.05)


# threshold and width can not be fixed.
@pytest.mark.parametrize("fixed_param",  ['lambda', 'gamma', 'eta', 'threshold', 'width'])
def test_parameter_recovery_fixed_params(fixed_param):
    sigmoid = "norm"
    width = 0.3000000000123
    stim_range = [0.001, 0.001 + width * 1.1]
    nsteps = 10
    sim_params = {
        "width" : width,
        "stim_range" : stim_range,
        "threshold" : stim_range[1]/3 + 0.000006767,
        "gamma" : 0.5000000543,
        "lambda" : 0.1000000978,
        "nsteps" : nsteps,
        "eta": 0,
        "stimulus_level": np.linspace(stim_range[0], stim_range[1], nsteps)
    }

    perccorr = psychometric(sim_params["stimulus_level"],
                            sim_params["threshold"],
                            sim_params["width"],
                            sim_params["gamma"],
                            sim_params["lambda"],
                            sigmoid)

    ntrials = np.ones(nsteps) * 9000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([sim_params["stimulus_level"], hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = 'yes/no'
    options["stimulus_range"] = stim_range
    options['fixed_parameters'] = {}
    # we fix it to a slightly off value, so we can check if stays fixed
    options['fixed_parameters'][fixed_param] = sim_params[fixed_param]

    res = psignifit(data, **options)
    # check fixed params are not touched
    assert res.parameter_estimate[fixed_param] == sim_params[fixed_param]

    for p in ['lambda', 'gamma', 'threshold', 'width', 'eta']:
        # check all other params are estimated correctly
        #print(p)
        assert np.isclose(res.parameter_estimate[p], sim_params[p], rtol=1e-4, atol=1/40),  f"failed for parameter {p} for estimation with fixed: {fixed_param}."


@pytest.mark.parametrize("sigmoid", list(ALL_SIGMOID_NAMES))
def test_parameter_recovery_YN(sigmoid):
    width = 0.3
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1]/3
    lambda_ = 0.0232
    gamma = 0.1

    nsteps = 20
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric(stimulus_level, threshold, width, gamma, lambda_,
                            sigmoid)
    ntrials = np.ones(nsteps) * 900000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = 'yes/no'
    options['fixed_parameters'] = {'lambda': lambda_}
    options["stimulus_range"] = stim_range

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate['lambda'], lambda_, atol=1e-4)
    assert np.isclose(res.parameter_estimate['gamma'], gamma, atol=1e-4)
    assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate['width'], width, atol=1e-4)


@pytest.mark.parametrize("sigmoid", list(ALL_SIGMOID_NAMES))
def test_parameter_recovery_eq_asymptote(sigmoid):
    width = 0.3
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1]/3
    lambda_ = 0.1
    gamma = 0.1

    nsteps = 20
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric(stimulus_level, threshold, width, gamma, lambda_,
                            sigmoid)
    ntrials = np.ones(nsteps) * 900000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = 'equal asymptote'
    options['fixed_parameters'] = {}
    options["stimulus_range"] = stim_range

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate['lambda'], lambda_, atol=1e-4)
    assert np.isclose(res.parameter_estimate['gamma'], gamma, atol=1e-4)
    assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate['width'], width, atol=1e-4)
