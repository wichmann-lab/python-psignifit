from itertools import product
import warnings

import numpy as np
import pytest

from psignifit import psignifit
from psignifit.sigmoids import ALL_SIGMOID_CLASSES, Gaussian
from psignifit.tools import psychometric_with_eta
from psignifit._utils import fp_error_handler


@pytest.mark.parametrize("sigmoid_class, negative", product(ALL_SIGMOID_CLASSES, [True, False]))
@fp_error_handler(over='ignore', invalid='ignore')
def test_parameter_recovery_2afc(sigmoid_class, negative):
    width = 0.3
    stim_range = [0.01, 0.01 + width * 1.1]
    threshold = stim_range[1] / 2.5
    gamma = 0.5  # 2-AFC
    lambda_ = 0.0532

    nsteps = 10
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    sigmoid = sigmoid_class(negative=negative)
    perccorr = sigmoid(stimulus_level, threshold=threshold, width=width, gamma=gamma, lambd=lambda_)

    ntrials = np.ones(nsteps) * 9000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': lambda_}

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate_MAP['lambda'], lambda_)
    assert np.isclose(res.parameter_estimate_MAP['gamma'], gamma)
    assert np.isclose(res.parameter_estimate_MAP['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['width'], width, atol=1e-4)


@pytest.mark.parametrize("eta", [0.1, 0.2, 0.3])
@fp_error_handler(over='ignore', invalid='ignore')
def test_parameter_recovery_2afc_eta(random_state, eta):
    sigmoid = "norm"
    width = 0.1
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1] / 2.5
    gamma = 0.5  # 2-AFC
    lambda_ = 0.0232

    nsteps = 200
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_,
                            sigmoid, eta=eta, random_state=random_state)

    ntrials = np.ones(nsteps) * 10000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': lambda_}

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate_MAP['lambda'], lambda_)
    assert np.isclose(res.parameter_estimate_MAP['gamma'], gamma)
    assert np.isclose(res.parameter_estimate_MAP['eta'], eta, atol=0.05)
    assert np.isclose(res.parameter_estimate_MAP['threshold'], threshold, atol=0.01)
    assert np.isclose(res.parameter_estimate_MAP['width'], width, atol=0.05)


# threshold and width can not be fixed.
@pytest.mark.parametrize("fixed_param",  ['lambda', 'gamma', 'eta', 'threshold', 'width'])
@fp_error_handler(over='ignore', invalid='ignore')
def test_parameter_recovery_fixed_params(fixed_param):
    sigmoid = Gaussian()
    width = 0.2000000000123
    stim_range = [0.001, 0.001 + width * 1.5]
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

    perccorr = sigmoid(
        sim_params["stimulus_level"],
        threshold=sim_params["threshold"],
        width=sim_params["width"],
        gamma=sim_params["gamma"],
        lambd=sim_params["lambda"],
    )

    ntrials = np.ones(nsteps) * 9000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([sim_params["stimulus_level"], hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = 'yes/no'
    options['fixed_parameters'] = {}
    # we fix it to a slightly off value, so we can check if stays fixed
    options['fixed_parameters'][fixed_param] = sim_params[fixed_param]

    with warnings.catch_warnings():
        # ignore warning about gamma behind unusual
        warnings.simplefilter("ignore")
        res = psignifit(data, **options)
    # check fixed params are not touched
    assert res.parameter_estimate_MAP[fixed_param] == sim_params[fixed_param]

    for p in ['lambda', 'gamma', 'threshold', 'width', 'eta']:
        # check all other params are estimated correctly
        assert np.isclose(res.parameter_estimate_MAP[p], sim_params[p], rtol=1e-4, atol=1 / 40), f"failed for parameter {p} for estimation with fixed: {fixed_param}."


@pytest.mark.parametrize("sigmoid_class, negative", product(ALL_SIGMOID_CLASSES, [True, False]))
@fp_error_handler(over='ignore', invalid='ignore')
def test_parameter_recovery_YN(sigmoid_class, negative):
    width = 0.3
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1]/3
    lambda_ = 0.0232
    gamma = 0.1

    nsteps = 20
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    sigmoid = sigmoid_class(negative=negative)
    perccorr = sigmoid(stimulus_level, threshold, width, gamma, lambda_)
    ntrials = np.ones(nsteps) * 900000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = 'yes/no'
    options['fixed_parameters'] = {'lambda': lambda_}

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate_MAP['lambda'], lambda_, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['gamma'], gamma, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['width'], width, atol=1e-4)


@pytest.mark.parametrize("sigmoid_class, negative", product(ALL_SIGMOID_CLASSES, [True, False]))
@fp_error_handler(over='ignore', invalid='ignore')
def test_parameter_recovery_eq_asymptote(sigmoid_class, negative):
    width = 0.3
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1]/3
    lambda_ = 0.1
    gamma = 0.1

    nsteps = 20
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    sigmoid = sigmoid_class(negative=negative)
    perccorr = sigmoid(stimulus_level, threshold, width, gamma, lambda_)
    ntrials = np.ones(nsteps) * 900000000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = 'equal asymptote'
    options['fixed_parameters'] = {}

    res = psignifit(data, **options)

    assert np.isclose(res.parameter_estimate_MAP['lambda'], lambda_, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['gamma'], gamma, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['eta'], 0, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['threshold'], threshold, atol=1e-4)
    assert np.isclose(res.parameter_estimate_MAP['width'], width, atol=1e-4)


@fp_error_handler(over='ignore', invalid='ignore')
def test_parameter_recovery_mean_estimate(random_state):
    # Check that the mean estimate also recovers the parameters with enough data
    sigmoid = "norm"
    width = 0.1
    eta = 0.25
    stim_range = [0.001, 0.001 + width * 1.1]
    threshold = stim_range[1] / 2.5
    gamma = 0.5  # 2-AFC
    lambda_ = 0.0232

    nsteps = 200
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    perccorr = psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_,
                            sigmoid, eta=eta, random_state=random_state)

    ntrials = np.ones(nsteps) * 10000
    hits = (perccorr * ntrials).astype(int)
    data = np.dstack([stimulus_level, hits, ntrials]).squeeze()

    options = {
        'sigmoid': sigmoid,
        'experiment_type': '2AFC',
        'fixed_parameters': {'lambda': lambda_},
    }
    result = psignifit(data, **options)

    expected_mean_dict = {
        'eta': eta,
        'gamma': gamma,
         'lambda': lambda_,
         'threshold': threshold,
         'width': width,
    }
    for p in expected_mean_dict.keys():
        assert np.isclose(result.parameter_estimate_mean[p], expected_mean_dict[p], atol=0.05)


@fp_error_handler(over='ignore', invalid='ignore')
def test_mean_vs_map_estimate(random_state):
    # Test a case where the mean and MAP estimates are expected to be different:
    # we generate data from a mixture of three sigmoids with different width, with
    # only a few trials. The posterior over width is broad and asymmetrical, so the
    # MAP and mean estimate are not identical.
    # The expected values of the parameters were computed independently in a notebook.

    widths = [1.1, 3.1, 6.3]
    stim_range = [0.001, 0.001 + 10 * 1.1]
    threshold = stim_range[1] / 2
    lambda_ = 0.0232
    gamma = 0.1
    num_trials = 3

    sigmoid = Gaussian()
    nsteps = 20
    stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)

    # The data is a mixture of three sigmoids
    perccorr1 = sigmoid(stimulus_level, threshold=threshold, width=widths[0], gamma=gamma, lambd=lambda_)
    perccorr2 = sigmoid(stimulus_level, threshold=threshold, width=widths[1], gamma=gamma, lambd=lambda_)
    perccorr3 = sigmoid(stimulus_level, threshold=threshold, width=widths[2], gamma=gamma, lambd=lambda_)
    perccorr = np.concatenate((perccorr1, perccorr2, perccorr3))
    levels = np.concatenate((stimulus_level, stimulus_level, stimulus_level))

    # Generate trial data
    # Fix the random state, because the expected parameters have been computed with this seed
    ntrials = np.ones(nsteps * 3, dtype=int) * num_trials
    random_state = np.random.RandomState(883)
    hits = random_state.binomial(ntrials, perccorr)

    data = np.dstack([levels, hits, ntrials]).squeeze()

    options = {
        'sigmoid': sigmoid,
        'experiment_type': 'yes/no',
        'fixed_parameters': {'lambda': 0.0232, 'gamma': 0.1},
    }
    # psignifit will complain because there are less than 3 trials per block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = psignifit(data, **options)

    expected_MAP_dict = {
        'eta': 1e-08,
        'gamma': gamma,
         'lambda': lambda_,
         'threshold': 5.459454390045572,
         'width': 3.679911898980962,
    }
    for p in expected_MAP_dict.keys():
        assert np.isclose(result.parameter_estimate_MAP[p], expected_MAP_dict[p])

    expected_mean_dict = {
        'eta': 0.07274831038039509,
        'gamma': gamma,
        'lambda': lambda_,
        'threshold': 5.429846600944022,
        'width': 4.027280086833293,
    }
    for p in expected_mean_dict.keys():
        assert np.isclose(result.parameter_estimate_mean[p], expected_mean_dict[p])
