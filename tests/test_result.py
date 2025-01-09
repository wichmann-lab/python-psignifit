import dataclasses

import numpy as np
import pytest

from psignifit import Configuration, Result, psignifit
from .fixtures import input_data


@pytest.fixture
def result():
    parameter_estimate_MAP = {
        'threshold': 1.005,
        'width': 0.005,
        'lambda': 0.0123,
        'gamma': 0.021,
        'eta': 0.0001
    }
    parameter_estimate_mean = {
        'threshold': 1.002,
        'width': 0.002,
        'lambda': 0.013,
        'gamma': 0.024,
        'eta': 0.001
    }
    confidence_intervals = {
        'threshold': {'0.95': [1.001, 1.005], '0.9': [1.005, 1.01]},
        'width': {'0.95': [0.001, 0.005], '0.9': [0.005, 0.01]},
        'lambda': {'0.95': [0.03, 0.08], '0.9': [0.1, 0.2]},
        'gamma': {'0.95': [0.03, 0.08], '0.9': [0.05, 0.2]},
        'eta': {'0.95': [0.001, 0.005], '0.9': [0.005, 0.01]},
    }
    return _build_result(parameter_estimate_MAP, parameter_estimate_mean, confidence_intervals)


def _build_result(parameter_estimate_MAP, parameter_estimate_mean, confidence_intervals):
    # We don't care about most of the input parameters of the Result object, fill them with junk
    result = Result(
        configuration=Configuration(),
        parameter_estimate_MAP=parameter_estimate_MAP,
        parameter_estimate_mean=parameter_estimate_mean,
        confidence_intervals=confidence_intervals,
        data=np.random.rand(5, 3).tolist(),
        parameter_values={
          'threshold': np.random.rand(10,).tolist(),
          'width': np.random.rand(3,).tolist(),
          'lambda': np.random.rand(5,).tolist(),
          'gamma': np.random.rand(5,).tolist(),
          'eta': np.random.rand(16,).tolist()
        },
        prior_values={
          'threshold': np.random.rand(10,).tolist(),
          'width': np.random.rand(3,).tolist(),
          'lambda': np.random.rand(5,).tolist(),
          'gamma': np.random.rand(5,).tolist(),
          'eta': np.random.rand(16,).tolist()
        },
        marginal_posterior_values={
          'threshold': np.random.rand(10,).tolist(),
          'width': np.random.rand(3,).tolist(),
          'lambda': np.random.rand(5,).tolist(),
          'gamma': np.random.rand(5,).tolist(),
          'eta': np.random.rand(16,).tolist()
        },
        debug={'posteriors': np.random.rand(5, ).tolist()},
    )
    return result


def test_from_to_result_dict(result):
    result_dict = result.as_dict()

    assert isinstance(result_dict, dict)
    assert isinstance(result_dict['configuration'], dict)
    for field in dataclasses.fields(Result):
        assert field.name in result_dict

    assert result == Result.from_dict(result_dict)
    assert result.configuration == Configuration.from_dict(result_dict['configuration'])


def test_threshold_raises_error_when_outside_valid_range(result):
    # proportion correct lower than gamma
    proportion_correct = np.array([result.parameter_estimate_MAP['gamma'] / 2.0])
    with pytest.raises(ValueError):
        result.threshold(proportion_correct)
    # proportion correct higher than gamma
    proportion_correct = np.array([result.parameter_estimate_MAP['lambda'] + 1e-4])
    with pytest.raises(ValueError):
        result.threshold(proportion_correct)


def test_threshold_slope(result):
    proportion_correct = np.linspace(0.2, 0.5, num=1000)
    stimulus_levels = result.threshold(proportion_correct, return_ci=False)
    np.testing.assert_allclose(
        result.slope(stimulus_levels),
        result.slope_at_proportion_correct(proportion_correct)
    )


def test_threshold_slope_ci_scaled(result):
    proportion_correct = [0.4, 0.5, 0.7]
    _, threshold_cis = result.threshold(proportion_correct, return_ci=True, unscaled=False)

    expected = {
        '0.95': [[1.000918, 1.001, 1.001171], [1.00454 , 1.005, 1.005969]],
        '0.9': [[1.004661, 1.005112, 1.006097], [1.008691, 1.01, 1.012941]],
    }
    assert list(threshold_cis.keys()) == ['0.95', '0.9']
    for coverage_key, cis in threshold_cis.items():
        # one CI per proportion_correct
        assert threshold_cis[coverage_key][0].shape[0] == len(proportion_correct)
        assert threshold_cis[coverage_key][1].shape[0] == len(proportion_correct)
        np.testing.assert_allclose(
            threshold_cis[coverage_key][0],
            expected[coverage_key][0],
            atol=1e-6
        )
        np.testing.assert_allclose(
            threshold_cis[coverage_key][1],
            expected[coverage_key][1],
            atol=1e-6
        )


def test_threshold_slope_ci_unscaled(result):
    proportion_correct = [0.4, 0.5, 0.7]
    _, threshold_cis = result.threshold(proportion_correct, return_ci=True, unscaled=True)

    expected = {
        '0.95': [[1.000923, 1.001, 1.001159], [1.004615, 1.005, 1.005797]],
        '0.9': [[1.004615, 1.005, 1.005797], [1.00923, 1.01, 1.011594]],
    }
    assert list(threshold_cis.keys()) == ['0.95', '0.9']
    for coverage_key, cis in threshold_cis.items():
        # one CI per proportion_correct
        assert threshold_cis[coverage_key][0].shape[0] == len(proportion_correct)
        assert threshold_cis[coverage_key][1].shape[0] == len(proportion_correct)
        np.testing.assert_allclose(
            threshold_cis[coverage_key][0],
            expected[coverage_key][0],
            atol=1e-6
        )
        np.testing.assert_allclose(
            threshold_cis[coverage_key][1],
            expected[coverage_key][1],
            atol=1e-6
        )


def test_threshold_value():
    # This test fails before PR #139
    lambda_ = 0.1
    gamma = 0.2
    width = 1.0 - 0.05*2
    parameter_estimate = {
        'threshold': 0.5,
        'width': width,
        'lambda': lambda_,
        'gamma': gamma,
        'eta': 0.0,
    }
    confidence_intervals = {
        'threshold': {'0.95': [0.5, 0.5]},
        'width': {'0.95': [width, width]},
        'lambda': {'0.95': [0.05, 0.2]},
        'gamma': {'0.95': [0.1, 0.3]},
        'eta': {'0.95': [0.0, 0.0]},
    }
    result = _build_result(parameter_estimate, parameter_estimate, confidence_intervals)

    # The threshold at the middle of the gamma-to-(1-lambda) range must be 0.5 for a Gaussian
    thr = result.threshold(
        proportion_correct=np.array([(1 - lambda_ - gamma) / 2.0 + gamma]),
        unscaled=False, return_ci=False,
    )
    expected_thr = np.array(0.5)  # by construction
    np.testing.assert_allclose(thr, expected_thr)

    # ... except when unscaled is True
    thr = result.threshold(
        proportion_correct=np.array([(1 - lambda_ - gamma) / 2.0 + gamma]),
        unscaled=True, return_ci=False,
    )
    expected_thr = np.array([0.5343785])  # computed by hand
    np.testing.assert_allclose(thr, expected_thr)

    # Compare to results computed by hand
    thr, thr_ci = result.threshold(
        proportion_correct=np.array([0.7]),
        unscaled=False, return_ci=True,
    )
    expected_thr = np.array([0.654833])  # Computed by hand
    np.testing.assert_allclose(thr, expected_thr, atol=1e-4)
    expected_thr_ci = {'0.95': [[0.648115], [0.730251]]}  # Computed by hand
    np.testing.assert_allclose(thr_ci['0.95'], expected_thr_ci['0.95'], atol=1e-4)


def _close_numpy_dict(first, second):
    """ Test if two dicts of numpy arrays are equal"""
    if first.keys() != second.keys():
        return False
    return np.all([np.all(np.isclose(first[key], second[key])) for key in first])

def _equal_numpy_dict(first, second):
    """ Test if two dicts of numpy arrays are equal"""
    if first.keys() != second.keys():
        return False
    return np.all([np.all(first[key] == second[key]) for key in first])

def test_save_load_result_json(result, tmp_path):
    result_file = tmp_path / 'result.json'

    assert not result_file.exists()
    result.save_json(result_file)
    assert result_file.exists()
    other = Result.load_json(result_file)

    assert result.parameter_estimate_MAP == other.parameter_estimate_MAP
    assert result.configuration == other.configuration
    assert result.confidence_intervals == other.confidence_intervals
    assert np.all(np.isclose(result.data, other.data))
    assert _close_numpy_dict(
        result.parameter_values, other.parameter_values)
    assert _close_numpy_dict(result.prior_values, other.prior_values)
    assert _close_numpy_dict(
        result.marginal_posterior_values, other.marginal_posterior_values)


def test_get_parameter_estimate(result):
    estimate = result.get_parameter_estimate(estimate_type='MAP')
    assert _close_numpy_dict(estimate, result.parameter_estimate_MAP)

    estimate = result.get_parameter_estimate(estimate_type='mean')
    assert _close_numpy_dict(estimate, result.parameter_estimate_mean)

    with pytest.raises(ValueError):
        result.get_parameter_estimate(estimate_type='foo')

def test_parameter_estimate_property(result):
    estimate = result.parameter_estimate
    # verify that we get the parameter estimate MAP by default
    assert _equal_numpy_dict(estimate, result.parameter_estimate_MAP)
    assert not _equal_numpy_dict(estimate, result.parameter_estimate_mean)
    # verify that we get the mean parameter estimate if we change the estimate type
    result.configuration.estimate_type = 'mean'
    estimate = result.parameter_estimate
    assert not _equal_numpy_dict(estimate, result.parameter_estimate_MAP)
    assert _equal_numpy_dict(estimate, result.parameter_estimate_mean)

def test_estimate_type_default(result):
    result.configuration.estimate_type = 'MAP'
    estimate = result.get_parameter_estimate()
    assert _close_numpy_dict(estimate, result.parameter_estimate_MAP)

    result.configuration.estimate_type = 'mean'
    estimate = result.get_parameter_estimate()
    assert _close_numpy_dict(estimate, result.parameter_estimate_mean)


def test_standard_parameter_estimate():
    width = 2.1
    threshold = 0.87
    parameter_estimate = {
        'threshold': threshold,
        'width': width,
        'lambda': 0.0,
        'gamma': 0.0,
        'eta': 0.0,
    }
    confidence_intervals = {}
    result = _build_result(parameter_estimate, parameter_estimate, confidence_intervals)

    # For a Gaussian sigmoid with alpha=0.05, PC=0.5
    expected_loc = threshold
    # 1.644853626951472 is the normal PPF at alpha=0.95
    expected_scale = width / (2 * 1.644853626951472)

    loc, scale = result.standard_parameter_estimate()
    np.testing.assert_allclose(loc, expected_loc)
    np.testing.assert_allclose(scale, expected_scale)


def test_threshold_bug_172(input_data):
    # Reproduce bug in issue #172

    options = {
        'sigmoid': 'norm',
        'experiment_type': '2AFC'
    }

    result = psignifit(input_data, **options)
    threshold = result.threshold(0.9, return_ci=False)  # which should be 0.0058

    expected_threshold = 0.0058
    np.testing.assert_allclose(threshold, expected_threshold, atol=1e-4)


def test_posterior_samples_raises_if_not_debug(input_data):
    result = psignifit(input_data[:3,:])
    with pytest.raises(ValueError):
        result.posterior_samples(n_samples=10)


def test_posterior_samples(result, random_state):

    params = ['eta', 'gamma', 'lambda', 'threshold', 'width']
    parameter_values = {
        'eta': np.array([0]),
        'gamma': np.array([0, 1, 2]),
        'lambda': np.array([0, 1]),
        'threshold': np.array([0]),
        'width': np.array([0, 1])
    }

    # Build a random posterior distribution
    posterior_shape = (1, 3, 2, 1, 2)
    posterior = random_state.uniform(size=posterior_shape)
    posterior = posterior / posterior.sum()

    # Inject in the Result object
    result.parameter_values = parameter_values
    result.debug['posteriors'] = posterior

    # Draw samples from the posterior
    n_samples = 150234
    samples = result.posterior_samples(n_samples=n_samples, random_state=random_state)

    # Check that the empirical posterior from the samples matches the random posterior
    for param in params:
        assert samples[param].shape == (n_samples,)

    counts = np.zeros(posterior_shape)
    for idx in range(n_samples):
        counts[
            samples['eta'][idx],
            samples['gamma'][idx],
            samples['lambda'][idx],
            samples['threshold'][idx],
            samples['width'][idx],
        ] += 1

    empirical_posterior = counts / n_samples
    np.testing.assert_allclose(empirical_posterior, posterior, atol=1e-2)


def test_result_proportion_correct(result):
    exp = np.linspace(0.2, 0.5, num=1000)
    stimulus_levels = result.threshold(exp, return_ci=False)
    out = result.proportion_correct(stimulus_levels, with_eta=False)
    np.testing.assert_allclose(exp, out)
    # we can't really test with_eta properly, but we can make sure
    # that given a small eta the output is still quite close
    out = result.proportion_correct(stimulus_levels, with_eta=True)
    np.testing.assert_allclose(exp, out, rtol=1e-3)
