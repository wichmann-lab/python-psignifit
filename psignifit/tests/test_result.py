import dataclasses

import numpy as np
import pytest

from psignifit import Configuration
from psignifit import Result


@pytest.fixture
def result():
    parameter_estimate = {
        'threshold': 0.005,
        'width': 0.005,
        'lambda': 0.0123,
        'gamma': 0.021,
        'eta': 0.0001
    }
    confidence_intervals = {
        'threshold': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
        'width': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
        'lambda': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
        'gamma': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
        'eta': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]]
    }
    return _build_result(parameter_estimate, confidence_intervals)


def _build_result(parameter_estimate, confidence_intervals):
    # We don't care about most of the input parameters of the Result object, fill them with junk
    result = Result(
        configuration=Configuration(),
        parameter_estimate=parameter_estimate,
        parameter_estimate_mean=parameter_estimate,
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
    proportion_correct = np.array([result.parameter_estimate['gamma'] / 2.0])
    with pytest.raises(ValueError):
        result.threshold(proportion_correct)
    # proportion correct higher than gamma
    proportion_correct = np.array([result.parameter_estimate['lambda'] + 1e-4])
    with pytest.raises(ValueError):
        result.threshold(proportion_correct)


def test_threshold_slope(result):
    proportion_correct = np.linspace(0.2, 0.5, num=1000)
    stimulus_levels, confidence_intervals = result.threshold(proportion_correct)
    np.testing.assert_allclose(
        result.slope(stimulus_levels),
        result.slope_at_proportion_correct(proportion_correct)
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
        'threshold': [[0.5, 0.5]],
        'width': [[width, width]],
        'lambda': [[0.05, 0.2]],
        'gamma': [[0.1, 0.3]],
        'eta': [[0.0, 0.0]]
    }
    result = _build_result(parameter_estimate, confidence_intervals)

    # The threshold at the middle of the gamma-to-(1-lambda) range must be 0.5 for a Gaussian
    thr, thr_ci = result.threshold(
        proportion_correct=np.array([(1 - lambda_ - gamma) / 2.0 + gamma]),
        unscaled=False, return_ci=True,
    )

    expected_thr = np.array(0.5)
    np.testing.assert_allclose(thr, expected_thr)

    # Compare to results computed by hand
    thr, thr_ci = result.threshold(
        proportion_correct=np.array([0.7]),
        unscaled=False, return_ci=True,
    )
    expected_thr = np.array([0.654833])  # Computed by hand
    np.testing.assert_allclose(thr, expected_thr, atol=1e-4)
    expected_thr_ci = [[[0.648115], [0.730251]]]  # Computed by hand
    np.testing.assert_allclose(thr_ci, expected_thr_ci, atol=1e-4)


def _close_numpy_dict(first, second):
    """ Test if two dicts of numpy arrays are equal"""
    if first.keys() != second.keys():
        return False
    return np.all(np.isclose(first[key], second[key]) for key in first)


def test_save_load_result_json(result, tmp_path):
    result_file = tmp_path / 'result.json'

    assert not result_file.exists()
    result.save_json(result_file)
    assert result_file.exists()
    other = Result.load_json(result_file)

    assert result.parameter_estimate == other.parameter_estimate
    assert result.configuration == other.configuration
    assert result.confidence_intervals == other.confidence_intervals
    assert np.all(np.isclose(result.data, other.data))
    assert _close_numpy_dict(
        result.parameter_values, other.parameter_values)
    assert _close_numpy_dict(result.prior_values, other.prior_values)
    assert _close_numpy_dict(
        result.marginal_posterior_values, other.marginal_posterior_values)
