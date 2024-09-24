import dataclasses

import pytest
import numpy as np

from psignifit import Result
from psignifit import Configuration


@pytest.fixture
def result():
    return Result(configuration=Configuration(),
                  parameter_estimate={
                      'threshold': 0.005,
                      'width': 0.005,
                      'lambda': 0.0123,
                      'gamma': 0.021,
                      'eta': 0.0001
                  },
                  confidence_intervals={
                      'threshold': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
                      'width': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
                      'lambda': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
                      'gamma': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]],
                      'eta': [[0.001, 0.005], [0.005, 0.01], [0.1, 0.2]]
                  },
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
                  debug={'posteriors': np.random.rand(5, ).tolist()})


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
    np.testing.assert_allclose(result.slope(stimulus_levels),
                               result.slope_at_proportion_correct(proportion_correct))


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
