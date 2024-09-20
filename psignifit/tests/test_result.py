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
                      'lambda': 1.-7,
                      'gamma': 0.5,
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
                  posterior_mass=np.random.rand(5, ))


def test_from_to_result_dict(result):
    result_dict = result.as_dict()

    assert isinstance(result_dict, dict)
    assert isinstance(result_dict['configuration'], dict)
    for field in dataclasses.fields(Result):
        assert field.name in result_dict

    assert result == Result.from_dict(result_dict)
    assert result.configuration == Configuration.from_dict(result_dict['configuration'])


def test_threshold_slope(result):
    with pytest.raises(ValueError):
        #  PC outside of sigmoid
        percentage_correct = np.linspace(1e-12, 1 - 1e-12, num=1000)
        result.threshold(percentage_correct)
    percentage_correct = np.linspace(0.2, 0.5, num=1000)
    stimulus_levels, confidence_intervals = result.threshold(percentage_correct)
    np.testing.assert_allclose(result.slope(stimulus_levels),
                               result.slope_at_percentage_correct(percentage_correct))


def test_save_load_result_json(result, tmp_path):
    result_file = tmp_path / 'result.json'

    assert not result_file.exists()
    result.save_json(result_file)
    assert result_file.exists()
    print(result)
    assert result == Result.load_json(result_file)
