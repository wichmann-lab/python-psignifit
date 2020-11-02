import dataclasses

import pytest

from psignifit.result import Result
from psignifit.configuration import Configuration


@pytest.fixture
def result():
    return Result(configuration=Configuration(),
                  sigmoid_parameters={
                      'threshold': 0.005,
                      'width': 0.005,
                      'lambda': 1.-7,
                      'gamma': 0.5,
                      'eta': 0.0001
                  })


def test_from_to_result_dict(result):
    result_dict = result.as_dict()

    assert isinstance(result_dict, dict)
    assert isinstance(result_dict['configuration'], dict)
    for field in dataclasses.fields(Result):
        assert field.name in result_dict

    assert result == Result.from_dict(result_dict)
    assert result.configuration == Configuration.from_dict(result_dict['configuration'])


def test_save_load_result_json(result, tmp_path):
    result_file = tmp_path / 'result.json'

    assert not result_file.exists()
    result.save_json(result_file)
    assert result_file.exists()

    assert result == Result.load_json(result_file)
