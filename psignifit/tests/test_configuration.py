import dataclasses
from unittest.mock import patch

import pytest

from psignifit._configuration import Configuration
from psignifit._utils import PsignifitException


def test_setting_valid_option():
    c = Configuration(verbose=20)
    assert c.verbose == 20


def test_setting_invalid_option():
    with pytest.raises(TypeError):
        Configuration(foobar=10)


def test_dict_conversion():
    config = Configuration()
    config_dict = config.as_dict()

    assert isinstance(config_dict, dict)
    for field in dataclasses.fields(Configuration):
        assert field.name in config_dict

    assert config == Configuration.from_dict(config_dict)


@patch.object(Configuration, 'check_bounds')
def test_check_option(mocked_check):
    Configuration(bounds=(10, 100))
    mocked_check.assert_called_with((10, 100))


def test_check_experiment_type():
    with pytest.raises(PsignifitException):
        Configuration(experiment_type='foobar')

    with pytest.raises(PsignifitException):
        Configuration(experiment_type='nAFC')
    assert Configuration(experiment_type='nAFC', experiment_choices=12) == Configuration(experiment_type='12AFC')

    steps_nafc = Configuration(experiment_type='2AFC').steps_moving_bounds
    steps_eqasymp = Configuration(experiment_type='equal asymptote').steps_moving_bounds
    steps_yesno = Configuration(experiment_type='yes/no').steps_moving_bounds
    assert steps_eqasymp == steps_nafc
    assert steps_nafc != steps_yesno


def test_set_bounds_with_nondict():
    with pytest.raises(PsignifitException):
        Configuration(bounds=(1, 2, 3))


def test_set_bounds_with_wrong_key():
    with pytest.raises(PsignifitException):
        Configuration(bounds={'foo': 'bar'})


def test_set_bounds_with_wrong_value1():
    with pytest.raises(PsignifitException):
        Configuration(bounds={'threshold': 10})


def test_set_bounds_with_wrong_value2():
    with pytest.raises(PsignifitException):
        Configuration(bounds={'threshold': (1, 2, 3)})


def test_set_wrong_sigmoid():
    with pytest.raises(PsignifitException):
        Configuration(sigmoid='foobaro')


def test_set_stimulus_range_wrong_type():
    with pytest.raises(PsignifitException):
        Configuration(stimulus_range=10)


def test_set_stimulus_range_wrong_length():
    with pytest.raises(PsignifitException):
        Configuration(stimulus_range=(1, 2, 3))


def test_set_width_alpha_wrong_type():
    with pytest.raises(PsignifitException):
        Configuration(width_alpha=(1, 2, 3))


def test_set_width_alpha_wrong_range():
    with pytest.raises(PsignifitException):
        Configuration(width_alpha=1.2)
    with pytest.raises(PsignifitException):
        Configuration(width_alpha=-1)


def test_set_width_min_wrong_type():
    with pytest.raises(PsignifitException):
        Configuration(width_min=(1, 2, 3))


def test_warning_for_2afc_and_wrong_gamma():
    sigmoid = "norm"
    stim_range = [0.001, 0.2]
    lambda_ = 0.0232
    gamma = 0.1

    options = {}
    options['sigmoid'] = sigmoid  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': lambda_,
                                   'gamma': gamma}
    options["stimulus_range"] = stim_range

    with pytest.warns(UserWarning, match='gamma was fixed'):
        Configuration(**options)
