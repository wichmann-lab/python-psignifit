import dataclasses
from unittest.mock import patch

import pytest

from psignifit.conf import Conf
from psignifit.utils import PsignifitException


def test_setting_valid_option():
    c = Conf(verbose=20)
    assert c.verbose == 20


def test_setting_invalid_option():
    with pytest.raises(dataclasses.FrozenInstanceError):
        c = Conf()
        c.verbose = 10
    # assignment shortcut
    with pytest.raises(TypeError):
        c = Conf(foobar=10)


@patch.object(Conf, 'check_bounds')
def test_check_option(mocked_check):
    __ = Conf(bounds=(10, 100))
    mocked_check.assert_called_with((10, 100))


def test_set_wrong_experiment_type():
    with pytest.raises(PsignifitException):
        Conf(experiment_type='foobar')


def test_set_bounds_with_nondict():
    with pytest.raises(PsignifitException):
        Conf(bounds=(1, 2, 3))


def test_set_bounds_with_wrong_key():
    with pytest.raises(PsignifitException):
        Conf(bounds={'foo': 'bar'})


def test_set_bounds_with_wrong_value1():
    with pytest.raises(PsignifitException):
        Conf(bounds={'threshold': 10})


def test_set_bounds_with_wrong_value2():
    with pytest.raises(PsignifitException):
        Conf(bounds={'threshold': (1, 2, 3)})


def test_set_wrong_sigmoid():
    with pytest.raises(PsignifitException):
        Conf(sigmoid='foobaro')


def test_set_stimulus_range_wrong_type():
    with pytest.raises(PsignifitException):
        Conf(stimulus_range=10)


def test_set_stimulus_range_wrong_length():
    with pytest.raises(PsignifitException):
        Conf(stimulus_range=(1, 2, 3))


def test_set_width_alpha_wrong_type():
    with pytest.raises(PsignifitException):
        Conf(width_alpha=(1, 2, 3))


def test_set_width_alpha_wrong_range():
    with pytest.raises(PsignifitException):
        Conf(width_alpha=1.2)
    with pytest.raises(PsignifitException):
        Conf(width_alpha=-1)


def test_set_width_min_wrong_type():
    with pytest.raises(PsignifitException):
        Conf(width_min=(1, 2, 3))
