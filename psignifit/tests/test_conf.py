import pytest

from psignifit.conf import Conf
from psignifit.utils import PsignifitException

def test_setting_valid_option():
    c = Conf()
    c.verbose = 10
    assert c.verbose == 10
    # assignment shortcut
    c = Conf(verbose=20)
    assert c.verbose == 20

def test_setting_invalid_option():
    with pytest.raises(PsignifitException):
        c = Conf()
        c.foobar = 10
    # assignment shortcut
    with pytest.raises(PsignifitException):
        c = Conf(foobar=10)

def test_setting_nonkw_argument():
    with pytest.raises(TypeError):
        Conf(10)

def test_check_option():
    # create a Conf with a single valid option
    class B(Conf):
        _valid_opts = Conf._valid_opts+('foobar', )
    # add a check for the fake option
    def check_foobar(self, value):
        if value > 10:
            raise PsignifitException
    B.check_foobar = check_foobar
    # instantiate the new conf
    c = B()
    # try set the value in the valid range
    c.foobar = 10
    assert c.foobar == 10
    # try going out of range and catch the resulting exception
    with pytest.raises(PsignifitException):
        c.foobar = 11

def test_repr():
    # create a Conf with two additional valid options
    class B(Conf):
        _valid_opts = Conf._valid_opts+('foobar', 'fiibur')
    # set only one option, so we can test the None case too
    c = B(foobar=10)
    # manually restrict the list of valid options
    c._valid_opts = ('foobar', 'fiibur')
    assert str(c) == 'fiibur: None\nfoobar: 10'

def test_private_attr():
    c = Conf()
    c._foobar = 10
    assert c._foobar == 10
    # it shouldn't appear in the str representation
    assert str(c).find('_foobar: 10') == -1

def test_set_wrong_experiment_type():
    with pytest.raises(PsignifitException):
        Conf(experiment_type = 'foobar')

def test_set_borders_with_nondict():
    with pytest.raises(PsignifitException):
        Conf(borders = (1, 2, 3))

def test_set_borders_with_wrong_key():
    with pytest.raises(PsignifitException):
        Conf(borders={'foo': 'bar'})

def test_set_borders_with_wrong_value1():
    with pytest.raises(PsignifitException):
        Conf(borders={'threshold': 10})

def test_set_borders_with_wrong_value2():
    with pytest.raises(PsignifitException):
        Conf(borders={'threshold': (1,2,3)})

def test_set_wrong_sigmoid():
    with pytest.raises(PsignifitException):
        Conf(sigmoid='foobaro')

def test_set_stimulus_range_wrong_type():
    with pytest.raises(PsignifitException):
        Conf(stimulus_range=10)

def test_set_stimulus_range_wrong_length():
    with pytest.raises(PsignifitException):
        Conf(stimulus_range=(1,2,3))

def test_set_width_alpha_wrong_type():
    with pytest.raises(PsignifitException):
        Conf(width_alpha=(1,2,3))

def test_set_width_alpha_wrong_range():
    with pytest.raises(PsignifitException):
        Conf(width_alpha=1.2)
    with pytest.raises(PsignifitException):
        Conf(width_alpha=-1)

def test_set_width_min_wrong_type():
    with pytest.raises(PsignifitException):
        Conf(width_min=(1,2,3))

