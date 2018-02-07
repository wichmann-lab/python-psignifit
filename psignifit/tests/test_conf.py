import pytest

from psignifit import conf

def test_setting_valid_option():
    c = conf.Conf()
    c.nblocks = 10
    assert c.nblocks == 10
    # assignment shortcut
    c = conf.Conf(nblocks=20)
    assert c.nblocks == 20

def test_setting_invalid_option():
    with pytest.raises(conf.PsignifitConfException):
        c = conf.Conf()
        c.foobar = 10
    # assignment shortcut
    with pytest.raises(conf.PsignifitConfException):
        c = conf.Conf(foobar=10)

def test_setting_nonkw_argument():
    with pytest.raises(TypeError):
        conf.Conf(10)

def test_check_option():
    C = conf.Conf
    # add a fake new option to the conf
    C.valid_opts = list(C.valid_opts)+['foobar']
    # add a check for the fake option
    def check_foobar(self, value):
        if value > 10:
            raise conf.PsignifitConfException
    C.check_foobar = check_foobar
    # instantiate the new conf
    c = C()
    # try set the value in the valid range
    c.foobar = 10
    assert c.foobar == 10
    # try going out of range and catch the resulting exception
    with pytest.raises(conf.PsignifitConfException):
        c.foobar = 11

def test_repr():
    C = conf.Conf
    # define two valid options
    C.valid_opts = ('foobar', 'fiibur')
    # set only one option, so we can test the None case too
    c = C(foobar=10,)
    assert str(c) == 'fiibur: None\nfoobar: 10'
