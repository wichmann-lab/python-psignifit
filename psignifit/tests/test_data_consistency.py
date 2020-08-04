import numpy as np
import pytest

from psignifit.conf import Conf
from psignifit.psignifit import psignifit, PsignifitException


# with pytest.raises(conf.Psignif
def test_novariance():
    data = np.random.random((10, 3)) * 10
    data[:, 0] = 1.
    with pytest.raises(PsignifitException):
        psignifit(data, Conf())


def test_nonint():
    data = np.random.random((10, 3))
    # repair ncorrect:
    data[:, 1] = np.round(data[:, 1])
    with pytest.raises(PsignifitException):
        psignifit(data, Conf())

    data = np.random.random((10, 3))
    # repair ncorrect:
    data[:, 2] = np.round(data[:, 2])
    with pytest.raises(PsignifitException):
        psignifit(data, Conf())
