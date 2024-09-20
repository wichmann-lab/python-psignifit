# -*- coding: utf-8 -*-
"""
Utils class capsulating all custom made probabilistic functions
"""
from typing import Optional

import numpy as np


# our own Exception class
class PsignifitException(Exception):
    pass


# create a decorator from the numpy errstate contextmanager, used to handle
# floating point errors. In our case divide-by-zero errors are usually harmless,
# because we are working in log space and log(0)==-inf is a valid result
class fp_error_handler(np.errstate):
    pass


def check_data(data: np.ndarray) -> np.ndarray:
    """ Check data format, type and range.

    Args:
        data: The data matrix with columns levels, number of correct and number of trials
    Returns:
        data as float numpy array
    Raises:
        PsignifitException: if checks fail.
    """
    data = np.asarray(data, dtype=float)
    if len(data.shape) != 2 and data.shape[1] != 3:
        raise PsignifitException("Expects data to be two dimensional with three columns, got {data.shape = }")
    levels, ncorrect, ntrials = data[:, 0], data[:, 1], data[:, 2]

    # levels should show some variance
    if levels.max() == levels.min():
        raise PsignifitException('Your stimulus levels are all identical.'
                                 ' They can not be fitted by a sigmoid!')
    # ncorrect and ntrials should be integers
    if not np.allclose(ncorrect, ncorrect.astype(int)):
        raise PsignifitException(
            'The number correct column contains non integer'
            ' numbers!')
    if not np.allclose(ntrials, ntrials.astype(int)):
        raise PsignifitException('The number of trials column contains non'
                                 ' integer numbers!')

    return data
