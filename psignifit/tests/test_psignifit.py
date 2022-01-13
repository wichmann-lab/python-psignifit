import numpy as np

from psignifit import psignifit
from .data import DATA


def test_psignifit_runs():
    # this should not error
    psignifit(DATA, sigmoid='norm', experiment_type='2AFC')


def test_psignifit_fixed_fit():
    parm_heiko = {'threshold': 0.0046448472488663396,
                  'width': 0.004683837353547434,
                  'lambda': 1.0676339572811912e-07,
                  'gamma': 0.5,
                  'eta': 0.00011599137494786461}

    results = psignifit(DATA, sigmoid='norm', experiment_type='2AFC')
    parm = results.parameter_estimate

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_heiko[p], rtol=1e-4, atol=1e-4)
