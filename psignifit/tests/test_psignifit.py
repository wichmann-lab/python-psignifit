import numpy as np

from psignifit import psignifit

DATA =  np.array([[0.0010, 45.0000, 90.0000],
                  [0.0015, 50.0000, 90.0000],
                  [0.0020, 44.0000, 90.0000],
                  [0.0025, 44.0000, 90.0000],
                  [0.0030, 52.0000, 90.0000],
                  [0.0035, 53.0000, 90.0000],
                  [0.0040, 62.0000, 90.0000],
                  [0.0045, 64.0000, 90.0000],
                  [0.0050, 76.0000, 90.0000],
                  [0.0060, 79.0000, 90.0000],
                  [0.0070, 88.0000, 90.0000],
                  [0.0080, 90.0000, 90.0000],
                  [0.0100, 90.0000, 90.0000]])


def test_psignifit_runs():
    # this should not error
    results = psignifit(DATA, sigmoid='norm', experiment_type='2AFC')

def test_psignifit_fixed_fit():
    parm_heiko = {'threshold': 0.0046448472488663396,
                  'width': 0.004683837353547434,
                  'lambda': 1.0676339572811912e-07,
                  'gamma': 0.5,
                  'eta': 0.00011599137494786461}

    results = psignifit(DATA, sigmoid='norm', experiment_type='2AFC')
    parm = results['sigmoid_parameters']

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_heiko[p], rtol=1e-4, atol=1e-4)

