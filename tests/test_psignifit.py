import numpy as np

from psignifit import psignifit
from psignifit._result import Result

from .fixtures import input_data


def test_psignifit_runs(input_data):
    result = psignifit(input_data, sigmoid='norm', experiment_type='2AFC')
    assert isinstance(result, Result)
    assert len(result.debug) == 0


def test_psignifit_debug_info(input_data):
    result = psignifit(input_data, sigmoid='norm', experiment_type='2AFC', debug=True)
    assert len(result.debug) > 0


def test_psignifit_fixed_fit(input_data):
    parm_heiko = {'threshold': 0.0046448472488663396,
                  'width': 0.004683837353547434,
                  'lambda': 1.0676339572811912e-07,
                  'gamma': 0.5,
                  'eta': 0.00011599137494786461}

    results = psignifit(input_data, sigmoid='norm', experiment_type='2AFC')
    parm = results.get_parameter_estimate(estimate_type='MAP')

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_heiko[p], rtol=1e-4, atol=1e-4)
