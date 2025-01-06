import numpy as np
import pytest

from psignifit import psignifit
from .fixtures import input_data

@pytest.mark.parametrize('ci_method', ('percentiles', 'project'))
def test_floats_not_0D_ndarray_ci(input_data, ci_method):
    result = psignifit(input_data[:3,:], experiment_type='yes/no', CI_method=ci_method)
    for parm, CI in result.confidence_intervals.items():
        for pc, (low, high) in CI.items():
            assert type(low) in (int, float), f'Confidence interval {pc} lower bound for {parm} is not a Python int/float'
            assert type(high) in (int, float), f'Confidence interval {pc} upper bound for  {parm} is not a Python int/float'

@pytest.mark.parametrize('fixed_parm', [(None, None),
                                        ('lambda', 3e-7),
                                        ('gamma', 0.5),
                                        ('eta', 1e-4),
                                        ('threshold', 0.0046),
                                        ('width', 0.0046)])
def test_floats_not_0D_ndarray_param_estimate(input_data, fixed_parm):
    parm, value = fixed_parm
    if parm:
        fixed_parm = {parm: value}
    else:
        fixed_parm = {}
    result = psignifit(input_data[:3,:], experiment_type='yes/no', fixed_parameters=fixed_parm)
    for parm, value in result.parameter_estimate_mean.items():
        assert type(value) in (int, float), f'Mean parameter estimate {parm} is not a Python int/float'
    for parm, value in result.parameter_estimate_MAP.items():
        assert type(value) in (int, float), f'MAP parameter estimate {parm} is not a Python int/float'


def test_floats_not_0D_ndarray_get_threshold(input_data):
    result = psignifit(input_data[:3,:], experiment_type='yes/no')
    value, ci = result.threshold(0.5)
    assert type(value) in (int, float)
    for pc, (low, high) in ci.items():
        assert type(low) in (int, float), f'Confidence interval {pc} lower bound is not a Python int/float'
        assert type(high) in (int, float), f'Confidence interval {pc} upper bound is not a Python int/float'

