import numbers

import numpy as np
import pytest

from psignifit import psignifit
from psignifit._utils import cast_np_scalar
from .fixtures import input_data
from psignifit.sigmoids import ALL_SIGMOID_CLASSES

@pytest.mark.parametrize('num', (1, 1., np.int64(1), np.float64(1.)))
def test_cast_np_scalar_numbers(num):
    cast = cast_np_scalar(num)
    assert not isinstance(cast, np.number)
    assert isinstance(cast, numbers.Number)


def test_cast_np_scalar_ndarray_noop():
    cast = cast_np_scalar(np.ones(1))
    assert isinstance(cast, np.ndarray)


def test_cast_np_scalar_ndarray_0d():
    cast = cast_np_scalar(np.array(1))
    assert isinstance(cast, numbers.Number)


@pytest.mark.parametrize('ci_method', ('percentiles', 'project'))
def test_py_not_np_scalar_ci(input_data, ci_method):
    result = psignifit(input_data[:3,:], experiment_type='yes/no', CI_method=ci_method)
    for parm, CI in result.confidence_intervals.items():
        for pc, (low, high) in CI.items():
            assert type(low) in (int, float), f'Confidence interval {pc} lower bound for {parm} is not a Python int/float'
            assert type(high) in (int, float), f'Confidence interval {pc} upper bound for  {parm} is not a Python int/float'


@pytest.mark.parametrize('fixed_parm', [(None, None),
                                        ('lambda', 3e-7),
                                        ('gamma', 0.1),
                                        ('eta', 1e-4),
                                        ('threshold', 0.0046),
                                        ('width', 0.0046)])
def test_py_not_np_scalar_param_estimate(input_data, fixed_parm):
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


def test_py_not_np_scalar_threshold(input_data):
    result = psignifit(input_data[:3,:], experiment_type='yes/no')
    value, ci = result.threshold(0.5)
    assert type(value) in (int, float)
    for pc, (low, high) in ci.items():
        assert type(low) in (int, float), f'Confidence interval {pc} lower bound is not a Python int/float'
        assert type(high) in (int, float), f'Confidence interval {pc} upper bound is not a Python int/float'


def test_py_not_np_scalar_slope(input_data):
    result = psignifit(input_data[:3,:], experiment_type='yes/no')
    value = result.slope(0.5)
    assert type(value) in (int, float)
    value = result.slope_at_proportion_correct(0.5, 0.3)
    assert type(value) in (int, float)


@pytest.mark.parametrize('with_eta', (True, False))
def test_py_not_np_scalar_proportion_correct(input_data, with_eta):
    result = psignifit(input_data[:3,:], experiment_type='yes/no')
    value = result.proportion_correct(input_data[0,0], with_eta=with_eta)
    assert type(value) in (int, float)


@pytest.mark.parametrize('negative', (True, False))
@pytest.mark.parametrize('sigmoid_class', ALL_SIGMOID_CLASSES)
@pytest.mark.parametrize('method', ('__call__', '_value', 'inverse', 'slope'))
def test_py_not_np_scalar_sigmoid_methods(negative, sigmoid_class, method):
    x, thr, wd = 0.3, 0.2, 0.1
    sigmoid = sigmoid_class()
    y = getattr(sigmoid, method)(x, thr, wd)
    assert type(y) in (int, float), f'method of Sigmoid does not return Python int/float'


@pytest.mark.parametrize('negative', (True, False))
@pytest.mark.parametrize('sigmoid_class', ALL_SIGMOID_CLASSES)
def test_py_not_np_scalar_sigmoid_standard_parameters(negative, sigmoid_class):
    x, thr, wd = 0.3, 0.2, 0.1
    sigmoid = sigmoid_class()
    st_params = sigmoid.standard_parameters(thr, wd)
    for parm in st_params:
        assert type(parm) in (int, float), f'Sigmoid.standard_parameters does not return Python int/float'
