from math import isclose

import matplotlib.pyplot as plt
import numpy as np
import pytest

from psignifit import psignifit, psigniplot
from .fixtures import input_data


def get_std_options():
    options = dict()
    options['sigmoid'] = 'norm'  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': 0.01}
    return options


def test_fit_basic(input_data):
    options = get_std_options()
    res = psignifit(input_data, **options)
    param = res.parameter_estimate_MAP
    assert isclose(param['threshold'], 0.0046, abs_tol=0.0001)
    assert isclose(param['width'], 0.0045, abs_tol=0.0001)
    assert isclose(param['lambda'], 0.01, abs_tol=0.0001)
    assert isclose(param['gamma'], 0.5, abs_tol=0.0001)
    assert isclose(param['eta'], 5.8e-06, abs_tol=0.0001)


def test_plot_psych(input_data):
    options = get_std_options()
    res = psignifit(input_data, **options)
    plt.figure()
    psigniplot.plot_psychometric_function(res)
    plt.close('all')


def test_plot_marginal(input_data):
    options = get_std_options()
    res = psignifit(input_data, debug=True, **options)
    plt.figure()
    psigniplot.plot_marginal(res, 'threshold')
    plt.close('all')


def test_plot2D(input_data):
    options = get_std_options()
    res = psignifit(input_data, debug=False, **options)

    with pytest.raises(ValueError):
        plt.figure()
        psigniplot.plot_2D_margin(res, 'threshold', 'width')
        plt.close('all')

    res = psignifit(input_data, debug=True, **options)
    plt.figure()
    psigniplot.plot_2D_margin(res, 'threshold', 'width')
    plt.close('all')


def test_bias_analysis(input_data):
    plt.figure()
    other_data = np.array(input_data)
    other_data[:, 1] += 2
    other_data[:, 2] += 5
    psigniplot.plot_bias_analysis(input_data, other_data)
    plt.close('all')


def test_fixed_parameters(input_data):
    options = get_std_options()
    res = psignifit(input_data, **options)
    estim_param = res.parameter_estimate_MAP
    fixed_param = res.configuration.fixed_parameters
    all_param_values = res.parameter_values

    # Check that fit and bounds are actually set to the fixed parameter values
    for name, fixed_value in fixed_param.items():
        assert isclose(estim_param[name], fixed_value)
        assert len(all_param_values[name]) == 1
        assert isclose(all_param_values[name][0], fixed_value)
