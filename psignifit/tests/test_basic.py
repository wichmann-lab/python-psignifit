from math import isclose

import matplotlib.pyplot as plt
import numpy as np
import pytest

from psignifit import psignifit, psigniplot


def get_data():
    return np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                     [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                     [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                     [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                     [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                     [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                     [0.0100, 90.0000, 90.0000]])


def get_std_options():
    options = dict()
    options['sigmoid'] = 'norm'  # choose a cumulative Gauss as the sigmoid
    options['experiment_type'] = '2AFC'
    options['fixed_parameters'] = {'lambda': 0.01,
                                   'gamma': 0.5}
    return options


def test_fit_basic():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, **options)
    param = res.parameter_estimate
    assert isclose(param['threshold'], 0.0046, abs_tol=0.0001)
    assert isclose(param['width'], 0.0045, abs_tol=0.0001)
    assert isclose(param['lambda'], 0.01, abs_tol=0.0001)
    assert isclose(param['gamma'], 0.5, abs_tol=0.0001)
    assert isclose(param['eta'], 5.8e-06, abs_tol=0.0001)


def test_plot_psych():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, **options)
    plt.figure()
    psigniplot.plot_psychometric_function(res)
    plt.close('all')


def test_plot_marginal():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, **options)
    plt.figure()
    psigniplot.plot_marginal(res, 'threshold')
    plt.close('all')


def test_plot2D():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, return_posterior=True, **options)
    plt.figure()
    psigniplot.plot_2D_margin(res, 'threshold', 'width')
    plt.close('all')

    with pytest.raises(ValueError):
        res.posterior_mass = None
        psigniplot.plot_2D_margin(res, 'threshold', 'width')


def test_bias_analysis():
    data = get_data()
    plt.figure()
    other_data = np.array(data)
    other_data[:, 1] += 2
    other_data[:, 2] += 5
    psigniplot.plot_bias_analysis(data, other_data)
    plt.close('all')


def test_fixedPars():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, **options)
    estim_param = res.parameter_estimate
    fixed_param = res.configuration.fixed_parameters
    all_param_values = res.parameter_values

    # Check that fit and bounds are actually set to the fixed parametervalues
    for name, fixed_value in fixed_param.items():
        assert isclose(estim_param[name], fixed_value)
        assert len(all_param_values[name]) == 1
        assert isclose(all_param_values[name][0], fixed_value)
