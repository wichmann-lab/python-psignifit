from math import isclose

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.skip("Skip basic tests until refactoring is complete")

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
    options['sigmoidName'] = 'norm'  # choose a cumulative Gauss as the sigmoid
    options.experiment_type = '2AFC'
    options['fixedPars'] = np.nan * np.ones(5)
    options['fixedPars'][2] = 0.01
    options['fixedPars'][3] = 0.5
    return options


def test_fit_basic():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, options)
    threshold, width, lapsus_rate, guess_rate, eta = res['Fit']
    assert isclose(threshold, 0.0045938670588426232)
    assert isclose(width, 0.0044734344997461074)
    assert isclose(lapsus_rate, 0.01)
    assert isclose(guess_rate, 0.5)
    assert isclose(eta, 5.8328681737030612e-06)


@pytest.mark.skip
def test_plotPsych():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, options)
    plt.figure()
    psigniplot.plotPsych(res, showImediate=False)
    plt.close('all')
    assert True


@pytest.mark.skip
def test_plotMarginal():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, options)
    plt.figure()
    psigniplot.plotMarginal(res, 0, showImediate=False)
    plt.close('all')
    assert True


@pytest.mark.skip
def test_plot2D():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, options)
    plt.figure()
    psigniplot.plot2D(res, 0, 1, showImediate=False)
    plt.close('all')
    assert True


def test_fixedPars():
    data = get_data()
    options = get_std_options()
    res = psignifit(data, options)
    # Check that fit and bounds are actually set to the fixed parametervalues
    assert np.all(res['Fit'][np.isfinite(options['fixedPars'])] ==
                  options['fixedPars'][np.isfinite(options['fixedPars'])])
    assert np.all(
        res['options']['bounds'][np.isfinite(options['fixedPars']), 0] ==
        options['fixedPars'][np.isfinite(options['fixedPars'])])
    assert np.all(
        res['options']['bounds'][np.isfinite(options['fixedPars']), 1] ==
        options['fixedPars'][np.isfinite(options['fixedPars'])])
