"""
The Matlab version of psignifit-3 was extensively tested with
thousands of simulations for publishing the original paper
(Schütt, Harmeling, Macke, & Wichmann; 2016). Thus we consider
the matlab results as ground-truth and test for regression against them.
"""
from pathlib import Path
import json
import random

import numpy as np
import pytest

import psignifit
from psignifit._matlab import param_pydict2matlist

CASE_DIR = Path(__file__).parent / 'matlab_test_cases'


def yield_test_cases():
    return (file.name.rsplit('_', maxsplit=1)[0] for file in sorted(CASE_DIR.glob('test_*_data.csv')))


def yield_test_data(cases=yield_test_cases()):
    for name in cases:
        dataset = CASE_DIR / f'{name}_data.csv'
        options = CASE_DIR / f'{name}_opt.json'
        results = CASE_DIR / f'{name}_res.json'

        data = np.loadtxt(dataset, delimiter=',')
        with open(options, 'r') as f:
            options = json.load(f)
        with open(results, 'r') as f:
            results = json.load(f)
        yield data, options, results


def compare_with_matlab(data, options, results):
    config = psignifit.config_from_matlab(options, raise_matlab_only=False)
    pyresults = psignifit.psignifit(data, conf=config)
    # Testing for similarity:
    # allclose(actual, desired) = abs(actual - desired) <= atol + rtol * abs(desired)
    # ==> atol=0.001 assures equality of small differences
    #     rtol=1 assures the same order of magnitude
    #
    # These matlab results are currently not tested:
    # 'X1D', 'logPmax', 'integral', 'marginals', 'marginalsX', 'marginalsW', 'data',
    # 'devianceResiduals', 'deviance', 'Cov', 'Cor', 'timestamp', 'meanFit', 'MAP'

    ## POINT ESTIMATE should only be allowed 5% off
    np.testing.assert_allclose(param_pydict2matlist(pyresults.parameter_fit),
                               results['Fit'],
                               atol=0.001, rtol=0.05)

    # INTERVAL ESTIMATE should only be allowed 10% off
    mat_ci = results['conf_Intervals']
    if options['expType'] == 'equalAsymptote':
        # matlab returns gamma filled with Nones, but python expects gamma==lambda
        mat_ci[3] = mat_ci[2]
    # matlab dims are (n_param, 2, n_ci), but python expects (n_param, n_ci, 2),
    mat_ci = np.transpose(mat_ci, (0, 2, 1))
    py_ci = param_pydict2matlist(pyresults.confidence_intervals)
    np.testing.assert_allclose(py_ci, mat_ci, atol=0.001, rtol=.1)


@pytest.mark.slow  # runs for about 30 minutes
@pytest.mark.parametrize('data,options,results', yield_test_data(), ids=yield_test_cases())
def test_all_matlab_cases(data, options, results):
    compare_with_matlab(data, options, results)


test_case_subset = random.sample(list(yield_test_cases()), 5)
@pytest.mark.parametrize('data,options,results', yield_test_data(test_case_subset), ids=test_case_subset)
def test_few_matlab_cases(data, options, results):
    compare_with_matlab(data, options, results)
