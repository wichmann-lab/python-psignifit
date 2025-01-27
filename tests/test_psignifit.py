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
    parm_matlab = {'threshold': 0.0046448472488663396,
                   'width': 0.004683837353547434,
                   'lambda': 1.0676339572811912e-07,
                   'gamma': 0.5,
                   'eta': 0.00011599137494786461}

    results = psignifit(input_data, sigmoid='norm', experiment_type='2AFC')
    parm = results.get_parameter_estimate(estimate_type='MAP')

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_matlab[p], rtol=1e-4, atol=1e-4)


def test_psignifit_matlab_test1():
    """
    Corresponds to "test1" in
    https://github.com/wichmann-lab/psignifit/tree/master/tests/test_cases
    This dataset comes from a simulation of a 2AFC experiment, Gaussian sigmoid.
    """

    data = [[0.25      ,  4.        , 10.        ],
            [0.41666667,  6.        , 10.        ],
            [0.58333333,  3.        , 10.        ],
            [0.75      ,  8.        , 10.        ],
            [0.91666667,  7.        , 10.        ],
            [1.08333333,  5.        , 10.        ],
            [1.25      ,  8.        , 10.        ],
            [1.41666667,  9.        , 10.        ],
            [1.58333333,  9.        , 10.        ],
            [1.75      , 10.        , 10.        ]]

    parm_matlab = {'threshold': 1.19233664e+00,
                  'width': 1.13590185e+00,
                  'lambda': 2.47397128e-14,
                  'gamma': 0.5,
                  'eta': 9.41489062e-14}

    results = psignifit(data, sigmoid='norm', experiment_type='2AFC')
    parm = results.get_parameter_estimate(estimate_type='MAP')

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_matlab[p], rtol=1e-2, atol=1e-3)


def test_psignifit_matlab_yesno_norm():
    """
    Corresponds to "test_YesNo_norm" in
    https://github.com/wichmann-lab/psignifit/tree/master/tests/test_cases
    """

    data = [[ 0.25      ,  0.        , 10.        ],
            [ 0.41666667,  2.        , 10.        ],
            [ 0.58333333,  2.        , 10.        ],
            [ 0.75      ,  3.        , 10.        ],
            [ 0.91666667,  7.        , 10.        ],
            [ 1.08333333,  7.        , 10.        ],
            [ 1.25      ,  8.        , 10.        ],
            [ 1.41666667,  8.        , 10.        ],
            [ 1.58333333,  9.        , 10.        ],
            [ 1.75      , 10.        , 10.        ]]

    parm_matlab = {'threshold': 0.897273717,
                  'width': 1.43410261,
                  'lambda': 4.51674584e-11,
                  'gamma': 7.81152463e-12,
                  'eta': 2.07505419e-11}

    results = psignifit(data, sigmoid='norm', experiment_type='yes/no')
    parm = results.get_parameter_estimate(estimate_type='MAP')

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_matlab[p], rtol=1e-2, atol=1e-3)


def test_psignifit_matlab_2afc_logistic():
    """
    Corresponds to "test_2AFC_logistic" in
    https://github.com/wichmann-lab/psignifit/tree/master/tests/test_cases
    """

    data = [[ 0.25      ,  5.        , 10.        ],
            [ 0.41666667,  5.        , 10.        ],
            [ 0.58333333,  5.        , 10.        ],
            [ 0.75      ,  6.        , 10.        ],
            [ 0.91666667,  5.        , 10.        ],
            [ 1.08333333,  7.        , 10.        ],
            [ 1.25      ,  9.        , 10.        ],
            [ 1.41666667,  9.        , 10.        ],
            [ 1.58333333,  8.        , 10.        ],
            [ 1.75      , 10.        , 10.        ]]

    parm_matlab = {'threshold': 1.18954146,
                  'width': 1.34590649,
                  'lambda': 9.16461320e-14,
                  'gamma': 0.5,
                  'eta': 8.26939572e-16}

    results = psignifit(data, sigmoid='logistic', experiment_type='2AFC')
    parm = results.get_parameter_estimate(estimate_type='MAP')

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_matlab[p], rtol=1e-2, atol=1e-3)


def test_psignifit_matlab_yesno_logistic():
    """
    Corresponds to "test_YesNo_logistic" in
    https://github.com/wichmann-lab/psignifit/tree/master/tests/test_cases
    """

    data = [[ 0.25      ,  1.        , 10.        ],
            [ 0.41666667,  1.        , 10.        ],
            [ 0.58333333,  4.        , 10.        ],
            [ 0.75      ,  6.        , 10.        ],
            [ 0.91666667,  5.        , 10.        ],
            [ 1.08333333,  6.        , 10.        ],
            [ 1.25      ,  7.        , 10.        ],
            [ 1.41666667,  9.        , 10.        ],
            [ 1.58333333,  7.        , 10.        ],
            [ 1.75      , 10.        , 10.        ]]

    parm_matlab = {'threshold': 0.879924438,
                  'width': 2.05631136,
                  'lambda': 2.18166108e-15,
                  'gamma': 1.05715731e-12,
                  'eta': 8.04608804e-07}

    results = psignifit(data, sigmoid='logistic', experiment_type='yes/no')
    parm = results.get_parameter_estimate(estimate_type='MAP')

    for p in parm:
        np.testing.assert_allclose(parm[p], parm_matlab[p], rtol=1e-2, atol=1e-3)
