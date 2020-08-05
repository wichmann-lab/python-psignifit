import pytest
from numpy.testing import assert_almost_equal

from psignifit.bounds import parameter_bounds
from psignifit.typing import ExperimentType


def test_parameter_bounds():
    bounds = parameter_bounds(wmin=0.1, etype=ExperimentType.EQ_ASYMPTOTE, srange=(0, 1), alpha=0.01)
    true_bounds = {'threshold': (-0.5, 1.5),
                   'width': (0.1, 4.242957250279646),
                   'lambda': (0.0, 0.5),
                   'gamma': None,
                   'eta': (0.0, 0.9999999999)}

    for key, true_value in true_bounds.items():
        actual_value = bounds[key]
        if actual_value is None:
            assert true_value is None
        else:
            assert_almost_equal(actual_value, true_value)

    parameter_bounds(wmin=0.1, etype=ExperimentType.YES_NO, srange=(0, 1), alpha=0.01)
    parameter_bounds(wmin=0.1, etype=ExperimentType.N_AFC, srange=(0, 1), alpha=0.01, echoices=2)
    with pytest.raises(ValueError):
        parameter_bounds(wmin=0.1, etype='unknown experiment', srange=(0, 1), alpha=0.01)
