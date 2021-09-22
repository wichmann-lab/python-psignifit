import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from psignifit.bounds import parameter_bounds, mask_bounds
from psignifit.typing import ExperimentType


def test_parameter_bounds():
    bounds = parameter_bounds(wmin=0.1, etype=ExperimentType.EQ_ASYMPTOTE.value, srange=(0, 1), alpha=0.01)
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

    parameter_bounds(wmin=0.1, etype=ExperimentType.YES_NO.value, srange=(0, 1), alpha=0.01)
    parameter_bounds(wmin=0.1, etype=ExperimentType.N_AFC.value, srange=(0, 1), alpha=0.01, echoices=2)
    with pytest.raises(ValueError):
        parameter_bounds(wmin=0.1, etype='unknown experiment', srange=(0, 1), alpha=0.01)


def test_mask_bounds():
    grid = {'A': np.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 0.1]),
            'B': np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.4, 0.0])}
    A, B = np.meshgrid(grid['A'], grid['B'], sparse=True)
    mask = (A + B) / 2 > 0.2
    # 1. (min, max) nonzero indices: [(4, 5), (2, 6)]
    # 2. increase / decrease by 1 and clip at border: [(3, 6), (1, 6)]
    # 3. return corresponding grid values:
    assert {'A': (0.2, 0.1), 'B': (0.05, 0.0)} == mask_bounds(grid, mask)
