import pytest
import numpy as np
from numpy.testing import assert_almost_equal

import psignifit._parameter
from psignifit._parameter import parameter_bounds, masked_parameter_bounds
from psignifit._typing import ExperimentType


def test_parameter_bounds():
    bounds = parameter_bounds(min_width=0.1, experiment_type=ExperimentType.EQ_ASYMPTOTE.value, stimulus_range=(0, 1), alpha=0.01)
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

    parameter_bounds(min_width=0.1, experiment_type=ExperimentType.YES_NO.value, stimulus_range=(0, 1), alpha=0.01)
    parameter_bounds(min_width=0.1, experiment_type=ExperimentType.N_AFC.value, stimulus_range=(0, 1), alpha=0.01, nafc_choices=2)
    with pytest.raises(ValueError):
        parameter_bounds(min_width=0.1, experiment_type='unknown experiment', stimulus_range=(0, 1), alpha=0.01)


def test_mask_bounds():
    grid = {'A': np.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 0.1]),
            'B': np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.4, 0.0])}
    A, B = np.meshgrid(grid['A'], grid['B'], sparse=True)
    mask = (A + B) / 2 > 0.2
    # 1. (min, max) nonzero indices: [(4, 5), (2, 6)]
    # 2. increase / decrease by 1 and clip at border: [(3, 6), (1, 6)]
    # 3. return corresponding grid values:
    assert {'A': (0.2, 0.1), 'B': (0.05, 0.0)} == masked_parameter_bounds(grid, mask)


def test_parameter_grid():
    bounds = {
        'none': None,
        'fixed': (0.5, 0.5),
        'normal': (0, 1),
    }
    steps = {
        'none': 3,
        'normal': 15,
    }

    grid = psignifit._parameter.parameter_grid(bounds, steps)
    assert grid['none'] is None
    assert grid['normal'].shape == (15,)
    np.testing.assert_equal(grid['fixed'], [0.5])