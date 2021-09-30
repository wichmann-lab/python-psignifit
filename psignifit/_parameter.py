# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Dict

import numpy as np
import scipy.stats

from ._typing import ExperimentType, ParameterBounds
from ._typing import ParameterBounds


def parameter_bounds(min_width: float, experiment_type: ExperimentType, stimulus_range: Tuple[float, float],
                     alpha: float, nafc_choices: Optional[int] = None) -> ParameterBounds:
    """ Specifies the minimum and maximum of the parameters for optimization.

    The parameter boundaries are calculated as follows:

    'threshold'
        within the level spread +/- 50%
    'width'
        between half the minimal distance of two stimulus levels and
        3 times the level spread (rescaled by the scale factor alpha)
    'lambda'
        the lapse rate is in (0, 0.5)
    'gamma'
        the guess rate is (0, 0.5) for 'yes/no' experiments,
        (1/n, 1/n) for nAFC experiments,
        None for 'equal asymptote experiments
    'eta'
        the overdispersion paramaters is between in (0, 1-1e-10)

    Args:
        min_width: lower bound of width parameter
        experiment_type: type of the experiment
        stimulus_range: range controlling the spread of all parameters
        alpha: Rescaling parameter of spread of width
        nafc_choices: number of forced choices of nAFC experiments

    Returns:
        A dictionary {'paramater_name': (left_bound, right_bound)}
    """
    # Threshold is assumed to be within the data spread +/- 50%
    spread = stimulus_range[1] - stimulus_range[0]
    threshold = (stimulus_range[0] - spread / 2, stimulus_range[1] + spread / 2)

    # The width is assumed to be between half the minimal distance of
    # two points and 3 times the range of data
    norminv = scipy.stats.norm(0, 1).ppf
    width_spread = spread / ((norminv(.95) - norminv(.05)) / (norminv(1 - alpha) - norminv(alpha)))

    experiment_type = ExperimentType(experiment_type)
    if experiment_type == ExperimentType.YES_NO:
        gamma = (0., 0.5)
    elif experiment_type == ExperimentType.EQ_ASYMPTOTE:
        gamma = None
    elif experiment_type == ExperimentType.N_AFC:
        gamma = (1. / nafc_choices, 1. / nafc_choices)

    return {
        'threshold': threshold,
        'width': (min_width, 3 * width_spread),
        'lambda': (0., 0.5),
        'gamma': gamma,
        'eta': (0., 1 - 1e-10)
    }


def masked_parameter_bounds(grid: Dict[str, Optional[np.ndarray]], mesh_mask: np.ndarray) -> ParameterBounds:
    """ Calculate the bounds as the first and last parameter value under the mask.

    Args:
        grid: Dict with arrays of possible parameter values.
        mesh_mask: Indicating the accepted(True) and ignored(False) parameter value combination.

    Returns:
        The new bounds per parameter as the first and last valid parameter value.
    """
    new_bounds = dict()
    mask_indices = mesh_mask.nonzero()
    for axis, (parameter_name, parameter_values) in enumerate(sorted(grid.items())):
        indices = mask_indices[axis]
        left, right = 0, len(parameter_values) - 1
        if len(indices) > 0:
            # get the first and last indices for this parameter's mask
            # and enlarged of one element in both directions
            left = max(left, indices.min() - 1)
            right = min(indices.max(), right)
        # update the bounds
        new_bounds[parameter_name] = (parameter_values[left], parameter_values[right])
    return new_bounds


def parameter_grid(bounds: ParameterBounds, steps: Dict[str, int]) -> Dict[str, Optional[np.ndarray]]:
    """Return uniformely spaced grid within given bounds.

    If the bound start and end values are close, a fixed value is assumed and the grid entry contains
    only the start.
    If the bound is None, the grid entry will be None.

    Args:
       bounds: a dictionary {parameter : (min_val, max_val)}
       steps: a dictionary {parameter : nsteps} where `nsteps` is the number of steps in the grid.

    Returns:
        grid:  a dictionary {parameter: (min_val, val1, val2, ..., max_val)}
    """
    grid = {}
    for param, bound in bounds.items():
        if bound is None:
            grid[param] = None
        elif np.isclose(bound[0], bound[1]):
            grid[param] = np.array([bound[0]])
        else:
            grid[param] = np.linspace(*bound, num=steps[param])
    return grid