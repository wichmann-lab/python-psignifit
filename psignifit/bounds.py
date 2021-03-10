# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Dict

import numpy as np

from .utils import norminv
from .typing import ExperimentType, ParameterBounds
from .typing import ParameterBounds


def parameter_bounds(wmin: float, etype: ExperimentType, srange: Tuple[float, float],
                     alpha: float, echoices: Optional[int] = None) -> ParameterBounds:
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
        wmin: lower bound of width parameter
        etype: type of the experiment
        srange: range controlling the spread of all parameters
        alpha: Rescaling parameter of spread of width
        echoices: number of forced choices of nAFC experiments

    Returns:
        A dictionary {'paramater_name': (left_bound, right_bound)}
    """
    # Threshold is assumed to be within the data spread +/- 50%
    spread = srange[1] - srange[0]
    threshold = (srange[0] - spread / 2, srange[1] + spread / 2)

    # The width is assumed to be between half the minimal distance of
    # two points and 3 times the range of data
    width_spread = spread / ((norminv(.95) - norminv(.05)) / (norminv(1 - alpha) - norminv(alpha)))

    etype = ExperimentType(etype)
    if etype == ExperimentType.YES_NO:
        gamma = (0., 0.5)
    elif etype == ExperimentType.EQ_ASYMPTOTE:
        gamma = None
    elif etype == ExperimentType.N_AFC:
        gamma = (1. / echoices, 1. / echoices)

    return {
        'threshold': threshold,
        'width': (wmin, 3 * width_spread),
        'lambda': (0., 0.5),
        'gamma': gamma,
        'eta': (0., 1 - 1e-10)
    }


def mask_bounds(grid: Dict[str, Optional[np.ndarray]], mesh_mask: np.ndarray) -> ParameterBounds:
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
        # get the first and last indices for this parameter's mask
        # and enlarged of one element in both directions
        left = max(0, mask_indices[axis].min() - 1)
        right = min(mask_indices[axis].max(), len(parameter_values) - 1)
        # update the bounds
        new_bounds[parameter_name] = (parameter_values[left], parameter_values[right])
    return new_bounds