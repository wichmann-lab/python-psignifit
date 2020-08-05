# -*- coding: utf-8 -*-
from typing import Tuple, Optional

from .utils import norminv
from .typing import ExperimentType
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
