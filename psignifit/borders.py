# -*- coding: utf-8 -*-

from .utils import norminv


def set_borders(data, wmin=None, etype=None, srange=None, alpha=None):
    """
    Set borders of the grid for the parameter optimization

    Returns a dictionary {'paramater_name': (left_border, right_border)},
    where parameters and borders are set like in the following:

    'threshold': within the level spread +/- 50%
        'width': between half the minimal distance of two stimulus levels and
                 3 times the level spread (rescaled by the scale factor alpha)
       'lambda': the lapse rate is in (0, 0.5)
        'gamma': the guess rate is (0, 0.5) for 'yes/no' experiments,
                                   (1/n, 1/n) for nAFC experiments,
                                   None for 'equal asymptote experiments
          'eta': the overdispersion paramaters is between in (0, 1-1e-10)
    """

    # Threshold is assumed to be within the data spread +/- 50%
    spread = srange[1] - srange[0]
    threshold = (srange[0] - spread / 2, srange[1] + spread / 2)

    # The width is assumed to be between half the minimal distance of
    # two points and 3 times the range of data
    spread /= (norminv(.95) - norminv(.05)) / (norminv(1 - alpha) -
                                               norminv(alpha))
    width = (wmin, 3 * spread)

    # The lapse rate lambda
    lambda_ = (0., 0.5)

    # The guess rate gamma
    if etype == 'yes/no':
        gamma = (0., 0.5)
    elif etype == 'equal asymptote':
        gamma = None
    else:
        # this is a nAFC experiment
        n = int(etype[0])
        gamma = (1. / n, 1. / n)

    # The overdispersion parameter eta
    eta = (0., 1 - 1e-10)

    return {
        'threshold': threshold,
        'width': width,
        'lambda': lambda_,
        'gamma': gamma,
        'eta': eta
    }
