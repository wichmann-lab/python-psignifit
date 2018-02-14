# -*- coding: utf-8 -*-
import numpy as np

from .getWeights import getWeights
from .likelihood import likelihood
from .marginalize import marginalize
from .utils import norminv


def set_borders(data, wmin=None, etype=None, srange=None,
                      alpha=None):
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
    threshold = (srange[0]-spread/2, srange[1]+spread/2)

    # The width is assumed to be between half the minimal distance of
    # two points and 3 times the range of data
    spread /= (norminv(.95)-norminv(.05))/(norminv(1-alpha)-norminv(alpha))
    width  = (wmin, 3*spread)

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
        gamma = (1./n, 1./n)

    # The overdispersion parameter eta
    eta = (0., 1-1e-10)

    return {'threshold': threshold, 'width': width, 'lambda': lambda_,
            'gamma': gamma, 'eta' : eta}

def moveBorders(data,options):
    """
    move parameter-boundaries to save computing power
    function borders=moveBorders(data, options)
    this function evaluates the likelihood on a much sparser, equally spaced
    grid definded by mbStepN and moves the borders in so that that
    marginals below tol are taken away from the borders.

    this is meant to save computing power by not evaluating the likelihood in
    areas where it is practically 0 everywhere.
    """
    borders = []

    tol = options['maxBorderValue']
    d = options['borders'].shape[0]

    MBresult = {'X1D':[]}

    ''' move borders inwards '''
    for idx in range(0,d):
        if (len(options['mbStepN']) >= idx and options['mbStepN'][idx] >= 2
            and options['borders'][idx,0] != options['borders'][idx,1]) :
            MBresult['X1D'].append(np.linspace(options['borders'][idx,0], options['borders'][idx,1], options['mbStepN'][idx]))
        else:
            if (options['borders'][idx,0] != options['borders'][idx,1] and options['expType'] != 'equalAsymptote'):
                warnings.warn('MoveBorders: You set only one evaluation for moving the borders!')

            MBresult['X1D'].append( np.array([0.5*np.sum(options['borders'][idx])]))


    MBresult['weight'] = getWeights(MBresult['X1D'])
    #kwargs = {'alpha': None, 'beta':None , 'lambda': None,'gamma':None , 'varscale':None }
    #fill_kwargs(kwargs,MBresult['X1D'])
    MBresult['Posterior'] = likelihood(data, options, MBresult['X1D'])[0]
    integral = sum(np.reshape(MBresult['Posterior'], -1) * np.reshape(MBresult['weight'], -1))
    MBresult['Posterior'] /= integral

    borders = np.zeros([d,2])

    for idx in range(0,d):
        (L1D,x,w) = marginalize(MBresult, np.array([idx]))
        x1 = x[np.max([np.where(L1D*w >= tol)[0][0] - 1, 0])]
        x2 = x[np.min([np.where(L1D*w >= tol)[0][-1]+1, len(x)-1])]

        borders[idx,:] = [x1,x2]

    return borders

