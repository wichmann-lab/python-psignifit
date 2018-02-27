# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
from functools import partial

import numpy as np
import datetime as _dt
import warnings
from copy import deepcopy as _deepcopy
import scipy

from . import priors as _priors
from .conf import Conf
from . import sigmoids
from . import likelihood as _l
from .borders import set_borders
from .utils import (norminv, norminvg, t1icdf, pool_data,
                    PsignifitException, normalize, fp_error_handler)

from .gridSetting import gridSetting
from .getWeights import getWeights
from .getConfRegion import getConfRegion
from .getSeed import getSeed
from .marginalize import marginalize
from .getSigmoidHandle import getSigmoidHandle

from . import psigniplot as plot


@fp_error_handler(divide='ignore')
def psignifit(data, conf=None, **kwargs):
    """
    main function for fitting psychometric functions
    function result=psignifit(data,options)
    This function is the user interface for fitting psychometric functions to data.

    pass your data in the n x 3 matrix of the form:
    [x-value, number correct, number of trials]

    options should be a dictionary in which you set the options for your fit.
    You can find a full overview over the options in demo002

    The result of this function is a dictionary, which contains all information the
    program produced for your fit. You can pass this as whole to all further
    processing function provided with psignifit. Especially to the plot functions.
    You can find an explanation for all fields of the result in demo006

    To get an introduction to basic usage start with demo001
    """
    if conf is None:
        conf = Conf(**kwargs)
    elif len(kwargs) > 0:
        # user shouldn't specify a conf object *and* kwargs simultaneously
        raise PsignifitException(
        "Can't handle conf together with other keyword arguments!")

    verbose = conf.verbose

    data = np.asarray(data, dtype=float)

    # check that the data are plausible
    levels, ncorrect, ntrials = data[:,0], data[:,1], data[:,2]

    # levels should show some variance
    if levels.max() == levels.min():
        raise PsignifitException('Your stimulus levels are all identical.'
                                 ' They can not be fitted by a sigmoid!')

    # ncorrect and ntrials should be integers
    if not np.allclose(ncorrect, ncorrect.astype(int)):
        raise PsignifitException('The number correct column contains non integer'
                                 ' numbers!')

    if not np.allclose(ntrials, ntrials.astype(int)):
        raise PsignifitException('The number of trials column contains non'
                                 ' integer numbers!')

    # options

    #if options['expType'] in ['2AFC', '3AFC', '4AFC']:
    #    options['expN'] = int(float(options['expType'][0]))
    #    options['expType'] = 'nAFC'

    #if options['expType'] == 'nAFC' and not('expN' in options.keys()):
    #    raise ValueError('For nAFC experiments please also pass the number of alternatives (options.expN)')

    # log space sigmoids
    # we fit these functions with a log transformed physical axis
    # This is because it makes the paramterization easier and also the priors
    # fit our expectations better then.
    # The flag is needed for the setting of the parameter bounds in setBorders

    # options['sigmoidName'] in ['Weibull','logn','weibull']:
    #        options['logspace'] = 1
    #        assert min(data[:,0]) > 0, 'The sigmoid you specified is not defined for negative data points!'
    #else:
    #    options['logspace'] = 0

    # stimulus range settings
    stimulus_range = conf.stimulus_range
    if stimulus_range is None:
        # derive the stimulus range from the data
        stimulus_range = (levels.min(), levels.max())
    if conf._logspace:
        # change it to logspace if needed
        stimulus_range = (np.log(stimulus_range[0]), np.log(stimulus_range[1]))

    # width_min is the minimum difference between two different stimulus levels
    width_min = conf.width_min
    if width_min is None:
        # if it wasn't overriden by the user, derive it from the data
        if conf.stimulus_range is None:
            # if the stimulus range is spanned by the input data
            if conf._logspace:
                width_min = np.diff(np.unique(np.log(levels))).min()
            else:
                width_min = np.diff(np.unique(levels)).min()
        else:
            # if the stimulus range was set manually, we can not derive width_min
            # from the data, so we will instead use a very conservative estimate,
            # i.e. 100 ULP from the largest end of the stimulus range
            # for ULP, see https://en.wikipedia.org/wiki/Unit_in_the_last_place
            width_min = 100*np.spacing(stimulus_range[1])

    # get priors
    #if options['threshPC'] != .5 and not(hasattr(options, 'priors')):
    #    warnings.warn('psignifit:TresholdPCchanged\n'\
    #        'You changed the percent correct corresponding to the threshold\n')

    #if not('priors' in options.keys()):
    #    options['priors'] = _p.getStandardPriors(data, options)
    #else:
    #
    #    priors = _p.getStandardPriors(data, options)
    #
    #    for ipar in range(5):
    #        if not(hasattr(options['priors'][ipar], '__call__')):
    #            options['priors'][ipar] = priors[ipar]
    #
    #    p.checkPriors(data, options)
    width_alpha = conf.width_alpha
    if conf.priors is None:
        priors = {
        'threshold': partial(_priors.pthreshold, st_range=stimulus_range),
        'width': partial(_priors.pwidth, wmin=width_min,
                                        wmax=stimulus_range[1]-stimulus_range[0],
                                        alpha=width_alpha),
        'lambda': _priors.plambda,
        'gamma': _priors.pgamma,
        'eta' : partial(_priors.peta, k=conf.beta_prior),
        }
    else:
        # will take care of user-specified priors later!
        #XXX TODO!
        raise NotImplementedError

    # sigmoid
    sigmoid = getattr(sigmoids, conf.sigmoid)
    # fix thresh_PC and width_alpha parameters for the sigmoid
    sigmoid = partial(sigmoid, PC=conf.thresh_PC, alpha=conf.width_alpha)

    #warning if many blocks were measured
    if verbose and len(levels) >= 25 and not conf.stimulus_range:
        print(
f"""The data you supplied contained {len(levels)}>= 25 stimulus levels.
Did you sample adaptively?
If so please specify a range which contains the whole psychometric function in
conf.stimulus_range.
An appropriate prior prior will be then chosen. For now we use the standard
heuristic, assuming that the psychometric function is covered by the stimulus
levels,which is frequently invalid for adaptive procedures!""")

    if verbose and ntrials.max() <= 5 and not conf.stimulus_range:
        print(
"""All provided data blocks contain <= 5 trials.
Did you sample adaptively?
If so please specify a range which contains the whole psychometric function in
conf.stimulus_range.
An appropriate prior prior will be then chosen. For now we use the standard
heuristic, assuming that the psychometric function is covered by the stimulus
levels, which is frequently invalid for adaptive procedures!""")

    if verbose and ntrials.max() == 1 or len(levels) > conf.pool_max_blocks:
        print(
"""Pooling data, to avoid problems with n=1 blocks or to save time fitting
because you have more than 25 blocks.
You can force acceptance of your blocks by increasing conf.pool_max_blocks""")
        data = pool_data(data, xtol=conf.pool_xtol, max_gap=conf.pool_max_gap,
                               max_length=conf.pool_max_length)

    # borders of integration
    exp_type = conf.experiment_type
    borders = set_borders(data, wmin=width_min, etype=exp_type,
                          srange=stimulus_range, alpha=width_alpha)

    # override at user request
    if conf.borders is not None:
        borders.update(conf.borders)

    # XXX FIXME: take care of fixed parameters later!
    # border_idx = np.where(np.isnan(options['fixedPars']) == False);
    # print(border_idx)
    # if (border_idx[0].size > 0):
        # options['borders'][border_idx[0],0] = options['fixedPars'][border_idx[0]]
        # options['borders'][border_idx[0],1] = options['fixedPars'][border_idx[0]]

    # normalize priors to first choice of borders
    for parameter, prior in priors.items():
        priors[parameter] = normalize(prior, borders[parameter])


    # XXX FIXME: take care later of moving borders
    # if options['moveBorders']:
        # options['borders'] = _b.moveBorders(data, options)
    if conf.move_borders:
        borders = _b.move_borders(data, borders=borders,
                                  steps=conf.steps_moving_borders,
                                  tol=conf.max_border_value,
                                  )

    # core
    #result = psignifitCore(data,options)

    # create a linear grid
    grid = {}
    for param in borders:
        if borders[param] is None:
            grid[param] == None
        else:
            grid[param] = np.linspace(*borders[param], num=conf.grid_steps[param])


    results = _l.likelihood(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # XXX FIXME: take care of post-ptocessing later
    # ''' after processing '''
    # # check that the marginals go to nearly 0 at the borders of the grid
    # if options['verbose'] > -5:

        # if result['marginals'][0][0] * result['marginalsW'][0][0] > .001:
            # warnings.warn('psignifit:borderWarning\n'\
                # 'The marginal for the threshold is not near 0 at the lower border.\n'\
                # 'This indicates that smaller Thresholds would be possible.')
        # if result['marginals'][0][-1] * result['marginalsW'][0][-1] > .001:
            # warnings.warn('psignifit:borderWarning\n'\
                # 'The marginal for the threshold is not near 0 at the upper border.\n'\
                # 'This indicates that your data is not sufficient to exclude much higher thresholds.\n'\
                # 'Refer to the paper or the manual for more info on this topic.')
        # if result['marginals'][1][0] * result['marginalsW'][1][0] > .001:
            # warnings.warn('psignifit:borderWarning\n'\
                # 'The marginal for the width is not near 0 at the lower border.\n'\
                # 'This indicates that your data is not sufficient to exclude much lower widths.\n'\
                # 'Refer to the paper or the manual for more info on this topic.')
        # if result['marginals'][1][-1] * result['marginalsW'][1][-1] > .001:
            # warnings.warn('psignifit:borderWarning\n'\
                # 'The marginal for the width is not near 0 at the lower border.\n'\
                # 'This indicates that your data is not sufficient to exclude much higher widths.\n'\
                # 'Refer to the paper or the manual for more info on this topic.')

    # result['timestamp'] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # if options['instantPlot']:
        # plot.plotPsych(result)

    return results

def psignifitFast(data,options):
    """
    this uses changed settings for the fit to obtain a fast point estimate to
    your data.
    The mean estimate with these settings is very crude, the MAP estimate is
    better, but takes a bit of time for the optimization (~100 ms)
    """

    warnings.warn('You use the speed optimized version of this program. \n' \
    'This is NOT suitable for the final analysis, but meant for online analysis, adaptive methods etc. \n'  \
    'It has not been tested how good the estimates from this method are!')

    options['stepN']     = [20,20,10,10,1]
    options['mbStepN']  = [20,20,10,10,1]
    options['fixedPars'] = np.array([np.NaN,np.NaN,np.NaN,np.NaN,0.0])
    options['fastOptim'] = True

    res = psignifit(data,options)

    return res


def psignifitCore(data, options):
    """
    This is the Core processing of psignifit, call the frontend psignifit!
    function result=psignifitCore(data,options)
    Data nx3 matrix with values [x, percCorrect, NTrials]

    sigmoid should be a handle to a function, which accepts
    X,parameters as inputs and gives back a value in [0,1]. ideally
    parameters(1) should correspond to the threshold and parameters(2) to
    the width (distance containing 95% of the function.
    """

    d = len(options['borders'])
    result = {'X1D': [], 'marginals': [], 'marginalsX': [], 'marginalsW': []}

    '''Choose grid dynamically from data'''
    if options['dynamicGrid']:
        # get seed from linear regression with logit transform
        Seed = getSeed(data,options)

        # further optimize the logliklihood to obtain a good estimate of the MAP
        if options['expType'] == 'YesNo':
            calcSeed = lambda X: -_l.logLikelihood(data, options, X[0], X[1], X[2], X[3], X[4])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = Seed)
        elif options['expType'] == 'nAFC':
            calcSeed = lambda X: -_l.logLikelihood(data, options, X[0], X[1], X[2], 1/options['expN'], X[3])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = [Seed[0:2], Seed[4]])
            Seed = [Seed[0:2], 1/options['expN'], Seed[3]] #ToDo check whether row or colum vector
        result['X1D'] = gridSetting(data,options, Seed)


    else: # for types which do not need a MAP estimate
        if (options['gridSetType'] == 'priorlike' or options['gridSetType'] == 'STD'
            or options['gridSetType'] == 'exp' or options['gridSetType'] == '4power'):
                result['X1D'] = gridSetting(data,options)
        else: # Use a linear grid
            for idx in range(0,d):
                # If there is an actual Interval
                if options['borders'][idx, 0] < options['borders'][idx,1]:

                    result['X1D'].append(np.linspace(options['borders'][idx,0], options['borders'][idx,1],
                                    num=options['stepN'][idx]))
                # if parameter was fixed
                else:
                    result['X1D'].append(np.array([options['borders'][idx,0]]))

    '''Evaluate likelihood and form it into a posterior'''

    (result['Posterior'], result['logPmax']) = _l.likelihood(data, options, result['X1D'])
    result['weight'] = getWeights(result['X1D'])
    integral = np.sum(np.array(result['Posterior'][:])*np.array(result['weight'][:]))
    result['Posterior'] = result['Posterior']/integral
    result['integral'] = integral

    '''Compute marginal distributions'''

    for idx in range(0,d):
        m, mX, mW = marginalize(result, np.array([idx]))
        result['marginals'].append(m)
        result['marginalsX'].append(mX)
        result['marginalsW'].append(mW)

    result['marginals'] = np.squeeze(result['marginals'])
    result['marginalsX'] = np.squeeze(result['marginalsX'])
    result['marginalsW'] = np.squeeze(result['marginalsW'])

    '''Find point estimate'''
    if (options['estimateType'] in ['MAP','MLE']):
        # get MLE estimate

        #start at most likely grid point
        index = np.where(result['Posterior'] == np.max(result['Posterior'].ravel()))

        Fit = np.zeros([d,1])
        for idx in range(0,d):
            Fit[idx] = result['X1D'][idx][index[idx]]

        if options['expType'] == 'YesNo':
            fun = lambda X, f: -_l.logLikelihood(data, options, [X[0],X[1],X[2],X[3],X[4]])
            x0 = _deepcopy(Fit)
            a = None

        elif options['expType'] == 'nAFC':
            #def func(X,f):
            #    return -_l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            #fun = func
            fun = lambda X, f:  -_l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            x0 = _deepcopy(Fit[0:3]) # Fit[3]  is excluded
            x0 = np.append(x0,_deepcopy(Fit[4]))
            a = np.array([1/options['expN']])

        elif options['expType'] == 'equalAsymptote':
            fun = lambda X, f: -_l.logLikelihood(data,options,[X[0], X[1], X[2], f, X[3]])
            x0 = _deepcopy(Fit[0:3])
            x0 = np.append(x0,_deepcopy(Fit[4]))
            a =  np.array([np.nan])

        else:
            raise ValueError('unknown expType')

        if options['fastOptim']:
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), xtol=0, ftol = 0, maxiter = 100, maxfun=100)
            warnings.warn('changed options for optimization')
        else:
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), disp = False)

        if options['expType'] == 'YesNo':
            result['Fit'] = _deepcopy(Fit)
        elif options['expType'] == 'nAFC':
            fit = _deepcopy(Fit[0:3])
            fit = np.append(fit, np.array([1/options['expN']]))
            fit = np.append(fit, _deepcopy(Fit[3]))
            result['Fit'] = fit
        elif options['expType'] =='equalAsymptote':
            fit = _deepcopy(Fit[0:3])
            fit = np.append(fit, Fit[2])
            fit = np.append(fit, Fit[3])
            result['Fit'] = fit
        else:
            raise ValueError('unknown expType')

        par_idx = np.where(np.isnan(options['fixedPars']) == False)
        for idx in par_idx[0]:
            result['Fit'][idx] = options['fixedPars'][idx]

    elif options['estimateType'] == 'mean':
        # get mean estimate
        Fit = np.zeros([d,1])
        for idx in range[0:d]:
            Fit[idx] = np.sum(result['marginals'][idx]*result['marginalsW'][idx]*result['marginalsX'][idx])

        result['Fit'] = _deepcopy(Fit)
        Fit = np.empty(Fit.shape)
    '''Include input into result'''
    result['options'] = options # no copies here, because they are not changing
    result['data'] = data

    '''Compute confidence intervals'''
    if ~options['fastOptim']:
        result['conf_Intervals'] = getConfRegion(result)

    return result


def getSlope(result, stimLevel):
    """
    function slope = getSlope(result, stimLevel)
    This function finds the slope of the psychometric function at a given
    performance level in percent correct.

    result is a result dictionary from psignifit

    stimLevel is the stimuluslevel at where to evaluate the slope

    This function cannot provide credible intervals.
    """

    if 'Fit' in result.keys():
        theta0 = result['Fit']
    else:
        raise ValueError('Result needs to contain a resulting fit generated by psignifit')



    #calculate point estimate -> transform only the fit

    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC    = result['options']['threshPC']
    else:
        PC = 0.5

    if result['options']['sigmoidName'][0:3]=='neg':
        PC = 1-PC;


    sigName = result['options']['sigmoidName']

    if sigName in ['norm','gauss','neg_norm','neg_gauss']:
        C         = norminv(1-alpha) - norminv(alpha)
        normalizedStimLevel = (stimLevel-theta0[0])/theta0[1]*C
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]
    elif sigName in ['logistic','neg_logistic']:
        C = 2 * np.log(1/alpha - 1) / theta0[1]
        d = np.log(1/PC-1)
        slope = C*np.exp(-C*(stimLevel-theta0[0])+d)/(1+np.exp(-C*(stimLevel-theta0[0])+d))**2
    elif sigName in  ['gumbel','neg_gumbel']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevel = C/theta0[1]*(stimLevel-theta0[0])+np.log(-np.log(1-PC))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in ['rgumbel','neg_rgumbel']:                  #reversed gumbel
        C      = np.log(-np.log(1-alpha)) - np.log(-np.log(alpha))
        stimLevel = C/theta0[1]*(stimLevel-theta0[0])+np.log(-np.log(PC))
        slope = -C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in ['logn','neg_logn']:
        C      = norminv(1-alpha) - norminv(alpha)
        normalizedStimLevel = (np.log(stimLevel)-theta0[0])/theta0[1]
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]/stimLevel
    elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevelNormalized = C/theta0[1]*(np.log(stimLevel)-theta0[0])+np.log(-np.log(1-PC))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevelNormalized))*np.exp(stimLevelNormalized)
        slope = slope/stimLevel
    elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
        # student T distribution with 1 df --> heavy tail distribution
        C      = (my_t1icdf(1-alpha) - my_t1icdf(alpha))
        stimLevel = (stimLevel-theta0[0])/theta0[1]*C+my_t1icdf(PC)
        slope = C/theta0[1]*t.pdf(stimLevel,df=1)
    else:
        raise ValueError('unknown sigmoid function')


    slope   = (1-theta0[2]-theta0[3])*slope

    if result['options']['sigmoidName'][0:3]=='neg':
        slope = -slope
    return slope


def getSlopePC(result, pCorrect, unscaled = False):
    """
    function slope = getSlopePC(result, pCorrect, unscaled = False)
    This function finds the slope of the psychometric function at a given
    performance level in percent correct.

    result is a result dictionary from psignifit

    pCorrrect is the proportion correct at which to evaluate the slope

    This function cannot provide credible intervals.
    """
    if 'Fit' in result.keys():
        theta0 = result['Fit']
    else:
        raise ValueError('Result needs to contain a resulting fit generated by psignifit')



    #calculate point estimate -> transform only the fit

    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC    = result['options']['threshPC']
    else:
        PC = 0.5

    if unscaled:
        assert ((pCorrect > 0) & (pCorrect < 1)), 'pCorrect must be in ]0,1[ '
        pCorrectUnscaled = pCorrect
    else:
        assert ((pCorrect > theta0[3]) & (pCorrect < (1-theta0[2]))), 'pCorrect must lay btw {:.2f} and {:.2f}'.format(theta0[3], (1-theta0[2]))
        pCorrectUnscaled = (pCorrect-theta0[3])/(1-theta0[2] - theta0[3])



    ''' find the (normalized) stimulus level, where the given percent correct is
    reached and evaluate slope there'''
    sigName = result['options']['sigmoidName'].lower()

    if sigName[0:3]=='neg':
        pCorrectUnscaled = 1-pCorrectUnscaled
        PC = 1-PC


    if sigName in ['norm', 'gauss','neg_norm', 'neg_gauss']:
        C         = norminv(1-alpha) - norminv(alpha)
        normalizedStimLevel = norminv(pCorrectUnscaled)
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]
    elif sigName in ['logistic','neg_logistic']:
        stimLevel = theta0[0] - theta0[1]*np.log((1/pCorrectUnscaled-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
        C = 2 * np.log(1/alpha - 1) / theta0[1]
        d = np.log(1/PC-1)
        slope = C*np.exp(-C*(stimLevel-theta0[0])+d)/(1+np.exp(-C*(stimLevel-theta0[0])+d))**2
    elif sigName in  ['gumbel','neg_gumbel']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevel = np.log(-np.log(1-pCorrectUnscaled))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in ['rgumbel','neg_rgumbel']:                  #reversed gumbel
        C      = np.log(-np.log(1-alpha)) - np.log(-np.log(alpha))
        stimLevel = np.log(-np.log(pCorrectUnscaled))
        slope = -C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in['logn', 'neg_logn']:
        C      = norminv(1-alpha) - norminv(alpha)
        stimLevel = np.exp(norminvg(pCorrectUnscaled, theta0[0]-norminvg(PC,0,theta0[1]/C), theta0[1] / C))
        normalizedStimLevel = norminv(pCorrectUnscaled)
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]/stimLevel
    elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevel = np.exp(theta0[0]+theta0[1]/C*(np.log(-np.log(1-pCorrectUnscaled))-np.log(-np.log(1-PC))))
        stimLevelNormalized = np.log(-np.log(1-pCorrectUnscaled))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevelNormalized))*np.exp(stimLevelNormalized)
        slope = slope/stimLevel
    elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
        # student T distribution with 1 df --> heavy tail distribution
        C      = (t1icdf(1-alpha) - t1icdf(alpha))
        stimLevel = t1icdf(pCorrectUnscaled)
        slope = C/theta0[1]*t.pdf(stimLevel,df=1)
    else:
        raise ValueError('unknown sigmoid function')


    slope   = (1-theta0[2]-theta0[3])*slope

    if sigName[0:3]=='neg':
        slope = -slope
    return slope


def getThreshold(result,pCorrect, unscaled=False):
    """
    function [threshold,CI] = getThreshold(result, pCorrect,unscaled)
     this function finds a threshold value for a given fit for different
     percent correct cutoffs

     result is a result dict from psignifit

     pCorrect is the percent correct at the threshold you want to calculate

     unscaled is whether the percent correct you provide are for the unscaled
     sigmoid or for the one scaled by lambda and gamma. By default this
     function returns the one for the scaled sigmoid.

     The CIs you may obtain from this are calculated based on the confidence
     intervals only, e.g. with the shallowest and the steepest psychometric
     function and may thus broaden if you move away from the standard
     threshold of unscaled sigmoid = .5 /= options['threshPC']

     For the sigmoids in logspace this also returns values in the linear
     stimulus level domain.


     For a more accurate inference use the changed level of percent correct
     directly to define the threshold in the inference by setting
     options['threshPC'] and adjusting the priors.
    """

    if 'Fit' in result.keys():
        theta0 = _deepcopy(result['Fit'])
    else:
        raise ValueError('Result needs to contain a resulting fit generated by psignifit.')
    if 'conf_Intervals' in result.keys():
        CIs = _deepcopy(result['conf_Intervals'])
    else:
        raise ValueError('Result needs to contain confidence intervals for the fitted parameter.')


    if unscaled: # set asymptotes to 0 for everything.
        theta0[2]  = 0
        theta0[3]  = 0
        CIs[2:4,:] = 0


    assert ((np.array(pCorrect)>theta0[3]) & (np.array(pCorrect)<(1-theta0[2]))), 'The threshold percent correct is not reached by the sigmoid!'

    pCorrectUnscaled = (pCorrect-theta0[3])/(1-theta0[2]-theta0[3])
    alpha = result['options']['widthalpha']
    if  'threshPC' in result['options'].keys():
        PC    = result['options']['threshPC']
    else :
        PC = 0.5

    sigName = result['options']['sigmoidName'].lower()
    if sigName[0:3]=='neg':
        PC = 1-PC
        pCorrectUnscaled = 1-pCorrectUnscaled

    if sigName in ['norm', 'gauss','neg_norm', 'neg_gauss']:
        C         = norminv(1-alpha) - norminv(alpha)
        threshold = norminvg(pCorrectUnscaled, theta0[0]-norminvg(PC,0,theta0[1]/C), theta0[1] / C)
    elif sigName in ['logistic','neg_logistic']:
        threshold = theta0[0]-theta0[1]*(np.log(1/pCorrectUnscaled-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
    elif sigName in ['gumbel','neg_gumbel']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        threshold = theta0[0] + (np.log(-np.log(1-pCorrectUnscaled))-np.log(-np.log(1-PC)))*theta0[1]/C
    elif sigName in ['rgumbel','neg_rgumbel']:
        C      = np.log(-np.log(1-alpha)) - np.log(-np.log(alpha))
        threshold = theta0[0] + (log(-log(pCorrectUnscaled))-log(-log(PC)))*theta0[1]/C
    elif sigName in ['logn','neg_logn']:
        C      = norminv(1-alpha) - norminv(alpha)
        threshold = np.exp(norminvg(pCorrectUnscaled, theta0[0]-norminvg(PC,0,theta0[1]/C), theta0[1] / C))
    elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        threshold = np.exp(theta0[0]+theta0[1]/C*(np.log(-np.log(1-pCorrectUnscaled))-np.log(-np.log(1-PC))))
    elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
        C      = (t1icdf(1-alpha) - t1icdf(alpha))
        threshold = (t1icdf(pCorrectUnscaled)-t1icdf(PC))*theta0[1] / C + theta0[0]
    else:
        raise ValueError('unknown sigmoid function')

    """ calculate CI -> worst case in parameter confidence intervals """

    warnings.warn('The CIs computed by this method are only upper bounds. For more accurate inference change threshPC in the options.')
    CI = np.zeros([len(result['options']['confP']),2])
    for iConfP in range(0,len(result['options']['confP'])):

        if sigName[0:3]=='neg':
            if pCorrectUnscaled < PC:
                thetaMax = [CIs[0,1,iConfP],CIs[1,0,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMin = [CIs[0,0,iConfP],CIs[1,1,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
            else:
                thetaMin = [CIs[0,1,iConfP],CIs[1,1,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMax = [CIs[0,0,iConfP],CIs[1,0,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
        else:
            if pCorrectUnscaled > PC:
                thetaMin = [CIs[0,0,iConfP],CIs[1,0,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMax = [CIs[0,1,iConfP],CIs[1,1,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
            else:
                thetaMin = [CIs[0,0,iConfP],CIs[1,1,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMax = [CIs[0,1,iConfP],CIs[1,0,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
        pCorrMin = (pCorrect-thetaMin[3])/(1-thetaMin[2]-thetaMin[3])
        pCorrMax = (pCorrect-thetaMax[3])/(1-thetaMax[2]-thetaMax[3])
        if sigName in ['norm', 'gauss','neg_norm', 'neg_gauss']:
            CI[iConfP,0]     = norminvg(pCorrMin, thetaMin[0]-norminvg(PC,0,thetaMin[1]/C), thetaMin[1] / C)
            CI[iConfP,1]     = norminvg(pCorrMax, thetaMax[0]-norminvg(PC,0,thetaMax[1]/C), thetaMax[1] / C)
        elif sigName in ['logistic','neg_logistic']:
            CI[iConfP,0]     = thetaMin[0]-thetaMin[1]*(np.log(1/pCorrMin-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
            CI[iConfP,1]     = thetaMax[0]-thetaMax[1]*(np.log(1/pCorrMax-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
        elif sigName in ['gumbel','neg_gumbel']:
            CI[iConfP,0] = thetaMin[0] + (np.log(-np.log(1-pCorrMin))-np.log(-np.log(1-PC)))*thetaMin[1]/C
            CI[iConfP,1] = thetaMax[0] + (np.log(-np.log(1-pCorrMax))-np.log(-np.log(1-PC)))*thetaMax[1]/C
        elif sigName in ['rgumbel','neg_rgumbel']:
            CI[iConfP,0] = thetaMin[0] + (np.log(-np.log(pCorrMin))-np.log(-np.log(PC)))*thetaMin[1]/C
            CI[iConfP,1] = thetaMax[0] + (np.log(-np.log(pCorrMax))-np.log(-np.log(PC)))*thetaMax[1]/C
        elif sigName in ['logn','neg_logn']:
            CI[iConfP,0] = np.exp(norminvg(pCorrMin, thetaMin[0]-norminvg(PC,0,thetaMin[1]/C), thetaMin[1]/ C))
            CI[iConfP,1] = np.exp(norminvg(pCorrMax, thetaMax[0]-norminvg(PC,0,thetaMax[1]/C), thetaMax[1] / C))
        elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
            CI[iConfP,0] = np.exp(thetaMin[0]+thetaMin[1]/C*(np.log(-np.log(1-pCorrMin))-np.log(-np.log(1-PC))))
            CI[iConfP,1] = np.exp(thetaMax[0]+thetaMax[1]/C*(np.log(-np.log(1-pCorrMax))-np.log(-np.log(1-PC))))
        elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
            CI[iConfP,0] = (t1icdf(pCorrMin)-t1icdf(PC))*thetaMin[1] / C + thetaMin[0]
            CI[iConfP,1] = (t1icdf(pCorrMax)-t1icdf(PC))*thetaMax[1] / C + thetaMax[0]
        else:
             raise ValueError('unknown sigmoid function')

        if (pCorrMin>1) | (pCorrMin<0):
            CI[iConfP,0] = np.nan

        if (pCorrMax>1) | (pCorrMax<0):
            CI[iConfP,1] = np.nan

    return (threshold,CI)


def biasAna(data1, data2,options):
    """ function biasAna(data1,data2,options)
 runs a short analysis to see whether two 2AFC datasets have a bias and
 whether it can be explained with a "finger bias"-> a bias in guessing """

    options = dict()
    options['borders'] = np.empty([5,2])
    options['borders'][:] = np.nan
    options['expType'] = 'YesNo'

    options['priors'] = [None]*5
    options['priors'][3] = lambda x: scipy.stats.beta.pdf(x,2,2)
    options['borders'][2,:] = np.array([0,.1])
    options['borders'][3,:] = np.array([.11,.89])
    options['fixedPars'] = np.ones([5,1])*np.nan
    options['fixedPars'][4] = 0
    options['stepN']   = np.array([40,40,40,40,1])
    options['mbStepN'] = np.array([30,30,20,20,1])

    resAll = psignifit(np.append(data1, data2, axis=0),options)
    res1 = psignifit(data1,options)
    res2 = psignifit(data2,options)

    plot.plt.figure()
    a1 = plot.plt.axes([0.15,4.35/6,0.75,1.5/6])

    plot.plotPsych(resAll,showImediate=False)
    plot.plt.hold(True)

    plot.plotPsych(res1, lineColor= [1,0,0], dataColor = [1,0,0],showImediate=False)
    plot.plotPsych(res2,lineColor= [0,0,1], dataColor = [0,0,1],showImediate=False)
    plot.plt.ylim([0,1])

    a2 = plot.plt.axes([0.15,3.35/6,0.75,0.5/6])

    plot.plotMarginal(resAll,dim = 0,prior = False, CIpatch = False, lineColor = [0,0,0],showImediate=False)
    plot.plt.hold(True)

    plot.plotMarginal(res1,dim = 0,lineColor = [1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim = 0,lineColor=[0,0,1],showImediate=False)
    a2.relim()
    a2.autoscale_view()

    a3 = plot.plt.axes([0.15,2.35/6,0.75,0.5/6])
    plot.plotMarginal(resAll,dim = 1,prior = False, CIpatch=False, lineColor = [0,0,0],showImediate=False)
    plot.plt.hold(True)

    plot.plotMarginal(res1,dim = 1,lineColor=[1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim = 1,lineColor=[0,0,1],showImediate=False)
    a3.relim()
    a3.autoscale_view()

    a4 = plot.plt.axes([0.15,1.35/6,0.75,0.5/6])

    plot.plotMarginal(resAll,dim = 2, prior = False, CIpatch = False, lineColor = [0,0,0],showImediate=False)
    plot.plt.hold(True)

    plot.plotMarginal(res1,dim = 2, lineColor=[1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim=2, lineColor=[0,0,1],showImediate=False)
    a4.relim()
    a4.autoscale_view()

    a5 = plot.plt.axes([0.15,0.35/6,0.75,0.5/6])

    plot.plotMarginal(resAll,dim = 3, prior = False, CIpatch = False, lineColor = [0,0,0],showImediate=False)
    plot.plt.hold(True)

    plot.plotMarginal(res1,dim = 3, lineColor=[1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim = 3, lineColor=[0,0,1],showImediate=False)
    a5.set_xlim([0,1])
    a5.relim()
    a5.autoscale_view()

    plot.plt.draw()

