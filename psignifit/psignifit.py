# -*- coding: utf-8 -*-

import warnings
from copy import deepcopy as _deepcopy
from functools import partial
from typing import Dict, Tuple

import numpy as np
import scipy

from . import priors as _priors
from . import psigniplot as plot
from . import sigmoids
from .bounds import parameter_bounds
from .conf import Conf
from .getConfRegion import getConfRegion
from .getSeed import getSeed
from .getWeights import getWeights
from .gridSetting import gridSetting
from .likelihood import posterior_grid, max_posterior
from .marginalize import marginalize
from .typing import ExperimentType, ParameterBounds, Prior
from .utils import (norminv, norminvg, t1icdf, pool_data, integral_weights,
                    PsignifitException, normalize, get_grid)


def psignifit(data, conf=None, **kwargs):
    """
    Main function for fitting psychometric functions function

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

    sigmoid = sigmoids.sigmoid_by_name(conf.sigmoid, PC=conf.thresh_PC, alpha=conf.width_alpha)
    data = _check_data(data, verbose=conf.verbose, logspace=sigmoid.logspace,
                       has_user_stimulus_range=conf.stimulus_range is not None)
    levels, ncorrect, ntrials = data[:, 0], data[:, 1], data[:, 2]

    stimulus_range = conf.stimulus_range
    if stimulus_range is None:
        stimulus_range = (levels.min(), levels.max())
    if sigmoid.logspace:
        stimulus_range = (np.log(stimulus_range[0]), np.log(stimulus_range[1]))
        levels = np.log(levels)

    width_min = conf.width_min
    if width_min is None:
        if conf.stimulus_range is None:
            width_min = np.diff(np.unique(levels)).min()
        else:
            # For user specified stimulus range, use very conservative estimate of width_min.
            # https: // en.wikipedia.org / wiki / Unit_in_the_last_place
            width_min = 100 * np.spacing(stimulus_range[1])

    if ntrials.max() == 1 or len(levels) > conf.pool_max_blocks:
        warnings.warn("""Pooling data, to avoid problems with n=1 blocks or to save time fitting
            because you have more than 25 blocks.
            You can force acceptance of your blocks by increasing conf.pool_max_blocks""")
        return pool_data(data, xtol=conf.pool_xtol, max_gap=conf.pool_max_gap, max_length=conf.pool_max_length)

    priors = _priors.default_priors(stimulus_range, width_min, conf.width_alpha,
                                    conf.beta_prior, thresh_PC=conf.thresh_PC)
    if conf.priors is not None:
        priors.update(conf.priors)
    _priors.check_priors(priors, stimulus_range, width_min)

    bounds = parameter_bounds(wmin=width_min, etype=conf.experiment_type, srange=stimulus_range,
                              alpha=conf.width_alpha, echoices = conf.experiment_choices)
    if conf.bounds is not None:
        bounds.update(conf.bounds)
    if conf.fixed_parameters is not None:
        for param, value in conf.fixed_parameters.items():
            bounds[param] = (value, value)


    # normalize priors to first choice of bounds
    for parameter, prior in priors.items():
        priors[parameter] = normalize(prior, bounds[parameter])

    fit_dict = _fit_parameters(data, bounds, priors, sigmoid, conf.steps_moving_bounds, conf.max_bound_value, conf.grid_steps)
    results = {'sigmoid_parameters': fit_dict}

    # take care of confidence intervals/condifence region

    # XXX FIXME: take care of post-ptocessing later
    # ''' after processing '''
    # # check that the marginals go to nearly 0 at the bounds of the grid
    # if options['verbose'] > -5:
    ### TODO ###
    # when the marginal on the bounds not smaller than 1/1000 of the peak
    # it means that the prior of the corresponding parameter has an influence of
    # the result ( 1)prior may be too narrow, 2) you know what you are doing).
    # if they were using the default, this is a bug in the software or your data
    # are highly unusual, if they changed the defaults the error can be more verbose
    # "the choice of your prior or of your bounds has a significant influence on the
    # confidence interval widths and or the max posterior_grid estimate"
    #########
    # if result['marginals'][0][0] * result['marginalsW'][0][0] > .001:
    # warnings.warn('psignifit:boundWarning\n'\
    # 'The marginal for the threshold is not near 0 at the lower bound.\n'\
    # 'This indicates that smaller Thresholds would be possible.')
    # if result['marginals'][0][-1] * result['marginalsW'][0][-1] > .001:
    # warnings.warn('psignifit:boundWarning\n'\
    # 'The marginal for the threshold is not near 0 at the upper bound.\n'\
    # 'This indicates that your data is not sufficient to exclude much higher thresholds.\n'\
    # 'Refer to the paper or the manual for more info on this topic.')
    # if result['marginals'][1][0] * result['marginalsW'][1][0] > .001:
    # warnings.warn('psignifit:boundWarning\n'\
    # 'The marginal for the width is not near 0 at the lower bound.\n'\
    # 'This indicates that your data is not sufficient to exclude much lower widths.\n'\
    # 'Refer to the paper or the manual for more info on this topic.')
    # if result['marginals'][1][-1] * result['marginalsW'][1][-1] > .001:
    # warnings.warn('psignifit:boundWarning\n'\
    # 'The marginal for the width is not near 0 at the lower bound.\n'\
    # 'This indicates that your data is not sufficient to exclude much higher widths.\n'\
    # 'Refer to the paper or the manual for more info on this topic.')

    # result['timestamp'] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # if options['instantPlot']:
    # plot.plotPsych(result)

    return results


def _fit_parameters(data: np.ndarray, bounds: ParameterBounds,
                    priors: Dict[str, Prior], sigmoid: sigmoids.Sigmoid,
                    steps_moving_bounds: Dict[str, int], max_bound_value: float,
                    grid_steps: Dict[str, int]) -> Dict[str, float]:
    """ Fit sigmoid parameters in a three step procedure. """
    # do first sparse grid posterior_grid evaluation
    grid = get_grid(bounds, steps_moving_bounds)
    posteriors_sparse, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # normalize the posterior_grid
    posterior_volumes = posteriors_sparse * integral_weights([grid_value for _, grid_value in sorted(grid.items())])
    posterior_integral = posterior_volumes.sum()
    # indices on the grid of the volumes that contribute more than `tol` to the overall integral
    mask = np.nonzero(posterior_volumes / posterior_integral >= max_bound_value)
    for idx, parm in enumerate(sorted(bounds.keys())):
        pgrid = grid[parm]
        # get the indeces for this parameter's axis and sort it
        axis = np.sort(mask[idx])
        # new bounds are the extrema of this axis, but enlarged of one element
        # in both directions
        left = max(0, axis[0] - 1)
        right = min(axis[-1] + 1, len(pgrid) - 1)
        # update the bounds
        bounds[parm] = (pgrid[left], pgrid[right])
    # do dense grid posterior_grid evaluation
    grid = get_grid(bounds, grid_steps)
    posteriors, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)
    print('fit0', grid_max)
    fixed_param = {}
    for parm_name, parm_values in grid.items():
        if parm_values is None:
            fixed_param[parm_name] = parm_values
        elif len(parm_values) <= 1:
            fixed_param[parm_name] = parm_values[0]
    fit_dict = max_posterior(data, param_init=grid_max, param_fixed=fixed_param, sigmoid=sigmoid, priors=priors)
    return fit_dict


def _check_data(data: np.ndarray, verbose: bool, logspace: bool, has_user_stimulus_range: bool) -> np.ndarray:
    """ Check data format, type and range.

    Args:
        data: The data matrix with columns levels, number of correct and number of trials
        verbose: Print warnings
        logspace: Data should be used logarithmically
        is_user_stimulus_range: User configured the stimulus range
    Returns:
        data as float numpy array
    """
    data = np.asarray(data, dtype=float)
    if len(data.shape) != 2 and data.shape[1] != 3:
        raise PsignifitException("Expects data to be two dimensional with three columns, got {data.shape = }")
    levels, ncorrect, ntrials = data[:, 0], data[:, 1], data[:, 2]

    # levels should show some variance
    if levels.max() == levels.min():
        raise PsignifitException('Your stimulus levels are all identical.'
                                 ' They can not be fitted by a sigmoid!')
    # ncorrect and ntrials should be integers
    if not np.allclose(ncorrect, ncorrect.astype(int)):
        raise PsignifitException(
            'The number correct column contains non integer'
            ' numbers!')
    if not np.allclose(ntrials, ntrials.astype(int)):
        raise PsignifitException('The number of trials column contains non'
                                 ' integer numbers!')
    if logspace and levels.min() <= 0:
        raise PsignifitException(f'Sigmoid {data.sigmoid} expects positive stimulus level data.')

    # warning if many blocks were measured
    if verbose and len(levels) >= 25 and not has_user_stimulus_range:
        warnings.warn(f"""The data you supplied contained {len(levels)}>= 25 stimulus levels.
            Did you sample adaptively?
            If so please specify a range which contains the whole psychometric function in
            conf.stimulus_range.
            An appropriate prior prior will be then chosen. For now we use the standard
            heuristic, assuming that the psychometric function is covered by the stimulus
            levels,which is frequently invalid for adaptive procedures!""")

    if verbose and ntrials.max() <= 5 and not is_user_stimulus_range:
        warnings.warn("""All provided data blocks contain <= 5 trials.
            Did you sample adaptively?
            If so please specify a range which contains the whole psychometric function in
            conf.stimulus_range.
            An appropriate prior prior will be then chosen. For now we use the standard
            heuristic, assuming that the psychometric function is covered by the stimulus
            levels, which is frequently invalid for adaptive procedures!""")
    return data



def psignifitFast(data, options):
    """
    this uses changed settings for the fit to obtain a fast point estimate to
    your data.
    The mean estimate with these settings is very crude, the MAP estimate is
    better, but takes a bit of time for the optimization (~100 ms)
    """

    warnings.warn('You use the speed optimized version of this program. \n' \
                  'This is NOT suitable for the final analysis, but meant for online analysis, adaptive methods etc. \n' \
                  'It has not been tested how good the estimates from this method are!')

    options['stepN'] = [20, 20, 10, 10, 1]
    options['mbStepN'] = [20, 20, 10, 10, 1]
    options['fixedPars'] = np.array([np.NaN, np.NaN, np.NaN, np.NaN, 0.0])
    options['fastOptim'] = True

    res = psignifit(data, options)

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
    print(options)
    d = len(options['bounds'])
    result = {'X1D': [], 'marginals': [], 'marginalsX': [], 'marginalsW': []}
    '''Choose grid dynamically from data'''
    if options['dynamicGrid']:
        # get seed from linear regression with logit transform
        Seed = getSeed(data, options)

        # further optimize the logliklihood to obtain a good estimate of the MAP
        if options.experiment_type == ExperimentType.YES_NO:
            calcSeed = lambda X: -_l.logLikelihood(data, options, X[0], X[1], X[
                2], X[3], X[4])
            Seed = scipy.optimize.fmin(func=calcSeed, x0=Seed)
        elif options.experiment_type == ExperimentType.N_AFC:
            calcSeed = lambda X: -_l.logLikelihood(data, options, X[0], X[1], X[
                2], 1 / options.experiment_choices, X[3])
            Seed = scipy.optimize.fmin(func=calcSeed, x0=[Seed[0:2], Seed[4]])
            Seed = [Seed[0:2], 1 / options.experiment_choices,
                    Seed[3]]  # ToDo check whether row or colum vector
        result['X1D'] = gridSetting(data, options, Seed)

    else:  # for types which do not need a MAP estimate
        if (options['gridSetType'] == 'priorlike' or
                options['gridSetType'] == 'STD' or
                options['gridSetType'] == 'exp' or
                options['gridSetType'] == '4power'):
            result['X1D'] = gridSetting(data, options)
        else:  # Use a linear grid
            for idx in range(0, d):
                # If there is an actual Interval
                if options['bounds'][idx, 0] < options['bounds'][idx, 1]:

                    result['X1D'].append(
                        np.linspace(options['bounds'][idx, 0],
                                    options['bounds'][idx, 1],
                                    num=options['stepN'][idx]))
                # if parameter was fixed
                else:
                    result['X1D'].append(np.array([options['bounds'][idx, 0]]))
    '''Evaluate posterior_grid and form it into a posterior_grid'''

    (result['Posterior'],
     result['logPmax']) = _l.posterior_grid(data, options, result['X1D'])
    result['weight'] = getWeights(result['X1D'])
    integral = np.sum(
        np.array(result['Posterior'][:]) * np.array(result['weight'][:]))
    result['Posterior'] = result['Posterior'] / integral
    result['integral'] = integral
    '''Compute marginal distributions'''

    for idx in range(0, d):
        m, mX, mW = marginalize(result, np.array(idx))
        result['marginals'].append(m)
        result['marginalsX'].append(mX)
        result['marginalsW'].append(mW)

    result['marginals'] = np.squeeze(result['marginals'])
    result['marginalsX'] = np.squeeze(result['marginalsX'])
    result['marginalsW'] = np.squeeze(result['marginalsW'])
    '''Find point estimate'''
    if (options['estimateType'] in ['MAP', 'MLE']):
        # get MLE estimate

        # start at most likely grid point
        index = np.where(
            result['Posterior'] == np.max(result['Posterior'].ravel()))

        Fit = np.zeros([d, 1])
        for idx in range(0, d):
            Fit[idx] = result['X1D'][idx][index[idx]]

        if options.experiment_type == ExperimentType.YES_NO:
            fun = lambda X, f: -_l.logLikelihood(data, options,
                                                 [X[0], X[1], X[2], X[3], X[4]])
            x0 = _deepcopy(Fit)
            a = None

        elif options.experiment_type == ExperimentType.N_AFC:
            # def func(X,f):
            #    return -_l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            # fun = func
            fun = lambda X, f: -_l.logLikelihood(data, options,
                                                 [X[0], X[1], X[2], f, X[3]])
            x0 = _deepcopy(Fit[0:3])  # Fit[3]  is excluded
            x0 = np.append(x0, _deepcopy(Fit[4]))
            a = np.array([1 / options.experiment_choices])

        elif options.experiment_type == ExperimentType.EQ_ASYMPTOTE:
            fun = lambda X, f: -_l.logLikelihood(data, options,
                                                 [X[0], X[1], X[2], f, X[3]])
            x0 = _deepcopy(Fit[0:3])
            x0 = np.append(x0, _deepcopy(Fit[4]))
            a = np.array([np.nan])

        else:
            raise ValueError('unknown experiment_type')

        if options['fastOptim']:
            Fit = scipy.optimize.fmin(fun,
                                      x0,
                                      args=(a,),
                                      xtol=0,
                                      ftol=0,
                                      maxiter=100,
                                      maxfun=100)
            warnings.warn('changed options for optimization')
        else:
            Fit = scipy.optimize.fmin(fun, x0, args=(a,), disp=False)

        if options.experiment_type == ExperimentType.YES_NO:
            result['Fit'] = _deepcopy(Fit)
        elif options.experiment_type == ExperimentType.N_AFC:
            fit = _deepcopy(Fit[0:3])
            fit = np.append(fit, np.array([1 / options.experiment_choices]))
            fit = np.append(fit, _deepcopy(Fit[3]))
            result['Fit'] = fit
        elif options.experiment_type == ExperimentType.EQ_ASYMPTOTE:
            fit = _deepcopy(Fit[0:3])
            fit = np.append(fit, Fit[2])
            fit = np.append(fit, Fit[3])
            result['Fit'] = fit
        else:
            raise ValueError('unknown experiment_type')

        par_idx = np.where(np.isnan(options['fixedPars']) == False)
        for idx in par_idx[0]:
            result['Fit'][idx] = options['fixedPars'][idx]

    elif options['estimateType'] == 'mean':
        # get mean estimate
        Fit = np.zeros([d, 1])
        for idx in range[0:d]:
            Fit[idx] = np.sum(result['marginals'][idx] *
                              result['marginalsW'][idx] *
                              result['marginalsX'][idx])

        result['Fit'] = _deepcopy(Fit)
        Fit = np.empty(Fit.shape)
    '''Include input into result'''
    result['options'] = options  # no copies here, because they are not changing
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
        raise ValueError(
            'Result needs to contain a resulting fit generated by psignifit')

    # calculate point estimate -> transform only the fit

    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC = result['options']['threshPC']
    else:
        PC = 0.5

    if result['options']['sigmoidName'][0:3] == 'neg':
        PC = 1 - PC

    sigName = result['options']['sigmoidName']

    if sigName in ['norm', 'gauss', 'neg_norm', 'neg_gauss']:
        C = norminv(1 - alpha) - norminv(alpha)
        normalizedStimLevel = (stimLevel - theta0[0]) / theta0[1] * C
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized * C / theta0[1]
    elif sigName in ['logistic', 'neg_logistic']:
        C = 2 * np.log(1 / alpha - 1) / theta0[1]
        d = np.log(1 / PC - 1)
        slope = C * np.exp(-C * (stimLevel - theta0[0]) +
                           d) / (1 + np.exp(-C *
                                            (stimLevel - theta0[0]) + d)) ** 2
    elif sigName in ['gumbel', 'neg_gumbel']:
        C = np.log(-np.log(alpha)) - np.log(-np.log(1 - alpha))
        stimLevel = C / theta0[1] * (stimLevel -
                                     theta0[0]) + np.log(-np.log(1 - PC))
        slope = C / theta0[1] * np.exp(-np.exp(stimLevel)) * np.exp(stimLevel)
    elif sigName in ['rgumbel', 'neg_rgumbel']:  # reversed gumbel
        C = np.log(-np.log(1 - alpha)) - np.log(-np.log(alpha))
        stimLevel = C / theta0[1] * (stimLevel -
                                     theta0[0]) + np.log(-np.log(PC))
        slope = -C / theta0[1] * np.exp(-np.exp(stimLevel)) * np.exp(stimLevel)
    elif sigName in ['logn', 'neg_logn']:
        C = norminv(1 - alpha) - norminv(alpha)
        normalizedStimLevel = (np.log(stimLevel) - theta0[0]) / theta0[1]
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized * C / theta0[1] / stimLevel
    elif sigName in ['Weibull', 'weibull', 'neg_Weibull', 'neg_weibull']:
        C = np.log(-np.log(alpha)) - np.log(-np.log(1 - alpha))
        stimLevelNormalized = C / theta0[1] * (
                np.log(stimLevel) - theta0[0]) + np.log(-np.log(1 - PC))
        slope = C / theta0[1] * np.exp(-np.exp(stimLevelNormalized)) * np.exp(
            stimLevelNormalized)
        slope = slope / stimLevel
    elif sigName in [
        'tdist', 'student', 'heavytail', 'neg_tdist', 'neg_student',
        'neg_heavytail'
    ]:
        # student T distribution with 1 df --> heavy tail distribution
        C = (my_t1icdf(1 - alpha) - my_t1icdf(alpha))
        stimLevel = (stimLevel - theta0[0]) / theta0[1] * C + my_t1icdf(PC)
        slope = C / theta0[1] * t.pdf(stimLevel, df=1)
    else:
        raise ValueError('unknown sigmoid function')

    slope = (1 - theta0[2] - theta0[3]) * slope

    if result['options']['sigmoidName'][0:3] == 'neg':
        slope = -slope
    return slope


def getSlopePC(result, pCorrect, unscaled=False):
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
        raise ValueError(
            'Result needs to contain a resulting fit generated by psignifit')

    # calculate point estimate -> transform only the fit

    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC = result['options']['threshPC']
    else:
        PC = 0.5

    if unscaled:
        assert ((pCorrect > 0) & (pCorrect < 1)), 'pCorrect must be in ]0,1[ '
        pCorrectUnscaled = pCorrect
    else:
        assert ((pCorrect > theta0[3]) & (pCorrect < (1 - theta0[2]))
                ), 'pCorrect must lay btw {:.2f} and {:.2f}'.format(
            theta0[3], (1 - theta0[2]))
        pCorrectUnscaled = (pCorrect - theta0[3]) / (1 - theta0[2] - theta0[3])
    ''' find the (normalized) stimulus level, where the given percent correct is
    reached and evaluate slope there'''
    sigName = result['options']['sigmoidName'].lower()

    if sigName[0:3] == 'neg':
        pCorrectUnscaled = 1 - pCorrectUnscaled
        PC = 1 - PC

    if sigName in ['norm', 'gauss', 'neg_norm', 'neg_gauss']:
        C = norminv(1 - alpha) - norminv(alpha)
        normalizedStimLevel = norminv(pCorrectUnscaled)
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized * C / theta0[1]
    elif sigName in ['logistic', 'neg_logistic']:
        stimLevel = theta0[0] - theta0[1] * np.log(
            (1 / pCorrectUnscaled - 1) -
            np.log(1 / PC - 1)) / 2 / np.log(1 / alpha - 1)
        C = 2 * np.log(1 / alpha - 1) / theta0[1]
        d = np.log(1 / PC - 1)
        slope = C * np.exp(-C * (stimLevel - theta0[0]) +
                           d) / (1 + np.exp(-C *
                                            (stimLevel - theta0[0]) + d)) ** 2
    elif sigName in ['gumbel', 'neg_gumbel']:
        C = np.log(-np.log(alpha)) - np.log(-np.log(1 - alpha))
        stimLevel = np.log(-np.log(1 - pCorrectUnscaled))
        slope = C / theta0[1] * np.exp(-np.exp(stimLevel)) * np.exp(stimLevel)
    elif sigName in ['rgumbel', 'neg_rgumbel']:  # reversed gumbel
        C = np.log(-np.log(1 - alpha)) - np.log(-np.log(alpha))
        stimLevel = np.log(-np.log(pCorrectUnscaled))
        slope = -C / theta0[1] * np.exp(-np.exp(stimLevel)) * np.exp(stimLevel)
    elif sigName in ['logn', 'neg_logn']:
        C = norminv(1 - alpha) - norminv(alpha)
        stimLevel = np.exp(
            norminvg(pCorrectUnscaled,
                     theta0[0] - norminvg(PC, 0, theta0[1] / C), theta0[1] / C))
        normalizedStimLevel = norminv(pCorrectUnscaled)
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized * C / theta0[1] / stimLevel
    elif sigName in ['Weibull', 'weibull', 'neg_Weibull', 'neg_weibull']:
        C = np.log(-np.log(alpha)) - np.log(-np.log(1 - alpha))
        stimLevel = np.exp(
            theta0[0] + theta0[1] / C *
            (np.log(-np.log(1 - pCorrectUnscaled)) - np.log(-np.log(1 - PC))))
        stimLevelNormalized = np.log(-np.log(1 - pCorrectUnscaled))
        slope = C / theta0[1] * np.exp(-np.exp(stimLevelNormalized)) * np.exp(
            stimLevelNormalized)
        slope = slope / stimLevel
    elif sigName in [
        'tdist', 'student', 'heavytail', 'neg_tdist', 'neg_student',
        'neg_heavytail'
    ]:
        # student T distribution with 1 df --> heavy tail distribution
        C = (t1icdf(1 - alpha) - t1icdf(alpha))
        stimLevel = t1icdf(pCorrectUnscaled)
        slope = C / theta0[1] * t.pdf(stimLevel, df=1)
    else:
        raise ValueError('unknown sigmoid function')

    slope = (1 - theta0[2] - theta0[3]) * slope

    if sigName[0:3] == 'neg':
        slope = -slope
    return slope


def getThreshold(result, pCorrect, unscaled=False):
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
        raise ValueError(
            'Result needs to contain a resulting fit generated by psignifit.')
    if 'conf_Intervals' in result.keys():
        CIs = _deepcopy(result['conf_Intervals'])
    else:
        raise ValueError(
            'Result needs to contain confidence intervals for the fitted parameter.'
        )

    if unscaled:  # set asymptotes to 0 for everything.
        theta0[2] = 0
        theta0[3] = 0
        CIs[2:4, :] = 0

    assert ((np.array(pCorrect) > theta0[3]) & (np.array(pCorrect) <
                                                (1 - theta0[2]))
            ), 'The threshold percent correct is not reached by the sigmoid!'

    pCorrectUnscaled = (pCorrect - theta0[3]) / (1 - theta0[2] - theta0[3])
    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC = result['options']['threshPC']
    else:
        PC = 0.5

    sigName = result['options']['sigmoidName'].lower()
    if sigName[0:3] == 'neg':
        PC = 1 - PC
        pCorrectUnscaled = 1 - pCorrectUnscaled

    if sigName in ['norm', 'gauss', 'neg_norm', 'neg_gauss']:
        C = norminv(1 - alpha) - norminv(alpha)
        threshold = norminvg(pCorrectUnscaled,
                             theta0[0] - norminvg(PC, 0, theta0[1] / C),
                             theta0[1] / C)
    elif sigName in ['logistic', 'neg_logistic']:
        threshold = theta0[0] - theta0[1] * (
                np.log(1 / pCorrectUnscaled - 1) -
                np.log(1 / PC - 1)) / 2 / np.log(1 / alpha - 1)
    elif sigName in ['gumbel', 'neg_gumbel']:
        C = np.log(-np.log(alpha)) - np.log(-np.log(1 - alpha))
        threshold = theta0[0] + (np.log(-np.log(1 - pCorrectUnscaled)) -
                                 np.log(-np.log(1 - PC))) * theta0[1] / C
    elif sigName in ['rgumbel', 'neg_rgumbel']:
        C = np.log(-np.log(1 - alpha)) - np.log(-np.log(alpha))
        threshold = theta0[0] + (np.log(-np.log(pCorrectUnscaled)) -
                                 np.log(-np.log(PC))) * theta0[1] / C
    elif sigName in ['logn', 'neg_logn']:
        C = norminv(1 - alpha) - norminv(alpha)
        threshold = np.exp(
            norminvg(pCorrectUnscaled,
                     theta0[0] - norminvg(PC, 0, theta0[1] / C), theta0[1] / C))
    elif sigName in ['Weibull', 'weibull', 'neg_Weibull', 'neg_weibull']:
        C = np.log(-np.log(alpha)) - np.log(-np.log(1 - alpha))
        threshold = np.exp(
            theta0[0] + theta0[1] / C *
            (np.log(-np.log(1 - pCorrectUnscaled)) - np.log(-np.log(1 - PC))))
    elif sigName in [
        'tdist', 'student', 'heavytail', 'neg_tdist', 'neg_student',
        'neg_heavytail'
    ]:
        C = (t1icdf(1 - alpha) - t1icdf(alpha))
        threshold = (t1icdf(pCorrectUnscaled) -
                     t1icdf(PC)) * theta0[1] / C + theta0[0]
    else:
        raise ValueError('unknown sigmoid function')
    """ calculate CI -> worst case in parameter confidence intervals """

    warnings.warn(
        'The CIs computed by this method are only upper bounds. For more accurate inference change threshPC in the options.'
    )
    CI = np.zeros([len(result['options']['confP']), 2])
    for iConfP in range(0, len(result['options']['confP'])):

        if sigName[0:3] == 'neg':
            if pCorrectUnscaled < PC:
                thetaMax = [
                    CIs[0, 1, iConfP], CIs[1, 0, iConfP], CIs[2, 0, iConfP],
                    CIs[3, 1, iConfP], 0
                ]
                thetaMin = [
                    CIs[0, 0, iConfP], CIs[1, 1, iConfP], CIs[2, 1, iConfP],
                    CIs[3, 0, iConfP], 0
                ]
            else:
                thetaMin = [
                    CIs[0, 1, iConfP], CIs[1, 1, iConfP], CIs[2, 0, iConfP],
                    CIs[3, 1, iConfP], 0
                ]
                thetaMax = [
                    CIs[0, 0, iConfP], CIs[1, 0, iConfP], CIs[2, 1, iConfP],
                    CIs[3, 0, iConfP], 0
                ]
        else:
            if pCorrectUnscaled > PC:
                thetaMin = [
                    CIs[0, 0, iConfP], CIs[1, 0, iConfP], CIs[2, 0, iConfP],
                    CIs[3, 1, iConfP], 0
                ]
                thetaMax = [
                    CIs[0, 1, iConfP], CIs[1, 1, iConfP], CIs[2, 1, iConfP],
                    CIs[3, 0, iConfP], 0
                ]
            else:
                thetaMin = [
                    CIs[0, 0, iConfP], CIs[1, 1, iConfP], CIs[2, 0, iConfP],
                    CIs[3, 1, iConfP], 0
                ]
                thetaMax = [
                    CIs[0, 1, iConfP], CIs[1, 0, iConfP], CIs[2, 1, iConfP],
                    CIs[3, 0, iConfP], 0
                ]
        pCorrMin = (pCorrect - thetaMin[3]) / (1 - thetaMin[2] - thetaMin[3])
        pCorrMax = (pCorrect - thetaMax[3]) / (1 - thetaMax[2] - thetaMax[3])
        if sigName in ['norm', 'gauss', 'neg_norm', 'neg_gauss']:
            CI[iConfP, 0] = norminvg(
                pCorrMin, thetaMin[0] - norminvg(PC, 0, thetaMin[1] / C),
                          thetaMin[1] / C)
            CI[iConfP, 1] = norminvg(
                pCorrMax, thetaMax[0] - norminvg(PC, 0, thetaMax[1] / C),
                          thetaMax[1] / C)
        elif sigName in ['logistic', 'neg_logistic']:
            CI[iConfP, 0] = thetaMin[0] - thetaMin[1] * (
                    np.log(1 / pCorrMin - 1) -
                    np.log(1 / PC - 1)) / 2 / np.log(1 / alpha - 1)
            CI[iConfP, 1] = thetaMax[0] - thetaMax[1] * (
                    np.log(1 / pCorrMax - 1) -
                    np.log(1 / PC - 1)) / 2 / np.log(1 / alpha - 1)
        elif sigName in ['gumbel', 'neg_gumbel']:
            CI[iConfP, 0] = thetaMin[0] + (
                    np.log(-np.log(1 - pCorrMin)) -
                    np.log(-np.log(1 - PC))) * thetaMin[1] / C
            CI[iConfP, 1] = thetaMax[0] + (
                    np.log(-np.log(1 - pCorrMax)) -
                    np.log(-np.log(1 - PC))) * thetaMax[1] / C
        elif sigName in ['rgumbel', 'neg_rgumbel']:
            CI[iConfP, 0] = thetaMin[0] + (np.log(-np.log(pCorrMin)) - np.log(
                -np.log(PC))) * thetaMin[1] / C
            CI[iConfP, 1] = thetaMax[0] + (np.log(-np.log(pCorrMax)) - np.log(
                -np.log(PC))) * thetaMax[1] / C
        elif sigName in ['logn', 'neg_logn']:
            CI[iConfP, 0] = np.exp(
                norminvg(pCorrMin,
                         thetaMin[0] - norminvg(PC, 0, thetaMin[1] / C),
                         thetaMin[1] / C))
            CI[iConfP, 1] = np.exp(
                norminvg(pCorrMax,
                         thetaMax[0] - norminvg(PC, 0, thetaMax[1] / C),
                         thetaMax[1] / C))
        elif sigName in ['Weibull', 'weibull', 'neg_Weibull', 'neg_weibull']:
            CI[iConfP, 0] = np.exp(
                thetaMin[0] + thetaMin[1] / C *
                (np.log(-np.log(1 - pCorrMin)) - np.log(-np.log(1 - PC))))
            CI[iConfP, 1] = np.exp(
                thetaMax[0] + thetaMax[1] / C *
                (np.log(-np.log(1 - pCorrMax)) - np.log(-np.log(1 - PC))))
        elif sigName in [
            'tdist', 'student', 'heavytail', 'neg_tdist', 'neg_student',
            'neg_heavytail'
        ]:
            CI[iConfP, 0] = (t1icdf(pCorrMin) -
                             t1icdf(PC)) * thetaMin[1] / C + thetaMin[0]
            CI[iConfP, 1] = (t1icdf(pCorrMax) -
                             t1icdf(PC)) * thetaMax[1] / C + thetaMax[0]
        else:
            raise ValueError('unknown sigmoid function')

        if (pCorrMin > 1) | (pCorrMin < 0):
            CI[iConfP, 0] = np.nan

        if (pCorrMax > 1) | (pCorrMax < 0):
            CI[iConfP, 1] = np.nan

    return (threshold, CI)


def biasAna(data1, data2, options=dict()):
    """ function biasAna(data1,data2,options)
    runs a short analysis to see whether two 2AFC datasets have a bias and
    whether it can be explained with a "finger bias"-> a bias in guessing """

    options = _deepcopy(options)
    options['bounds'] = np.empty([5, 2])
    options['bounds'][:] = np.nan
    options.experiment_type = ExperimentType.YES_NO

    options['priors'] = [None] * 5
    options['priors'][3] = lambda x: scipy.stats.beta.pdf(x, 2, 2)
    options['bounds'][2, :] = np.array([0, .1])
    options['bounds'][3, :] = np.array([.11, .89])
    options['fixedPars'] = np.ones([5, 1]) * np.nan
    options['fixedPars'][4] = 0
    options['stepN'] = np.array([40, 40, 40, 40, 1])
    options['mbStepN'] = np.array([30, 30, 20, 20, 1])

    resAll = psignifit(np.append(data1, data2, axis=0), options)
    res1 = psignifit(data1, options)
    res2 = psignifit(data2, options)

    plot.plt.figure()
    a1 = plot.plt.axes([0.15, 4.35 / 6, 0.75, 1.5 / 6])

    plot.plotPsych(resAll, showImediate=False)

    plot.plotPsych(res1,
                   lineColor=[1, 0, 0],
                   dataColor=[1, 0, 0],
                   showImediate=False)
    plot.plotPsych(res2,
                   lineColor=[0, 0, 1],
                   dataColor=[0, 0, 1],
                   showImediate=False)
    plot.plt.ylim([0, 1])

    a2 = plot.plt.axes([0.15, 3.35 / 6, 0.75, 0.5 / 6])

    plot.plotMarginal(resAll,
                      dim=0,
                      prior=False,
                      CIpatch=False,
                      lineColor=[0, 0, 0],
                      showImediate=False)

    plot.plotMarginal(res1, dim=0, lineColor=[1, 0, 0], showImediate=False)
    plot.plotMarginal(res2, dim=0, lineColor=[0, 0, 1], showImediate=False)
    a2.relim()
    a2.autoscale_view()

    a3 = plot.plt.axes([0.15, 2.35 / 6, 0.75, 0.5 / 6])
    plot.plotMarginal(resAll,
                      dim=1,
                      prior=False,
                      CIpatch=False,
                      lineColor=[0, 0, 0],
                      showImediate=False)

    plot.plotMarginal(res1, dim=1, lineColor=[1, 0, 0], showImediate=False)
    plot.plotMarginal(res2, dim=1, lineColor=[0, 0, 1], showImediate=False)
    a3.relim()
    a3.autoscale_view()

    a4 = plot.plt.axes([0.15, 1.35 / 6, 0.75, 0.5 / 6])

    plot.plotMarginal(resAll,
                      dim=2,
                      prior=False,
                      CIpatch=False,
                      lineColor=[0, 0, 0],
                      showImediate=False)

    plot.plotMarginal(res1, dim=2, lineColor=[1, 0, 0], showImediate=False)
    plot.plotMarginal(res2, dim=2, lineColor=[0, 0, 1], showImediate=False)
    a4.relim()
    a4.autoscale_view()

    a5 = plot.plt.axes([0.15, 0.35 / 6, 0.75, 0.5 / 6])

    plot.plotMarginal(resAll,
                      dim=3,
                      prior=False,
                      CIpatch=False,
                      lineColor=[0, 0, 0],
                      showImediate=False)

    plot.plotMarginal(res1, dim=3, lineColor=[1, 0, 0], showImediate=False)
    plot.plotMarginal(res2, dim=3, lineColor=[0, 0, 1], showImediate=False)
    a5.set_xlim([0, 1])
    a5.relim()
    a5.autoscale_view()

    plot.plt.draw()


def getDeviance(result, Nsamples=None):
    fit = result['Fit']
    data = result['data']
    pPred = fit[3] + (1 - fit[2] - fit[3]) * result['options']['sigmoidHandle'](
        data[:, 0], fit[0], fit[1])

    pMeasured = data[:, 1] / data[:, 2]
    loglikelihoodPred = data[:, 1] * np.log(pPred) + (data[:, 2] -
                                                      data[:, 1]) * np.log(
        (1 - pPred))
    loglikelihoodMeasured = data[:, 1] * np.log(pMeasured) + (
            data[:, 2] - data[:, 1]) * np.log((1 - pMeasured))
    loglikelihoodMeasured[pMeasured == 1] = 0
    loglikelihoodMeasured[pMeasured == 0] = 0

    devianceResiduals = -2 * np.sign(pMeasured - pPred) * (
            loglikelihoodMeasured - loglikelihoodPred)
    deviance = np.sum(np.abs(devianceResiduals))

    if Nsamples is None:
        return devianceResiduals, deviance
    else:
        samples_devianceResiduals = np.zeros((Nsamples, data.shape[0]))
        for iData in range(data.shape[0]):
            samp_dat = np.random.binomial(data[iData, 2], pPred[iData],
                                          Nsamples)
            pMeasured = samp_dat / data[iData, 2]
            loglikelihoodPred = samp_dat * np.log(pPred[iData]) + (
                    data[iData, 2] - samp_dat) * np.log(1 - pPred[iData])
            loglikelihoodMeasured = samp_dat * np.log(pMeasured) + (
                    data[iData, 2] - samp_dat) * np.log(1 - pMeasured)
            loglikelihoodMeasured[pMeasured == 1] = 0
            loglikelihoodMeasured[pMeasured == 0] = 0
            samples_devianceResiduals[:, iData] = -2 * np.sign(
                pMeasured - pPred[iData]) * (loglikelihoodMeasured -
                                             loglikelihoodPred)
        samples_deviance = np.sum(np.abs(samples_devianceResiduals), axis=1)
        return devianceResiduals, deviance, samples_devianceResiduals, samples_deviance
