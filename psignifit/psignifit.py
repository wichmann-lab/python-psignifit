# -*- coding: utf-8 -*-

import warnings
from typing import Dict

import numpy as np

from . import priors as _priors
from . import sigmoids
from .bounds import parameter_bounds
from .configuration import Configuration
from .likelihood import posterior_grid, max_posterior
from .result import Result
from .typing import ParameterBounds, Prior
from .utils import (pool_data, integral_weights, PsignifitException, normalize, get_grid)


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
        conf = Configuration(**kwargs)
    elif len(kwargs) > 0:
        # user shouldn't specify a conf object *and* kwargs simultaneously
        raise PsignifitException(
            "Can't handle conf together with other keyword arguments!")

    sigmoid = sigmoids.sigmoid_by_name(conf.sigmoid, PC=conf.thresh_PC, alpha=conf.width_alpha)
    data = _check_data(data, verbose=conf.verbose, logspace=sigmoid.logspace,
                       has_user_stimulus_range=conf.stimulus_range is not None)
    levels, ntrials = data[:, 0], data[:, 2]

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
                              alpha=conf.width_alpha, echoices=conf.experiment_choices)
    if conf.bounds is not None:
        bounds.update(conf.bounds)
    if conf.fixed_parameters is not None:
        for param, value in conf.fixed_parameters.items():
            bounds[param] = (value, value)

    # normalize priors to first choice of bounds
    for parameter, prior in priors.items():
        priors[parameter] = normalize(prior, bounds[parameter])

    fit_dict = _fit_parameters(data, bounds, priors, sigmoid, conf.steps_moving_bounds, conf.max_bound_value, conf.grid_steps)

    # take care of confidence intervals/condifence region

    # XXX FIXME: take care of post-ptocessing later
    # ''' after processing '''
    # # check that the marginals go to nearly 0 at the bounds of the grid
    # if options['verbose'] > -5:
    # ## TODO ###
    # when the marginal on the bounds not smaller than 1/1000 of the peak
    # it means that the prior of the corresponding parameter has an influence of
    # the result ( 1)prior may be too narrow, 2) you know what you are doing).
    # if they were using the default, this is a bug in the software or your data
    # are highly unusual, if they changed the defaults the error can be more verbose
    # "the choice of your prior or of your bounds has a significant influence on the
    # confidence interval widths and or the max posterior_grid estimate"
    # ########
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

    return Result(sigmoid_parameters=fit_dict,
                  configuration=conf)


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

    if verbose and ntrials.max() <= 5 and not has_user_stimulus_range:
        warnings.warn("""All provided data blocks contain <= 5 trials.
            Did you sample adaptively?
            If so please specify a range which contains the whole psychometric function in
            conf.stimulus_range.
            An appropriate prior prior will be then chosen. For now we use the standard
            heuristic, assuming that the psychometric function is covered by the stimulus
            levels, which is frequently invalid for adaptive procedures!""")
    return data
