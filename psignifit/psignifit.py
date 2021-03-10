# -*- coding: utf-8 -*-

import warnings
from typing import Dict

import numpy as np

from . import priors as _priors
from . import sigmoids
from .bounds import parameter_bounds, mask_bounds
from .configuration import Configuration
from .getConfRegion import confidence_intervals
from .likelihood import posterior_grid, max_posterior
from .result import Result
from .typing import ParameterBounds, Prior
from .utils import (PsignifitException, normalize, get_grid, check_data)


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
        conf = Configuration(**kwargs)
    elif len(kwargs) > 0:
        # user shouldn't specify a conf object *and* kwargs simultaneously
        raise PsignifitException(
            "Can't handle conf together with other keyword arguments!")

    sigmoid = conf.make_sigmoid()
    data = check_data(data, logspace=sigmoid.logspace)

    levels, ntrials = data[:, 0], data[:, 2]
    if conf.verbose:
        _warn_common_data_mistakes(levels, ntrials, has_user_stimulus_range=conf.stimulus_range is not None,
                                   pool_max_blocks=conf.pool_max_blocks)

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

    fit_dict, posteriors, grid = _fit_parameters(data, bounds, priors, sigmoid, conf.steps_moving_bounds,
                                                 conf.max_bound_value, conf.grid_steps)

    grid_values = [grid_value for _, grid_value in sorted(grid.items())]
    intervals = confidence_intervals(posteriors, grid_values, conf.confP, conf.CI_method)
    intervals_dict = {param: interval_per_p.tolist()
                      for param, interval_per_p in zip(sorted(grid.keys()), intervals)}
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
                  configuration=conf,
                  confidence_intervals=intervals_dict,
                  posterior=posteriors)


def _warn_common_data_mistakes(levels, ntrials, has_user_stimulus_range, pool_max_blocks) -> None:
    """ Show warnings for common mistakes.

    Checks for too many blocks and too few trials.
    The warnings recommend to use pooling or to manually specify stimulus ranges.

    Args:
        level: Array of stimulus level per block
        ntrial: Array of trial numbers per block
        has_user_stimulus_range: User configured the stimulus range
        pool_max_blocks: Maximum number of blocks until print of pool warning.
    Returns:
        None
    """
    if ntrials.max() == 1:
        warnings.warn("All blocks in data have only 1 trial.\n"
                      "To avoid problems during fitting, consider aggregating blocks of same stimulus level using "
                      "psignifit.pool_blocks(data).")
    if len(levels) > pool_max_blocks:
        warnings.warn(f"Expects at most {pool_max_blocks} blocks in data, got {len(levels)}.\n"
                      "To save fitting time, consider aggregating blocks of same stimulus level "
                      "psignifit.pool_blocks(data).\n"
                      "Hide this warning by increasing conf.pool_max_blocks.")
    # warning if many blocks were measured
    if len(levels) >= 25 and not has_user_stimulus_range:
        warnings.warn(f"""The data you supplied contained {len(levels)}>= 25 stimulus levels.
            Did you sample adaptively?
            If so please specify a range which contains the whole psychometric function in
            conf.stimulus_range.
            An appropriate prior prior will be then chosen. For now we use the standard
            heuristic, assuming that the psychometric function is covered by the stimulus
            levels,which is frequently invalid for adaptive procedures!""")
    if ntrials.max() <= 5 and not has_user_stimulus_range:
        warnings.warn("""All provided data blocks contain <= 5 trials.
            Did you sample adaptively?
            If so please specify a range which contains the whole psychometric function in
            conf.stimulus_range.
            An appropriate prior prior will be then chosen. For now we use the standard
            heuristic, assuming that the psychometric function is covered by the stimulus
            levels, which is frequently invalid for adaptive procedures!""")


def _fit_parameters(data: np.ndarray, bounds: ParameterBounds,
                    priors: Dict[str, Prior], sigmoid: sigmoids.Sigmoid,
                    steps_moving_bounds: Dict[str, int], max_bound_value: float,
                    grid_steps: Dict[str, int]) -> Dict[str, float]:
    """ Fit sigmoid parameters in a three step procedure.

    1. Estimate posterior on wide bounds with large steps in between.
    2. Estimate tighter bounds of relevant probability mass (> max_bound_values)
       and calculate posterior there using fine steps.
    3. Fit the sigmoid parameters using the fine and tight posterior grid.

    Args:
         data: Training samples.
         bounds: Dict of (min, max) parameter value.
         priors: Dict of prior function per parameter.
         sigmoid: Sigmoid function to fit.
         steps_moving_bounds: Dict of number of possible parameter values for loose bounds.
         max_bound_value: Threshold posterior on loose grid, used to tighten bounds.
         grid_steps: Dict of number of possible parameter values for tight bounds.

    Returns:
        fit_dict: Dict of fitted parameter value.
        posteriors: Probability per parameter combination.
        grid: Dict of possible parameter values.
    """
    # do first sparse grid posterior_grid evaluation
    grid = get_grid(bounds, steps_moving_bounds)
    posteriors_sparse, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # indices on the grid of the volumes that contribute more than `tol` to the overall integral
    tighter_bounds = mask_bounds(grid, posteriors_sparse >= max_bound_value)
    # do dense grid posterior_grid evaluation
    grid = get_grid(tighter_bounds, grid_steps)
    posteriors, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)
    print('fit0', grid_max)
    fixed_param = {}
    for parm_name, parm_values in grid.items():
        if parm_values is None:
            fixed_param[parm_name] = parm_values
        elif len(parm_values) <= 1:
            fixed_param[parm_name] = parm_values[0]
    fit_dict = max_posterior(data, param_init=grid_max, param_fixed=fixed_param, sigmoid=sigmoid, priors=priors)

    return fit_dict, posteriors, grid


