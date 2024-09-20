# -*- coding: utf-8 -*-

import warnings
from typing import Dict, Optional

import numpy as np

from . import sigmoids
from ._parameter import parameter_bounds, masked_parameter_bounds, parameter_grid
from ._configuration import Configuration
from ._confidence import confidence_intervals
from ._posterior import posterior_grid, maximize_posterior, marginalize_posterior
from ._priors import setup_priors
from ._result import Result
from ._typing import ParameterBounds, Prior
from ._utils import (PsignifitException, check_data)


def psignifit(data: np.ndarray, conf: Optional[Configuration] = None,
              return_posterior: bool = False, **kwargs) -> Result:
    """ Fit a psychometric function to experimental data.

    This function is the user interface for fitting psychometric functions to data.

    Notice that the parameters of the psychometric function are always fit in linear space, even
    for psychometric function that are supposed to work in a logarithmic space, like the Weibull
    function. It is left to the user to transform the stimulus level to logarithmic space before
    calling this function.

    pass your data in the n x 3 matrix of the form:
    [x-value, number correct, number of trials]

    options should be a dictionary in which you set the options for your fit.
    You can find a full overview over the options in demo002

    The result of this function is a dictionary, which contains all information the
    program produced for your fit. You can pass this as whole to all further
    processing function provided with psignifit. Especially to the plot functions.
    You can find an explanation for all fields of the result in demo006

    To get an introduction to basic usage start with demo001


    Args:
        data: Trials as described above.
        conf: Optional configuration object.
        return_posterior: If true, posterior matrix will be added to result object.
        kwargs: Configurations as function parameters.
    """
    if conf is None:
        conf = Configuration(**kwargs)
    elif len(kwargs) > 0:
        # user shouldn't specify a conf object *and* kwargs simultaneously
        raise PsignifitException(
            "Can't handle conf together with other keyword arguments!")

    sigmoid = conf.make_sigmoid()
    data = check_data(data)

    levels, ntrials = data[:, 0], data[:, 2]
    if conf.verbose:
        _warn_common_data_mistakes(levels, ntrials, has_user_stimulus_range=conf.stimulus_range is not None,
                                   pool_max_blocks=conf.pool_max_blocks)

    stimulus_range = conf.stimulus_range
    if stimulus_range is None:
        stimulus_range = (levels.min(), levels.max())

    width_min = conf.width_min
    if width_min is None:
        if conf.stimulus_range is None:
            width_min = np.diff(np.unique(levels)).min()
        else:
            # For user specified stimulus range, use very conservative estimate of width_min.
            # https: // en.wikipedia.org / wiki / Unit_in_the_last_place
            width_min = 100 * np.spacing(stimulus_range[1])

    bounds = parameter_bounds(min_width=width_min, experiment_type=conf.experiment_type, stimulus_range=stimulus_range,
                              alpha=conf.width_alpha, nafc_choices=conf.experiment_choices)
    if conf.bounds is not None:
        bounds.update(conf.bounds)
    if conf.fixed_parameters is not None:
        for param, value in conf.fixed_parameters.items():
            bounds[param] = (value, value)

    priors = setup_priors(custom_priors=conf.priors, bounds=bounds,
                          stimulus_range=stimulus_range, width_min=width_min, width_alpha=conf.width_alpha,
                          beta_prior=conf.beta_prior, threshold_perc_correct=conf.thresh_PC)
    fit_dict, posteriors, grid = _fit_parameters(data, bounds, priors, sigmoid, conf.steps_moving_bounds,
                                                 conf.max_bound_value, conf.grid_steps)

    grid_values = [grid_value for _, grid_value in sorted(grid.items())]
    intervals = confidence_intervals(posteriors, grid_values, conf.confP, conf.CI_method)
    intervals_dict = {param: interval_per_p.tolist()
                      for param, interval_per_p in zip(sorted(grid.keys()), intervals)}
    marginals = marginalize_posterior(grid, posteriors)

    if conf.verbose:
        _warn_marginal_sanity_checks(marginals)

    if not return_posterior:
        posteriors = None

    return Result(parameter_estimate=fit_dict,
                  configuration=conf,
                  confidence_intervals=intervals_dict,
                  parameter_values=grid,
                  prior_values={param: priors[param](values) for param, values in grid.items()},
                  marginal_posterior_values=marginals,
                  posterior_mass=posteriors,
                  data=data.tolist())


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


def _warn_marginal_sanity_checks(marginals):
    if marginals['threshold'][0] > .001:
        warnings.warn('psignifit:boundWarning\n'
                      'The marginal for the threshold is not near 0 at the bound.\n'
                      'This indicates that smaller Thresholds would be possible.')
    if marginals['threshold'][-1] > .001:
        warnings.warn('psignifit:boundWarning\n'
                      'The marginal for the threshold is not near 0 at the upper bound.\n'
                      'This indicates that your data is not sufficient to exclude much higher thresholds.\n'
                      'Refer to the paper or the manual for more info on this topic.')
    if marginals['width'][0] > .001:
        warnings.warn('psignifit:boundWarning\n'
                      'The marginal for the width is not near 0 at the lower bound.\n'
                      'This indicates that your data is not sufficient to exclude much lower widths.\n'
                      'Refer to the paper or the manual for more info on this topic.')
    if marginals['width'][-1] > .001:
        warnings.warn('psignifit:boundWarning\n'
                      'The marginal for the width is not near 0 at the lower bound.\n'
                      'This indicates that your data is not sufficient to exclude much higher widths.\n'
                      'Refer to the paper or the manual for more info on this topic.')


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
    grid = parameter_grid(bounds, steps_moving_bounds)
    posteriors_sparse, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # indices on the grid of the volumes that contribute more than `tol` to the overall integral
    tighter_bounds = masked_parameter_bounds(grid, posteriors_sparse >= max_bound_value)
    # do dense grid posterior_grid evaluation
    grid = parameter_grid(tighter_bounds, grid_steps)
    posteriors, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)

    fixed_param = {}
    for parm_name, parm_values in grid.items():
        if parm_values is None:
            fixed_param[parm_name] = parm_values
        elif len(parm_values) <= 1:
            fixed_param[parm_name] = parm_values[0]
    fit_dict = maximize_posterior(data, param_init=grid_max, param_fixed=fixed_param, sigmoid=sigmoid, priors=priors)

    return fit_dict, posteriors, grid
