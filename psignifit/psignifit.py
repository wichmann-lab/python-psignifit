# -*- coding: utf-8 -*-

import warnings
from typing import Dict, Optional

import numpy as np

from . import sigmoids
from ._confidence import confidence_intervals
from ._configuration import Configuration
from ._parameter import masked_parameter_bounds, parameter_bounds, parameter_grid
from ._posterior import marginalize_posterior, maximize_posterior, posterior_grid
from ._priors import setup_priors
from ._result import Result
from ._typing import ParameterBounds, Prior
from ._utils import PsignifitException, cast_np_scalar, check_data


def psignifit(data: np.ndarray, conf: Optional[Configuration] = None,
              debug: bool = False, **kwargs) -> Result:
    """ Fit a psychometric function to experimental data.

    This function is the main entry point for fitting psychometric functions to experimental data.

    Notice that the parameters of the psychometric function are always fit in linear space, even
    for psychometric function that are supposed to work in a logarithmic space, like the Weibull
    function. It is left to the user to transform the stimulus levels to logarithmic space before
    calling this function (see the `Parameter recovery in log-space` demo in the documentation).

    The data format must be an `n x 3` numpy array of the form:
    `[x-value, number correct, number of trials]`

    Options for the fit can be passed using a `Configuration` object and the `conf` argument, or as keyword
    arguments (more common). You can find an overview of the available options in the pages `Basic options` and
    `Advanced options` in the documentation.

    The result of this function is a `Result` object, which contains all information that `psignfit`
    produced for the fit. You can pass this object to all further processing function provided with
    `psignifit`, including the plotting functions. A description of the `Result` object is provided in the section
    `Result object` in the documentation.

    Args:
        data: Trials data as described above.
        conf: Optional configuration object.
        debug: If true, the posterior probability grid and the prior functions will be returned in the `Result`
            object in the `Result.debug` dictionary. In this mode, the `Result` object cannot be serialized.
        kwargs: Fit options as keyword arguments (optional).
    Returns:
        `Result` object with all the fitting results.
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
            # For user specified stimulus range, use conservative estimate of width_min.
            width_min = (conf.stimulus_range[1] - conf.stimulus_range[0]) / 100

    bounds = parameter_bounds(min_width=width_min, experiment_type=conf.experiment_type, stimulus_range=stimulus_range,
                              alpha=conf.width_alpha, nafc_choices=conf.experiment_choices)
    if conf.bounds is not None:
        bounds.update(conf.bounds)
    if conf.fixed_parameters is not None:
        for param, value in conf.fixed_parameters.items():
            bounds[param] = (value, value)

    priors = setup_priors(custom_priors=conf.priors, bounds=bounds,
                          stimulus_range=stimulus_range, width_min=width_min, width_alpha=conf.width_alpha,
                          beta_prior=conf.beta_prior, threshold_prop_correct=conf.thresh_PC)
    estimate_MAP_dict, estimate_mean_dict, posteriors, grid = _fit_parameters(
        data, bounds, priors, sigmoid,
        conf.steps_moving_bounds, conf.max_bound_value, conf.grid_steps,
    )

    grid_values = [grid_value for _, grid_value in sorted(grid.items())]
    intervals = confidence_intervals(posteriors, grid_values, conf.confP, conf.CI_method)
    intervals_dict = {}
    for idx, param in enumerate(sorted(grid.keys())):
        intervals_dict[param] = {k: v[idx] for k, v in intervals.items()}
    marginals = marginalize_posterior(grid, posteriors)

    if conf.verbose:
        _warn_marginal_sanity_checks(marginals)

    if conf.experiment_type == 'equal asymptote':
        estimate_MAP_dict['gamma'] = estimate_MAP_dict['lambda']
        estimate_mean_dict['gamma'] = estimate_mean_dict['lambda']
        grid['gamma'] = grid['lambda'].copy()
        priors['gamma'] = priors['lambda']
        marginals['gamma'] = marginals['lambda'].copy()
        # we may want to add a dimension to the posterior array for gamma,
        # which is a copy of the lambda dimension

    debug_dict = {}
    if debug:
        debug_dict['posteriors'] = posteriors
        debug_dict['priors'] = priors
        debug_dict['bounds'] = bounds

    return Result(parameter_estimate_MAP=estimate_MAP_dict,
                  parameter_estimate_mean=estimate_mean_dict,
                  configuration=conf,
                  confidence_intervals=intervals_dict,
                  parameter_values=grid,
                  prior_values={param: priors[param](values) for param, values in grid.items()},
                  marginal_posterior_values=marginals,
                  debug=debug_dict,
                  data=data)


def _warn_common_data_mistakes(levels, ntrials, has_user_stimulus_range, pool_max_blocks) -> None:
    """ Show warnings for common mistakes.

    Checks for too many blocks and too few trials.
    The warnings recommend to use pooling or to manually specify stimulus ranges.

    Args:
        levels: Array of stimulus level per block
        ntrials: Array of trial numbers per block
        has_user_stimulus_range: User configured the stimulus range
        pool_max_blocks: Maximum number of blocks until print of pool warning.
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
    """ Raise warnings for common issues in the marginals of the posterior distribution. """
    if 'threshold' in marginals:
        threshold_marginals = marginals['threshold'] / np.sum(marginals['threshold'])
        if threshold_marginals[0] > .001:
            warnings.warn('psignifit:boundWarning\n'
                        'The posterior marginal distribution of the parameter <threshold> can not be fully captured inside the estimation grid which is based on the prior\n'
                        'The probability at the lower bound (i.e. the lowest sampled value) is not near 0.\n'
                        'This indicates that your data is not sufficient to exclude lower thresholds than are included in the estimation space.\n'
                        'Either change the prior or ensure that the data is sufficient to constrain the posterior.\n'
                        'Refer to the paper or the manual for more info on this topic.')
        if threshold_marginals[-1] > .001:
            warnings.warn('psignifit:boundWarning\n'
                        'The posterior marginal distribution of the parameter <threshold> can not be fully captured inside the estimation grid which is based on the prior\n'
                        'The probability at the upper bound (i.e. the highest sampled value) is not near 0.\n'
                        'This indicates that your data is not sufficient to exclude higher thresholds than are included in the estimation space.\n'
                        'Either change the prior or ensure that the data is sufficient to constrain the posterior.\n'
                        'Refer to the paper or the manual for more info on this topic.')

    if 'width' in marginals:
        width_marginals = marginals['width'] / np.sum(marginals['width'])
        if width_marginals[0] > .001:
            warnings.warn('psignifit:boundWarning\n'
                        'The posterior marginal distribution of the parameter <width> can not be fully captured inside the estimation grid which is based on the prior\n'
                        'The probability at the lower bound (i.e. the lowest sampled value) is not near 0.\n'
                        'This indicates that your data is not sufficient to exclude lower widths than are included in the estimation space.\n'
                        'Either change the prior or ensure that the data is sufficient to constrain the posterior.\n'
                        'Refer to the paper or the manual for more info on this topic.')
        if width_marginals[-1] > .001:
            warnings.warn('psignifit:boundWarning\n'
                        'The posterior marginal distribution of the parameter <width> can not be fully captured inside the estimation grid which is based on the prior\n'
                        'The probability at the upper bound (i.e. the highest sampled value) is not near 0.\n'
                        'This indicates that your data is not sufficient to exclude higher widths than are included in the estimation space.\n'
                        'Either change the prior or ensure that the data is sufficient to constrain the posterior.\n'
                        'Refer to the paper or the manual for more info on this topic.')


def _fit_parameters(data: np.ndarray, bounds: ParameterBounds,
                    priors: Dict[str, Prior], sigmoid: sigmoids.Sigmoid,
                    steps_moving_bounds: Dict[str, int], max_bound_value: float,
                    grid_steps: Dict[str, int]) -> Dict[str, float]:
    """ Fit sigmoid parameters in a three-step procedure.

    1. Estimate posterior over parameters on a wider grid with coarse-grained steps (see `steps_moving_bounds`).
    2. Estimate tighter bounds of relevant probability mass (>= max_bound_values)
       and calculate the posterior over parameters there using finer-grained steps (see `grid_steps`).
    3. Fit the sigmoid parameters using the finer-grained posterior grid.

    Args:
         data: Training data.
         bounds: Dict mapping parameter names to (min, max) parameter value bounds.
         priors: Dict mapping parameter names to prior functions.
         sigmoid: `Sigmoid` class to fit.
         steps_moving_bounds: Dict mapping parameter names to the number of steps to use for the parameter grid in
            step 1. If the `bounds` for this parameter indicate that the parameter is fixed (i.e., the min bound is
            the same as the max bound), this value is not used for the corresponding parameter.
         max_bound_value: Minimum posterior probability value to consider when computing tighter bounds for the
            finer-grained grid (step 2).
         grid_steps: Dict mapping parameter names to the number of steps to use for the finer-grained parameter grid in
            step 2. If the `bounds` for this parameter indicate that the parameter is fixed (i.e., the min bound is
            the same as the max bound), this value is not used for the corresponding parameter.

    Returns:
        estimate_MAP_dict: Dict mapping parameter names to the corresponding value of the MAP point estimate.
        estimate_mean_dict: Dict mapping parameter names to the corresponding value for the mean point estimate.
        posteriors: Posterior probability over parameters, over the finer-grained parameter grid of step 2.
        grid: Dict mapping parameter names to the parameter values for the finer-grained parameter grid.
    """
    # do first sparse grid posterior_grid evaluation
    grid = parameter_grid(bounds, steps_moving_bounds)
    posteriors_sparse, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)
    # indices on the grid of the volumes that contribute more than `tol` to the overall integral
    tighter_bounds = masked_parameter_bounds(grid, posteriors_sparse >= max_bound_value)
    # do dense grid posterior_grid evaluation
    grid = parameter_grid(tighter_bounds, grid_steps)
    posteriors, grid_max = posterior_grid(data, sigmoid=sigmoid, priors=priors, grid=grid)

    # Estimate parameters as the mean of the posterior
    estimate_mean_dict = {}
    params_values = [grid[p] for p in sorted(grid.keys())]
    params_grid = np.meshgrid(*params_values, indexing='ij')
    for idx, p in enumerate(sorted(grid.keys())):
        estimate_mean_dict[p] = cast_np_scalar((params_grid[idx] * posteriors).sum())

    # Estimate parameters as the mode of the posterior (MAP)
    fixed_param = {}
    for parm_name, parm_values in grid.items():
        if len(parm_values) == 1:
            fixed_param[parm_name] = parm_values[0]
    # Compute MAP estimate of parameters on the joint posterior
    estimate_MAP_dict = maximize_posterior(data, param_init=grid_max, param_fixed=fixed_param,
                                           sigmoid=sigmoid, priors=priors)

    return estimate_MAP_dict, estimate_mean_dict, posteriors, grid
