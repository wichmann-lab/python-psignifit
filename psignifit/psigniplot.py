# -*- coding: utf-8 -*-
from typing import Union, List
import warnings

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import scipy.stats

from . import psignifit
from ._typing import EstimateType, ExperimentType
from ._result import Result


def plot_psychometric_function(result: Result,  # noqa: C901, this function is too complex
                               ax: matplotlib.axes.Axes = None,
                               plot_data: bool = True,
                               plot_parameter: bool = True,
                               data_color: str = '#0069AA',  # blue
                               line_color: str = '#000000',  # black
                               line_width: float = 2,
                               extrapolate_stimulus: float = 0.2,
                               x_label='Stimulus Level',
                               y_label='Proportion Correct',
                               estimate_type: EstimateType = None):
    """ Plot psychometric function fit together with the data.
    """
    if ax is None:
        ax = plt.gca()

    params = result.get_parameter_estimate(estimate_type=estimate_type)
    data = np.asarray(result.data)
    config = result.configuration

    if params['gamma'] is None:
        params['gamma'] = params['lambda']
    if len(data) == 0:
        return
    data_size = min(20, 10000. / np.sum(data[:, 2]))

    if ExperimentType.N_AFC == ExperimentType(config.experiment_type):
        ymin = 1. / config.experiment_choices
        ymin = min([ymin, min(data[:, 1] / data[:, 2])])
    else:
        ymin = 0

    x_data = data[:, 0]
    if plot_data:
        y_data = data[:, 1] / data[:, 2]
        # the size is proportional to the sqrt of the data size, as in the MATLAB version.
        # We added a factor of 10 to make visually similar to the MATLAB version
        size = np.sqrt(data_size / data[:, 2])*1000
        ax.scatter(x_data, y_data, s=size, color=data_color, marker='.', clip_on=False)

    sigmoid = config.make_sigmoid()
    x = np.linspace(x_data.min(), x_data.max(), num=1000)
    x_low = np.linspace(x[0] - extrapolate_stimulus * (x[-1] - x[0]), x[0], num=100)
    x_high = np.linspace(x[-1], x[-1] + extrapolate_stimulus * (x[-1] - x[0]), num=100)
    y = sigmoid(
        np.r_[x_low, x, x_high],
        threshold=params['threshold'],
        width=params['width'],
        gamma=params['gamma'],
        lambd=params['lambda'],
    )
    ax.plot(x, y[len(x_low):-len(x_high)], c=line_color, lw=line_width, clip_on=False)
    ax.plot(x_low, y[:len(x_low)], '--', c=line_color, lw=line_width, clip_on=False)
    ax.plot(x_high, y[-len(x_high):], '--', c=line_color, lw=line_width, clip_on=False)

    if plot_parameter:
        x = [params['threshold'], params['threshold']]
        y = [ymin, params['gamma'] + (1 - params['lambda'] - params['gamma']) * config.thresh_PC]
        ax.plot(x, y, '-', c=line_color)

        ax.axhline(y=1 - params['lambda'], linestyle=':', color=line_color)
        ax.axhline(y=params['gamma'], linestyle=':', color=line_color)

        CI_95 = result.confidence_intervals['threshold']['0.95']
        y = np.array([params['gamma'] + .5 * (1 - params['lambda'] - params['gamma'])] * 2)
        ax.plot(CI_95, y, c=line_color)
        ax.plot([CI_95[0]] * 2, y + [-.01, .01], c=line_color)
        ax.plot([CI_95[1]] * 2, y + [-.01, .01], c=line_color)

    # AXIS SETTINGS
    plt.axis('tight')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.ylim([ymin, 1])
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_stimulus_residuals(result: Result, ax: matplotlib.axes.Axes = None,
                            estimate_type: EstimateType = None) -> matplotlib.axes.Axes:
    if ax is None:
        ax = plt.gca()
    return _plot_residuals(result.data[:, 0], 'Stimulus Level', result, ax, estimate_type=estimate_type)


def plot_block_residuals(result: Result, ax: matplotlib.axes.Axes = None,
                         estimate_type: EstimateType = None) -> matplotlib.axes.Axes:
    if ax is None:
        ax = plt.gca()
    return _plot_residuals(result.data[:, 0], 'Block Number', result, ax, estimate_type=estimate_type)


def _plot_residuals(x_values: np.ndarray,
                    x_label: str,
                    result: Result,
                    ax: matplotlib.axes.Axes = None,
                    estimate_type: EstimateType = None):
    if ax is None:
        ax = plt.gca()
    params = result.get_parameter_estimate(estimate_type=estimate_type)
    data = result.data
    sigmoid = result.configuration.make_sigmoid()

    std_model = sigmoid(
        data[:, 0],
        threshold=params['threshold'],
        width=params['width'],
        gamma=params['gamma'],
        lambd=params['lambda'],
    )
    deviance = data[:, 1] / data[:, 2] - std_model
    std_model = np.sqrt(std_model * (1 - std_model))
    deviance = deviance / std_model
    x = np.linspace(x_values.min(), x_values.max(), 1000)

    ax.plot(x_values, deviance, 'k.', ms=10, clip_on=False)
    devfit = np.polyfit(x_values, deviance, 1)
    ax.plot(x, np.polyval(devfit, x), 'k-', clip_on=False)
    devfit = np.polyfit(x_values, deviance, 2)
    ax.plot(x, np.polyval(devfit, x), 'k--', clip_on=False)
    devfit = np.polyfit(x_values, deviance, 3)
    ax.plot(x, np.polyval(devfit, x), 'k:', clip_on=False)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Deviance', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_modelfit(result: Result, estimate_type: EstimateType = None) -> matplotlib.figure.Figure:
    """ Plot utilities to judge model fit.

    Plots some standard plots, meant to help you judge whether there are
    systematic deviations from the model. We dropped the statistical tests
    here though.

    The left plot shows the psychometric function with the data.

    The central plot shows the Deviance residuals against the stimulus level.
    Systematic deviations from 0 here would indicate that the measured data
    shows a different shape than the fitted one.

    The right plot shows the Deviance residuals against "time", e.g. against
    the order of the passed blocks. A trend in this plot would indicate
    learning/ changes in performance over time.

    For the central and right plot, dashes lines depict a line, quadratic and
    cubic fit; these should help in detecting systematic deviations from zero.
    """
    fig = plt.figure(figsize=(15, 5))

    ax = plt.subplot(1, 3, 1)
    plot_psychometric_function(result, ax, plot_data=True, plot_parameter=False, extrapolate_stimulus=0,
                               estimate_type=estimate_type)
    ax.set_title('Psychometric Function')

    ax = plt.subplot(1, 3, 2)
    plot_stimulus_residuals(result, ax, estimate_type=estimate_type)
    ax.set_title('Shape Check')

    ax = plt.subplot(1, 3, 3)
    plot_block_residuals(result, ax, estimate_type=estimate_type)
    ax.set_title('Time Dependence?')

    fig.tight_layout()
    return fig


def plot_bayes(result: Result) -> matplotlib.figure.Figure:
    """ Plot all posteriors.

    Plots 2D marginals for all combinations of parameters.
    """
    if result.debug=={}:
        raise ValueError("Expects posterior_mass in result, got None. You could try psignifit(debug=True).")

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    panel_indices = {'threshold': 0,
                     'width': 1,
                     'lambda': 2,
                     'gamma': 3,
                     'eta': 4}
    for yparam, i in panel_indices.items():
        for xparam, j in panel_indices.items():
            if yparam == xparam or i>j:
                continue
            try:
                plot_2D_margin(result, yparam, xparam, ax=axes[i][j-1])
            except ValueError:
                if len(result.parameter_values[xparam])==1:
                    fixedparam = xparam
                elif len(result.parameter_values[yparam])==1:
                    fixedparam = yparam
                axes[i][j-1].text(0.5, 0.5,
                                  f"Parameter {fixedparam} was\nfixed during fitting,\nthere is no data to show",
                                  ha='center')
                axes[i][j-1].axis("off")

    # hide unused axes
    axes[1][0].axis("off")
    axes[2][0].axis("off")
    axes[3][0].axis("off")
    axes[2][1].axis("off")
    axes[3][1].axis("off")
    axes[3][2].axis("off")
    plt.tight_layout()
    return fig

def plot_marginal(result: Result,
                  parameter: str,
                  ax: matplotlib.axes.Axes = None,
                  line_color: Union[str, List[float], np.ndarray] = '#0069AA',  # blue
                  line_width: float = 2,
                  y_label: str ='Marginal Density',
                  plot_prior: bool = True,
                  prior_color: Union[str, List[float], np.ndarray] = '#B2B2B2',  # light gray
                  plot_estimate: bool = True,
                  plot_ci: bool = True,
                  estimate_type: EstimateType = None):
    """ Plots the marginal for a single dimension.

    Args:
        result: should be a result object from the main psignifit routine
        parameter: The name of the parameter to plot,'threshold', 'width', 'lambda', 'gamma', 'eta'
    """
    if ax is None:
        ax = plt.gca()
    if parameter not in result.marginal_posterior_values:
        raise ValueError(f'Expects parameter {parameter} in {{{result.marginal_posterior_values.keys()}}}')

    marginal = result.marginal_posterior_values[parameter]
    if marginal is None:
        raise ValueError(f'Parameter {parameter} was fixed during optimization. No marginal to plot.')

    x_label = _parameter_label(parameter)

    x = np.asarray(result.parameter_values[parameter])
    xmin, xmax = x.min(), x.max()
    if plot_estimate:
        if plot_ci:
            # takes 95% confidence interval
            CI = result.confidence_intervals[parameter]['0.95']
            ci_x = np.r_[CI[0], x[(x >= CI[0]) & (x <= CI[1])], CI[1]]
            ax.fill_between(ci_x, np.zeros_like(ci_x), np.interp(ci_x, x, marginal), color=line_color, alpha=0.5)

        estimate = result.get_parameter_estimate(estimate_type=estimate_type)
        param_value = estimate[parameter]
        ax.plot([param_value] * 2, [0, np.interp(param_value, x, marginal)], color='#000000')

    if plot_prior and result.debug!={}:
        prior_x, prior_val = _get_prior_values(result, parameter)
        ax.plot(prior_x, prior_val, ls='--', color=prior_color)
        xmin = np.concatenate((x, prior_x)).min()
        xmax = np.concatenate((x, prior_x)).max()

    elif plot_prior and result.debug=={}:
        warnings.warn("""Cannot plot priors without debug mode. Try calling psignifit(..., debug=True)""")

    ax.plot(x, marginal, lw=line_width, c=line_color)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1.1*marginal.max())
    ax.spines[['top', 'right']].set_visible(False)

    return ax

def _get_prior_values(result, param):
    """Get the prior evaluated on the whole prior range. This is used for plotting"""

    priors_func = result.debug['priors']
    bounds = result.debug['bounds']

    prior_x = np.linspace(bounds[param][0], bounds[param][1], 1000)
    prior_vals = priors_func[param](prior_x)

    return (prior_x, prior_vals)

def _parameter_label(parameter):
    label_defaults = {'threshold': 'Threshold', 'width': 'Width',
                      'lambda': '\u03BB', 'gamma': '\u03B3', 'eta': '\u03B7'}
    return label_defaults[parameter]


def plot_prior(result: Result,
               line_color: Union[str, List[float], np.ndarray] = '#0069AA',  # blue
               line_width: float = 2,
               marker_size: float = 30,
               estimate_type: EstimateType = None):
    """ Plot the priors for the threshold, width and lambda parameters.

    The upper panels show the priors. The lower panels show a set of psychometric
    functions at selected prior values; these values are shown as markers in the
    upper row panels.

    The black function/markers indicate the value of the estimated psychometric function
    from the data. The coloured function/markers correspond to the
    0%, 25%, 75% and 100% quantiles of the prior.
    """
    if result.debug=={}:
        raise ValueError("Expects priors and marginals saved. Try running psignifit(....., debug=True).")

    fig = plt.figure(figsize=(12, 8))

    data = result.data
    bounds = result.debug['bounds']
    estimate = result.get_parameter_estimate(estimate_type=estimate_type)
    sigmoid = result.configuration.make_sigmoid()

    sigmoid_x = np.linspace(bounds['threshold'][0], bounds['threshold'][1], 1000)

    colors = ['k', [1, 200 / 255, 0], 'r', 'b', 'g']
    titles = {'threshold': 'Threshold',
              'width': 'Width',
              'lambda': '\u03BB'}

    parameter_keys = ['threshold', 'width', 'lambda']
    sigmoid_params = {param: estimate[param] for param in parameter_keys}
    for i, param in enumerate(parameter_keys):

        prior_x, prior_val = _get_prior_values(result, param)

        # Compute the CDF of the prior, to calculate the quantiles
        prior_w = prior_x[1] - prior_x[0]
        prior_cdf = np.cumsum(prior_val * prior_w)
        q25_index = np.argmax(prior_cdf > 0.25)
        q75_index = np.argmax(prior_cdf > 0.75)
        prior_mean = np.sum(prior_x * prior_val)/np.sum(prior_val)

        x_percentiles = [prior_mean,
                         min(prior_x),
                         prior_x[q25_index],
                         prior_x[q75_index],
                         max(prior_x)]

        plt.subplot(2, 3, i + 1)
        plt.plot(prior_x, prior_val, lw=line_width, c=line_color)
        # zorder: plot dots in front of the prior line
        plt.scatter(x_percentiles, np.interp(x_percentiles, prior_x, prior_val), s=marker_size, c=colors, zorder=1000)
        plt.ylabel('Density')
        plt.title(titles[param])
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.subplot(2, 3, i + 4)
        for param_value, color in zip(x_percentiles, colors):
            this_sigmoid_params = dict(sigmoid_params)
            this_sigmoid_params[param] = param_value
            y = sigmoid(
                sigmoid_x,
                threshold=this_sigmoid_params['threshold'],
                width=this_sigmoid_params['width'],
                gamma=estimate['gamma'],
                lambd=this_sigmoid_params['lambda'],
            )
            plt.plot(sigmoid_x, y, linewidth=line_width, color=color)

        plt.scatter(data[:, 0], np.zeros(data[:, 0].shape), s=marker_size*.75, c='k', clip_on=False)
        plt.xlabel('Stimulus Level')
        plt.ylabel('Proportion Correct')
        plt.xlim(min(sigmoid_x), max(sigmoid_x))
        plt.ylim(0, 1)
        plt.gca().spines[['top', 'right']].set_visible(False)


def plot_2D_margin(result: Result,
                   first_param: str,
                   second_param: str,
                   ax: matplotlib.axes.Axes = None):
    """ Constructs a 2 dimensional marginal plot of the posterior density for
    two given parameters."""
    if ax is None:
        ax = plt.gca()
    if result.debug=={}:
        raise ValueError("Expects priors and marginals saved. Try running psignifit(....., debug=True).")

    parameter_keys = result.parameter_estimate_MAP.keys()
    parameter_indices = {param: i for i, param in enumerate(sorted(parameter_keys))}
    other_param_ix = tuple(i for param, i in parameter_indices.items()
                           if param != first_param and param != second_param)
    marginal_2d = np.sum(result.debug['posteriors'], axis=other_param_ix)
    extent = [result.parameter_values[second_param][0], result.parameter_values[second_param][-1],
              result.parameter_values[first_param][-1], result.parameter_values[first_param][0]]

    if len(np.squeeze(marginal_2d).shape) != 2 or np.any(np.array(marginal_2d.shape) == 1):
        len_first = len(result.parameter_values[first_param])
        len_second = len(result.parameter_values[second_param])

        # if first_param is singleton, we copy the marginal into a matrix
        if len_first == 1 and len_second != 1:
            marginal_2d = np.broadcast_to(marginal_2d,
                                          (len(result.parameter_values[second_param]),
                                           len(result.parameter_values[second_param]))
                                          )
            extent[2] = 1  # replace range for a mockup range between 0 and 1
            extent[3] = 0

        # if second param is singleton
        elif len_first != 1 and len_second == 1:
            marginal_2d = np.broadcast_to(marginal_2d,
                                          (len(result.parameter_values[first_param]),
                                           len(result.parameter_values[first_param]))
                                          )
            extent[0] = 0
            extent[1] = 1

        # if both params are singletons, we return a matrix full of ones
        elif len_first == 1 and len_second == 1:
            marginal_2d = np.ones((len(result.parameter_values[first_param]),
                                   len(result.parameter_values[second_param]))
                                  )
            extent = [0, 1, 1, 0]

    if parameter_indices[first_param] > parameter_indices[second_param]:
        marginal_2d = np.transpose(marginal_2d)

    ax.imshow(marginal_2d, extent=extent, cmap='Reds_r',  aspect='auto')
    ax.set_xlabel(_parameter_label(second_param))
    ax.set_ylabel(_parameter_label(first_param))


def plot_bias_analysis(data: np.ndarray, compare_data: np.ndarray,
                       estimate_type: EstimateType = None, **kwargs) -> None:
    """ Analyse and plot 2-AFC dataset bias.

    This analysis is used to see whether two 2AFC datasets have a bias and
    whether it can be explained with a "finger bias" (a bias in guessing).

    It runs psignifit on the datasets `data`, `compare_data`, and
    their combination. Then the corresponding psychometric functions and marginal
    posterior distributions in 1, 2, and 3 dimensions are plotted.

    Args:
         data: First dataset as expected by :func:`psignifit.psignifit`.
         compare_data: Second dataset for :func:`psignifit.psignifit`.
         kwargs: Additional configuration arguments for :func:`psignifit.psignifit`.
    """
    config = dict(experiment_type=ExperimentType.YES_NO.value,
                  bounds={'lambda': [0, .1],
                          'gamma': [.11, .89]},
                  fixed_parameters={'eta': 0},
                  grid_steps={'threshold': 40,
                              'width': 40,
                              'lambda': 40,
                              'gamma': 40,
                              'eta': 1},
                  steps_moving_bounds={'threshold': 30,
                                       'width': 30,
                                       'lambda': 20,
                                       'gamma': 20,
                                       'eta': 1},
                  priors={'gamma': lambda x: scipy.stats.beta.pdf(x, 2, 2)},
                  pool_max_blocks=30,
                  debug=True,
                  **kwargs)
    result_combined = psignifit(np.r_[data, compare_data], **config)
    result_data = psignifit(data, **config)
    result_compare_data = psignifit(compare_data, **config)

    fig = plt.figure(constrained_layout=True, figsize=(5, 15))
    gs = fig.add_gridspec(6, 1)

    ax1 = fig.add_subplot(gs[0:2, 0])
    plot_psychometric_function(result_combined, ax=ax1, estimate_type=estimate_type)
    plot_psychometric_function(result_data, ax=ax1, line_color=[1, 0, 0], data_color=[1, 0, 0],
                               estimate_type=estimate_type)
    plot_psychometric_function(result_compare_data, ax=ax1, line_color=[0, 0, 1], data_color=[0, 0, 1],
                               estimate_type=estimate_type)
    ax1.set_ylim((0, 1))
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[4, 0])
    ax5 = fig.add_subplot(gs[5, 0])

    axesmarginals = [ax2, ax3, ax4, ax5]

    for param, ax in zip(['threshold', 'width', 'lambda', 'gamma'], axesmarginals):

        plot_marginal(result_combined, param, ax=ax, plot_prior=False,
                      line_color=[0, 0, 0], estimate_type=estimate_type,
                      plot_ci=False)

        plot_marginal(result_data, param, ax=ax, plot_prior=False,
                      line_color=[1, 0, 0], estimate_type=estimate_type,
                      plot_ci=False)


        plot_marginal(result_compare_data, param, ax=ax, plot_prior=False,
                      line_color=[0, 0, 1], estimate_type=estimate_type,
                      plot_ci=False)

    for ax in axesmarginals:
        ax.autoscale()


