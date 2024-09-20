# -*- coding: utf-8 -*-
from typing import Union, List

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import scipy.stats
from scipy.signal import convolve

from . import psignifit
from ._typing import ExperimentType
from ._result import Result

def plot_psychometric_function(result: Result,  # noqa: C901, this function is too complex
                               ax: matplotlib.axes.Axes = None,
                               plot_data: bool = True,
                               plot_parameter: bool = True,
                               data_color: Union[str, List[float], np.ndarray] = '#0069AA',  # blue
                               line_color: Union[str, List[float], np.ndarray] = '#000000',  # black
                               line_width: float = 2,
                               extrapolate_stimulus: float = 0.2,
                               x_label='Stimulus Level',
                               y_label='Proportion Correct'):
    """ Plot oted psychometric function with the data.
    """
    if ax is None:
        ax = plt.gca()

    params = result.parameter_estimate
    data = np.asarray(result.data)
    config = result.configuration

    if params['gamma'] is None:
        params['gamma'] = params['lambda']
    if len(data) == 0:
        return
    data_size = 10000. / np.sum(data[:, 2])

    if ExperimentType.N_AFC == ExperimentType(config.experiment_type):
        ymin = 1. / config.experiment_choices
        ymin = min([ymin, min(data[:, 1] / data[:, 2])])
    else:
        ymin = 0

    x_data = data[:, 0]
    if plot_data:
        y_data = data[:, 1] / data[:, 2]
        size = np.sqrt(data_size / 2 * data[:, 2])
        ax.scatter(x_data, y_data, s=size, c=data_color, marker='.', clip_on=False)

    sigmoid = config.make_sigmoid()
    x = np.linspace(x_data.min(), x_data.max(), num=1000)
    x_low = np.linspace(x[0] - extrapolate_stimulus * (x[-1] - x[0]), x[0], num=100)
    x_high = np.linspace(x[-1], x[-1] + extrapolate_stimulus * (x[-1] - x[0]), num=100)
    y = sigmoid(np.r_[x_low, x, x_high], params['threshold'], params['width'])
    y = (1 - params['gamma'] - params['lambda']) * y + params['gamma']
    ax.plot(x, y[len(x_low):-len(x_high)], c=line_color, lw=line_width, clip_on=False)
    ax.plot(x_low, y[:len(x_low)], '--', c=line_color, lw=line_width, clip_on=False)
    ax.plot(x_high, y[-len(x_high):], '--', c=line_color, lw=line_width, clip_on=False)

    if plot_parameter:
        x = [params['threshold'], params['threshold']]
        y = [ymin, params['gamma'] + (1 - params['lambda'] - params['gamma']) * config.thresh_PC]
        ax.plot(x, y, '-', c=line_color)

        ax.axhline(y=1 - params['lambda'], linestyle=':', color=line_color)
        ax.axhline(y=1 - params['gamma'], linestyle=':', color=line_color)

        CI = np.asarray(result.confidence_intervals['threshold'])
        y = np.array([params['gamma'] + .5 * (1 - params['lambda'] - params['gamma'])] * 2)
        ax.plot(CI[0, :], y, c=line_color)
        ax.plot([CI[0, 0]] * 2, y + [-.01, .01], c=line_color)
        ax.plot([CI[0, 1]] * 2, y + [-.01, .01], c=line_color)

    # AXIS SETTINGS
    plt.axis('tight')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([ymin, 1])
    return ax


def plot_stimulus_residuals(result: Result, ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
    if ax is None:
        ax = plt.gca()
    return _plot_residuals(result.data[:, 0], 'Stimulus Level', result, ax)


def plot_block_residuals(result: Result, ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
    if ax is None:
        ax = plt.gca()
    return _plot_residuals(range(result.data.shape[0]), 'Block Number', result, ax)


def _plot_residuals(x_values: np.ndarray, x_label: str, result: Result, ax: matplotlib.axes.Axes = None):
    if ax is None:
        ax = plt.gca()
    params = result.parameter_estimate
    data = result.data
    sigmoid = result.configuration.make_sigmoid()

    std_model = params['gamma'] + (1 - params['lambda'] - params['gamma']) * sigmoid(
        data[:, 0], params['threshold'], params['width'])
    deviance = data[:, 1] / data[:, 2] - std_model
    std_model = np.sqrt(std_model * (1 - std_model))
    deviance = deviance / std_model
    x = np.linspace(x_values.min(), x_values.max(), 1000)

    ax.plot(x_values, deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(x_values, deviance, 1)
    ax.plot(x, np.polyval(linefit, x), 'k-', clip_on=False)
    linefit = np.polyfit(x_values, deviance, 2)
    ax.plot(x, np.polyval(linefit, x), 'k--', clip_on=False)
    linefit = np.polyfit(x_values, deviance, 3)
    ax.plot(x, np.polyval(linefit, x), 'k:', clip_on=False)

    ax.xlabel(x_label, fontsize=14)
    ax.ylabel('Deviance', fontsize=14)
    return ax


def plot_modelfit(result: Result) -> matplotlib.figure.Figure:
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
    """
    fig = plt.figure(figsize=(15, 5))

    ax = plt.subplot(1, 3, 1)
    plot_psychometric_function(result, ax, plot_data=True, plot_parameter=False, extrapolate_stimulus=0)
    ax.set_title('Psychometric Function')

    ax = plt.subplot(1, 3, 2)
    plot_stimulus_residuals(result, ax)
    ax.set_title('Shape Check')

    ax = plt.subplot(1, 3, 3)
    plot_block_residuals(result, ax)
    ax.set_title('Time Dependence?')

    fig.tight_layout()
    return fig


def plot_marginal(result: Result,
                  parameter: str,
                  ax: matplotlib.axes.Axes = None,
                  line_color: Union[str, List[float], np.ndarray] = '#0069AA',  # blue
                  line_width: float = 2,
                  y_label: str ='Marginal Density',
                  plot_prior: bool = True,
                  prior_color: Union[str, List[float], np.ndarray] = '#B2B2B2',  # light gray
                  plot_estimate: bool = True):
    """ Plots the marginal for a single dimension.

    Args:
        result: should be a result struct from the main psignifit routine
        dim: The parameter to plot. 1=threshold, 2=width, 3=lambda, 4=gamma, 5=sigma
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
    if plot_estimate:
        CIs = np.asarray(result.confidence_intervals[parameter])
        for CI in CIs:
            ci_x = np.r_[CI[0], x[(x >= CI[0]) & (x <= CI[1])], CI[1]]
            ax.fill_between(ci_x, np.zeros_like(ci_x), np.interp(ci_x, x, marginal), color=line_color, alpha=0.5)

        param_value = result.parameter_estimate[parameter]
        ax.plot([param_value] * 2, [0, np.interp(param_value, x, marginal)], color=line_color)

    if plot_prior:
        ax.plot(x, result.prior_values[parameter], ls='--', color=prior_color, clip_on=False)

    ax.plot(x, marginal, lw=line_width, c=line_color, clip_on=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax


def _parameter_label(parameter):
    label_defaults = {'threshold': 'Threshold', 'width': 'Width',
                      'lambda': '\u03BB', 'gamma': '\u03B3', 'eta': '\u03B7'}
    return label_defaults[parameter]


def plot_prior(result: Result,
               line_color: Union[str, List[float], np.ndarray] = '#0069AA',  # blue
               line_width: float = 2,
               marker_size: float = 30):
    """ Plot the priors on the different parameters.

    The coloured psychometric functions correspond to the 0%, 25%, 75% and 100%
    quantiles of the prior.
    """
    data = result.data
    params = result.parameter_values
    priors = result.prior_values
    sigmoid = result.configuration.make_sigmoid()

    colors = ['k', [1, 200 / 255, 0], 'r', 'b', 'g']
    stimulus_range = result.configuration.stimulus_range
    if stimulus_range is None:
        stimulus_range = [data[:, 0].min(), data[:, 0].max()]
    width = stimulus_range[1] - stimulus_range[0]
    stimulus_range = [stimulus_range[0] - .5 * width , stimulus_range[1] + .5 * width]

    titles = {'threshold': 'Threshold m',
              'width': 'Width w',
              'lambda': r'Lapse Rate $\lambda$'}

    parameter_keys = ['threshold', 'width', 'lambda']
    sigmoid_x = np.linspace(stimulus_range[0], stimulus_range[1], 10000)
    sigmoid_params = {param: result.parameter_estimate[param] for param in parameter_keys}
    for i, param in enumerate(parameter_keys):
        prior_x = params[param]
        weights = convolve(np.diff(prior_x), np.array([0.5, 0.5]))
        cumprior = np.cumsum(priors[param] * weights)
        x_percentiles = [result.parameter_estimate[param], min(prior_x), prior_x[-cumprior[cumprior >= .25].size],
                         prior_x[-cumprior[cumprior >= .75].size], max(prior_x)]
        plt.subplot(2, 3, i + 1)
        plt.plot(params[param], priors[param], lw=line_width, c=line_color)
        plt.scatter(x_percentiles, np.interp(x_percentiles, prior_x, priors[param]), ms=marker_size, c=colors)
        plt.xlabel('Stimulus Level')
        plt.ylabel('Density')
        plt.title(titles[param])

        plt.subplot(2, 3, i + 4)
        for param_value, color in zip(x_percentiles, colors):
            this_sigmoid_params = dict(sigmoid_params)
            this_sigmoid_params[param] = param_value
            plt.plot(sigmoid_x, sigmoid(sigmoid_x, **this_sigmoid_params), line_width=line_width, color=color)
        plt.plot(data[:, 0], np.zeros(data[:, 0].shape), 'k.', ms=marker_size * .75)
        plt.xlabel('Stimulus Level')
        plt.ylabel('Percent Correct')


def plot_2D_margin(result: Result,
                   first_param: str,
                   second_param: str,
                   ax: matplotlib.axes.Axes = None):
    """ Constructs a 2 dimensional marginal plot of the posterior density. """
    if ax is None:
        ax = plt.gca()
    if result.posterior_mass is None:
        ValueError("Expects posterior_mass in result, got None. You could try psignifit(return_posterior=True).")

    parameter_indices = {param: i for i, param in enumerate(sorted(result.parameter_estimate.keys()))}
    other_param_ix = tuple(i for param, i in parameter_indices.items()
                           if param != first_param and param != second_param)
    marginal_2d = np.sum(result.posterior_mass, axis=other_param_ix)
    if len(marginal_2d.shape) != 2 or np.any(marginal_2d.shape == 1):
        raise ValueError(f'The marginal is not two-dimensional. Were the parameters fixed during optimization?')

    if parameter_indices[first_param] > parameter_indices[second_param]:
        (second_param, first_param) = (first_param, second_param)
    extent = [result.parameter_values[second_param][0], result.parameter_values[second_param][-1],
              result.parameter_values[first_param][0], result.parameter_values[first_param][-1]]
    ax.imshow(marginal_2d, extent=extent)
    ax.set_xlabel(_parameter_label(second_param))
    ax.set_ylabel(_parameter_label(first_param))


def plot_bias_analysis(data: np.ndarray, compare_data: np.ndarray, **kwargs) -> None:
    """ Analyse and plot 2-AFC dataset bias.

    This short analysis is used to see whether two 2AFC datasets have a bias and
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
                  **kwargs)
    result_combined = psignifit(np.r_[data, compare_data], **config)
    result_data = psignifit(data, **config)
    result_compare_data = psignifit(compare_data, **config)

    plt.figure()
    ax = plt.axes([0.15, 4.35 / 6, 0.75, 1.5 / 6])

    plot_psychometric_function(result_combined, ax=ax)
    plot_psychometric_function(result_data, ax=ax, line_color=[1, 0, 0], data_color=[1, 0, 0])
    plot_psychometric_function(result_compare_data, ax=ax, line_color=[0, 0, 1], data_color=[0, 0, 1])
    plt.ylim([0, 1])

    for param in ['threshold', 'width', 'lambda', 'gamma']:
        ax = plt.axes([0.15, 3.35 / 6, 0.75, 0.5 / 6])
        plot_marginal(result_combined, param, ax=ax, plot_prior=False, line_color=[0, 0, 0])

        plot_marginal(result_data, param, ax=ax, line_color=[1, 0, 0])
        plot_marginal(result_compare_data, param, ax=ax, line_color=[0, 0, 1])
        ax.relim()
        ax.autoscale_view()
