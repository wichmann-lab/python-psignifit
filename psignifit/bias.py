import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from .typing import ExperimentType
from .psignifit import psignifit
from .psigniplot import plotPsych
from .psigniplot import plotMarginal


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

    result_combined = psignifit(np.append(data, compare_data, axis=0), **config)
    result_data = psignifit(data, **config)
    result_compare_data = psignifit(compare_data, **config)

    plt.figure()
    a1 = plt.axes([0.15, 4.35 / 6, 0.75, 1.5 / 6])

    plotPsych(result_combined, showImediate=False)
    plt.hold(True)

    plotPsych(result_data,
              lineColor=[1, 0, 0],
              dataColor=[1, 0, 0],
              showImediate=False)
    plotPsych(result_compare_data,
              lineColor=[0, 0, 1],
              dataColor=[0, 0, 1],
              showImediate=False)
    plt.ylim([0, 1])

    a2 = plt.axes([0.15, 3.35 / 6, 0.75, 0.5 / 6])

    plotMarginal(result_combined,
                 dim=0,
                 prior=False,
                 CIpatch=False,
                 lineColor=[0, 0, 0],
                 showImediate=False)
    plt.hold(True)

    plotMarginal(result_data, dim=0, lineColor=[1, 0, 0], showImediate=False)
    plotMarginal(result_compare_data, dim=0, lineColor=[0, 0, 1], showImediate=False)
    a2.relim()
    a2.autoscale_view()

    a3 = plt.axes([0.15, 2.35 / 6, 0.75, 0.5 / 6])
    plotMarginal(result_combined,
                 dim=1,
                 prior=False,
                 CIpatch=False,
                 lineColor=[0, 0, 0],
                 showImediate=False)
    plt.hold(True)

    plotMarginal(result_data, dim=1, lineColor=[1, 0, 0], showImediate=False)
    plotMarginal(result_compare_data, dim=1, lineColor=[0, 0, 1], showImediate=False)
    a3.relim()
    a3.autoscale_view()

    a4 = plt.axes([0.15, 1.35 / 6, 0.75, 0.5 / 6])

    plotMarginal(result_combined,
                 dim=2,
                 prior=False,
                 CIpatch=False,
                 lineColor=[0, 0, 0],
                 showImediate=False)
    plt.hold(True)

    plotMarginal(result_data, dim=2, lineColor=[1, 0, 0], showImediate=False)
    plotMarginal(result_compare_data, dim=2, lineColor=[0, 0, 1], showImediate=False)
    a4.relim()
    a4.autoscale_view()

    a5 = plt.axes([0.15, 0.35 / 6, 0.75, 0.5 / 6])

    plotMarginal(result_combined,
                 dim=3,
                 prior=False,
                 CIpatch=False,
                 lineColor=[0, 0, 0],
                 showImediate=False)
    plt.hold(True)

    plotMarginal(result_data, dim=3, lineColor=[1, 0, 0], showImediate=False)
    plotMarginal(result_compare_data, dim=3, lineColor=[0, 0, 1], showImediate=False)
    a5.set_xlim([0, 1])
    a5.relim()
    a5.autoscale_view()

    plt.draw()