import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from psignifit import sigmoids

COLS=4

X = np.linspace(1e-12, 1-1e-12, num=10000)
M = 0.5
WIDTH = 0.9
PC = 0.5
ALPHA = 0.05


def plot_sigmoid(sigmoid_name, x, threshold=M, width=WIDTH, axes=None):
    if axes is None:
        axes = plt.gca()

    sigmoid = sigmoids.sigmoid_by_name(sigmoid_name, PC=PC, alpha=ALPHA)
    if sigmoid.logspace:
        x = np.exp(x)
    y = sigmoid(x, threshold, width)
    slope = sigmoid.slope(x, threshold, width)

    axes.plot(x, y, color='C0')
    axes.set_ylabel('value', color='C0')
    axes.tick_params(axis='y', labelcolor='C0')

    axes2 = axes.twinx()
    axes2.plot(x, slope, color='C1')
    axes2.set_ylabel('slope', color='C1')
    axes2.tick_params(axis='y', labelcolor='C1')

    axes.set(title=sigmoid_name)
    axes.grid()


if __name__ == '__main__':
    # total number of plots
    tot_plots = len(sigmoids.ALL_SIGMOID_NAMES)
    # we want 4 columns
    cols = 4
    rows = tot_plots // cols + tot_plots % cols

    fig, axes = plt.subplots(rows, cols, constrained_layout=True)
    for sigmoid, ax in zip(sigmoids.ALL_SIGMOID_NAMES, axes.ravel()):
        plot_sigmoid(sigmoid, X, axes=ax)
    plt.show()
