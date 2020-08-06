import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import psignifit.sigmoids as sg

COLS=4

X = np.linspace(1e-12, 1-1e-12, num=10000)
M = 0.5
WIDTH = 0.9
PC = 0.5
ALPHA = 0.05

# list of all sigmoids (after having removed aliases)
ALL_SIGS = { getattr(sg, name) for name in dir(sg) if not name.startswith('_') }

def plot_sigmoid(sigmoid, x, threshold=M, width=WIDTH, PC=PC, alpha=ALPHA, axes=None):
    y = sigmoid(x, threshold, width, PC=PC, alpha=alpha)

    if axes is None:
        fig, axes = plt.subplots()
    axes.plot(x, y)
    #axes.set(xlabel='stimulus level', ylabel='percent correct', title=sigmoid.__name__)
    axes.set(title=sigmoid.__name__)
    axes.grid()

if __name__ == '__main__':
    fig = plt.figure()
    # total number of plots
    tot_plots = len(ALL_SIGS)
    # we want 4 columns
    cols = 4
    rows = tot_plots // cols + tot_plots % cols

    for position, s in enumerate(ALL_SIGS):
        axes = fig.add_subplot(rows, cols, position+1)
        plot_sigmoid(s, X, axes=axes)
    fig.tight_layout()
    plt.show()
