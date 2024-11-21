---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Sigmoids

## Supported sigmoids

Out of the box psignifit supports the following sigmoid functions,
chosen by the argument `sigmoid =...`:


:::{list-table} Supported sigmoids
:widths: auto
:header-rows: 1

*   -  `sigmoid_name = ...`
	- Distribution
*   - `norm`
    - Cumulative gauss distribution. The default.
*   - `logistic`
	- Logistic function. Common alternative.
*   - `gumbel`
    - Cumulative gumbel distribution. Asymmetric, with a longer lower tail.
*   - `rgumbel`
    - Reversed gumbel distribution. Asymmetric, with a longer upper tail.
*   - `tdist`
    - Student t-distribution with df=1 for heavy tailed functions.
*   - `weibull` 
    - Weibull distribution
:::

For positive data on a log-scale, we define the 'weibull' sigmoid class. Notice that it is left
to the user to transform the stimulus level in logarithmic space, and the threshold and width
back to linear space. 'weibull' is therefore just an alias for 'gumbel'.


## Visualizing all sigmoids

In this example, we access all sigmoids that come with psignifit
and visualize their function values and slopes.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
import matplotlib.pyplot as plt
import numpy as np

from psignifit import sigmoids


LOG_SIGMOIDS = ('weibull',)
STIMULUS_LEVELS = np.linspace(1e-12, 1-1e-12, num=10000)
M = 0.5
WIDTH = 0.9
PC = 0.5
ALPHA = 0.05


def plot_sigmoid(sigmoid, name, x, threshold=M, width=WIDTH, axes=None):
    if axes is None:
        axes = plt.gca()

    y = sigmoid(x, threshold, width)
    slope = sigmoid.slope(x, threshold, width)

    axes2 = axes.twinx()
    axes2.plot(x, slope, color='C1', linestyle='--', zorder=-5)
    axes2.set_yticks([])
    
    axes.plot(x, y, color='C0')
    axes.set_ylabel('value')
    axes.set_ylim(0, 1)
    axes.set(title=name)
    axes.grid()


def plot_sigmoids(sigmoids_with_name, x):
    cols=3
    rows = int(np.ceil(len(sigmoids_with_name) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    for (sigmoid, name), ax in zip(sigmoids_with_name, axes.ravel()):
        plot_sigmoid(sigmoid, name, x, axes=ax)     
    for ax in axes.flatten():
        if not ax.has_data():
            ax.set_axis_off()
```

We select all the available sigmoids by their name.
Some of them are synonyms, so we aggregate all names.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
unique_sigmoids, names = list(), list()
for name in sigmoids.ALL_SIGMOID_NAMES:
    sigmoid = sigmoids.sigmoid_by_name(name, PC=PC, alpha=ALPHA)
    if sigmoid in unique_sigmoids:
        # synonym found
        ix = unique_sigmoids.index(sigmoid)
        names[ix] = f'{names[ix]}, {name}'
    else:
        # new sigmoid
        unique_sigmoids.append(sigmoid)
        names.append(name)

print(f"Found {len(unique_sigmoids)} sigmoids and "
      f"{len(sigmoids.ALL_SIGMOID_NAMES) - len(unique_sigmoids)} synonym names.")
```

First, let's plot the basic sigmoids. In each of the plots the sigmoid is the continuous line, and its slope the dashed line.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
standard_sigmoids = [(sigmoid, name) for sigmoid, name in zip(unique_sigmoids, names)
                     if not sigmoid.negative and name not in LOG_SIGMOIDS]
plot_sigmoids(standard_sigmoids, STIMULUS_LEVELS)
```

Looking at the slopes helps us to distinguish their different steepness. 

For Weibull functions it makes sense
that the stimulus level is on a log-scale.
Here we convert the data appropriately

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
log_sigmoids = [(sigmoid, name) for sigmoid, name in zip(unique_sigmoids, names)
                if not sigmoid.negative and name in LOG_SIGMOIDS]
plot_sigmoids(log_sigmoids, STIMULUS_LEVELS)
```

For each sigmoid, also a decreasing variant exists.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
neg_sigmoids = [(sigmoid, name) for sigmoid, name in zip(unique_sigmoids, names) if sigmoid.negative]
plot_sigmoids(neg_sigmoids, STIMULUS_LEVELS)
```
