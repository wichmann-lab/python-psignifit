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

```{warning}
This documentation page is still work in progress! Some information might be outdated.
```


# Plot Sigmoids

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
    axes2.plot(x, slope, color='C1')
    axes2.set_ylabel('slope', color='C1')
    axes2.tick_params(axis='y', labelcolor='C1')

    axes.plot(x, y, color='C0')
    axes.set_ylabel('value', color='C0')
    axes.tick_params(axis='y', labelcolor='C0')
    axes.set(title=name)
    axes.grid()


def plot_sigmoids(sigmoids_with_name, x, cols):
    rows = int(np.ceil(len(sigmoids_with_name) / cols))
    fig, axes = plt.subplots(rows, cols, constrained_layout=True)
    for (sigmoid, name), ax in zip(sigmoids_with_name, axes.ravel()):
        plot_sigmoid(sigmoid, name, x, axes=ax)
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

First, let's plot the basic sigmoids.


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
standard_sigmoids = [(sigmoid, name) for sigmoid, name in zip(unique_sigmoids, names)
                     if not sigmoid.negative and name not in LOG_SIGMOIDS]
plot_sigmoids(standard_sigmoids, STIMULUS_LEVELS, cols=3)
```

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
plot_sigmoids(log_sigmoids, STIMULUS_LEVELS, cols=2)
```

For each sigmoid, also a decreasing variant exists.



```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
neg_sigmoids = [(sigmoid, name) for sigmoid, name in zip(unique_sigmoids, names) if sigmoid.negative]
plot_sigmoids(neg_sigmoids, STIMULUS_LEVELS, cols=4)
```
