---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Plotting

Here we explain the basic plot functions which come with psignifit.
All these functions are located in the submodule `psigniplot`.

Most of the functions take an matplotlib axes handle, and 
return it. This enables you to personalize where (in which figure
or subplot) you wish to plot. 
After the function call the returned axes handle also allows
you to further tweak the plot. 


We first use the same data as in the [basic usage example](../basic-usage).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import psignifit as ps
from psignifit import psigniplot

data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC')
```

## Plot psychometric function

This function plots the fitted psychometric function with the measured data.
It takes as input the result object you want to plot.

```{code-cell} ipython3
plt.figure()
ps.psigniplot.plot_psychometric_function(res)
```

The function also take the following arguments, here we print the defaults

```
plot_data: True
plot_parameter: True
data_color: '#0069AA'  # blue
line_color: '#000000'  # black
line_width: 2
extrapolate_stimulus: 0.2
x_label='Stimulus Level'
y_label='Proportion Correct'
```                               

For example, to increase the linewidth and change it to red, you could pass

```{code-cell} ipython3
plt.figure()
psigniplot.plot_psychometric_function(res, line_width=5, line_color='r')
```

## Plotting more than one function

The mentioned above the function also takes a matplotlib axes handle,
which you can use to plot in a particular subplot.
In the following example we create an artificial second dataset; this 
is just a shifted version of the first one

```{code-cell} ipython3
# copy the data and introduce a shift in all stimulus values
otherdata = np.copy(data)
otherdata[:, 0] = otherdata[:, 0] + 0.01

# fit with exact same options
other_res = ps.psignifit(otherdata, res.configuration)
```

Now we plot both datasets and fitted functions in a single plot

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1)
psigniplot.plot_psychometric_function(res, ax=ax)
psigniplot.plot_psychometric_function(other_res, ax=ax, line_color='r', data_color='r')
plt.show()
```

Alternatively we can plot them in different subplots

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
psigniplot.plot_psychometric_function(res, ax=axes[0])
psigniplot.plot_psychometric_function(other_res, ax=axes[1], line_color='r', data_color='r')
plt.show()
```



## Plot model fit

We offer you a function which creates the plots psignifit 2 created for
checking the modelfit.
It can be run with the following command follows:


```{code-cell} ipython3
fig = psigniplot.plot_modelfit(res)
```

This method will show you three plots, based on the deviance residuals,
which are the normalized deviations from the fitted psychometric function:

- (left) the psychometric function with the data around it as a first general
check.

- (middle) deviance residuals against the stimulus level. 
This is a check whether the data systematically lie above or below the 
psychometric function at a range of stimulus levels. 
The three lines are polinomials of first
second and third order fitted to the points. If these dots deviate
strongly and/or systematically from 0, this is worrysome. Such deviations
indicate that the shape of the psychometric function fitted does not
match the data.

- (right) deviance residuals against the block order. 
This plot is meant as a help to detect, when performance changes over time. 
Assuming that your blocks somewhat reflect the order in which the data were collected, this shows
you how the deviations from the psychometric function changed over time.
Again strong and/or systematic deviations from 0 are a cause for worry.


## Plot marginal posterior

This function plots the marginal posterior density for a single parameter.
As input it requires a results object and as a second parameter
which parameter to plot, as a string 
('threshold', 'width', 'lambda', 'gamma' or 'eta')

```{code-cell} ipython3
plt.figure()
psigniplot.plot_marginal(res, 'threshold')
```

The blue shadow corresponds to the chosen confidence interval and the black
line shows the point estimate for the plotted parameter.

When the fit is done in debug mode (`debug=True` as option),
then the plot also includes the prior as a gray dashed line.

```{code-cell} ipython3
# we refit in debug mode
res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC', debug=True)

plt.figure()
psigniplot.plot_marginal(res, 'threshold')
```

The following list shows the options for this plot and the defaults

```
line_color: '#0069AA',  # blue
line_width: 2
y_label: 'Marginal Density'
plot_prior: True
prior_color: '#B2B2B2'  # light gray
plot_estimate: True
```

## Plot 2-D posterior

This plots 2 dimensional posterior marginals.
As input this function expects the result object and the two parameters
to plot against each other.

```{code-cell} ipython3
plt.figure()
ps.psigniplot.plot_2D_margin(res, 'threshold', 'width')
```

## Plot all 2D posteriors

Same type of plot as the above but now for all parameter combinations.

```{code-cell} ipython3
fig = ps.psigniplot.plot_bayes(res)
```

## Plot samples from the posterior over psychometric functions

To get an idea of the range of  psychometric functions that are compatible with the data, it's possible to plot 
samples from the posterior over  psychometric functions. A number of samples are plot semi-transparently over one 
another, creating a shaded area around the point estimate of the psychometric function.

```{code-cell} ipython3
plt.figure()

n_samples = 100
random_state = np.random.RandomState(7474)

psigniplot.plot_posterior_samples(result=res, n_samples=n_samples, samples_alpha=0.1, 
                                  samples_color='k', random_state=random_state)
```


## Plot priors
As a tool this function plots the actually used priors of the provided
result dictionary.

```{code-cell} ipython3
plt.figure()
ps.psigniplot.plot_prior(res)
```

```{code-cell} ipython3

```
