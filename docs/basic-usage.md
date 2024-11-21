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

# Basic usage

In this guide, we show the main features of *python-psignifit*. Please
look at the [installation guide](./install_guide) for instructions 
how to install this package.

*python-psignifit* is a toolbox to fit psychometric functions. It comes
with tools to visualize and evaluate the fit.


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
import matplotlib.pyplot as plt

import psignifit as ps
from psignifit import psigniplot
```


## Trial data format

Your data for each psychometric function should be formatted as a *nx3
matrix* with columns for the stimulus level, the number of correct
responses and the number of total responses.

It should look something like this example dataset:

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: True
---
#        levels, n-correct,   n-total
data = [[0.0010,   45.0000,   90.0000],
        [0.0015,   50.0000,   90.0000],
        [0.0020,   44.0000,   90.0000],
        [0.0025,   44.0000,   90.0000],
        [0.0030,   52.0000,   90.0000],
        [0.0035,   53.0000,   90.0000],
        [0.0040,   62.0000,   90.0000],
        [0.0045,   64.0000,   90.0000],
        [0.0050,   76.0000,   90.0000],
        [0.0060,   79.0000,   90.0000],
        [0.0070,   88.0000,   90.0000],
        [0.0080,   90.0000,   90.0000],
        [0.0100,   90.0000,   90.0000]]
```

This dataset comes from a simple signal detection experiment.

## Fitting a psychometric function

A simple call to `psignifit.psignifit` 
will fit your sigmoid function to the data:


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: False
---

result = ps.psignifit(data, experiment_type='2AFC');
```

*python-psignifit* comes with presets for different psychometric
experiments. 
Apart from *nAFC* (`2AFC`, `3AFC`, ...) 
we provide two other options:  `yes/no` which enables a 
free upper and lower asymptote and,
`equal asymptote`, 
which assumes that the upper and the lower asymptote are equal. 
You find a more detailed description of the 
[experiment types here](experiment-types).

You also might want to specify the sigmoid you want to use. 
You do this by setting the paramter `sigmoid`. Default is 
the cummulative Gauss (`sigmoid=gauss').

Advanced users can pass more arguments to fine-tune the fitting procedure,
[as described here](options-dictionary) and use [different sigmoids](examples/plot_all_sigmoids)


## Getting results from the fit

The `result` is a python object with all information obtained from
fitting your data. Perhaps of primary interest are the fitted parameters
and the confidence intervals:

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: False
---
print(result.parameter_estimate)
```

This is a python dictionary containing the estimated parameters.
The parameters estimated by psignifit are:

1.  *threshold*, the stimulus value of equal-odds
2.  *width*, the difference between the 5 and the 95 percentile of the
    unscaled sigmoid
3.  *lambda*, the lapse rate (upper asymptote of the sigmoid)
4.  *gamma*, the guess rate (lower asymptote of the sigmoid). This
    parameter is fixed for nAFC experiment types.
5.  *eta*,the overdispersion parameter. A value near zero indicates your
    data behaves binomially distributed, whereas values near one
    indicate severely overdispersed data.


Then, to obtain the threhsold you run

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: False
---
print(result.parameter_estimate['threshold'])
```



Similarly, psignifit also returns the confidence intervals for 
each parameter. For example for the threshold 

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: False
---
print(result.confidence_intervals['threshold'])
```

This is a list of lists. Each element in the list contain the lower and
upper bound for the asked confidences. In this case the default returns
a list of 3 for the 95%, 90% and 68% confidence interval (in that
order). So to obtain the 95% confidence interval, you do

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: False
---
print(result.confidence_intervals['threshold'][0])
```


## Plotting the fitted function

The toolbox comes with a whole collection of visulizations. We provide
some basic plotting of the psychometric function. 


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: False
---

plt.figure()
psigniplot.plot_psychometric_function(result)
plt.show()
```

See `this user guide <plot-functions>` to 
learn more about the visualizations.


## Next steps

We covered the basic steps in using *python-psignifit*. Please refer to
the examples following this page to learn how to change the default
parameters and explore other possibilities.

The `api_ref` are helpful resources to
dive deeper.