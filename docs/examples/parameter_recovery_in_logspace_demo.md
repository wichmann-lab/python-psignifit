---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernel_info:
  name: datana
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Parameter Recovery in Log-Space Demo

In this demo, we show a parameter recovery using `psignifit`, in the case where we want to fit a sigmoid in log space. 

This demo is very similar to the "Parameter Recovery Demo", and you should have a look at that one first, but it also shows how to fit the sigmoid in log-space, and how to recover width, threshold, and confidence intercals in the original simulus space.

We will cover the following steps:

  1. Simulating data with known parameters.
  
  2. Fitting the model to the simulated data in logspace.

  3. Converting the fitted parameters back to stimulus space.
  
  4. Comparing the recovered parameters to the original parameters.


```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt

import psignifit
from psignifit import psigniplot
from psignifit.sigmoids import sigmoid_by_name
```

First, we set parameters using which the data will be simulated

```{code-cell} ipython3
stim_range = [1.0, 1000.0]
threshold = 134

lambda_ = 0.0232
gamma = 0.1
nsteps = 20
num_trials = 50000
# We are going to fit a log-Weibull sigmoid
sigmoid_name = 'weibull'
sigmoid = sigmoid_by_name('weibull')
# We choose levels spaced logarithmically in the stimuli space, because that is often what happens
# in experiments where the fit is done in logarithmic space
stimulus_level = np.logspace(np.log10(stim_range[0]), np.log10(stim_range[1]), nsteps, base=10)
# However, one could just as well space the stimuli linearly
#stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)
```

Using the sigmoid object, we can simulate percent correct values for each stimulus level, without introducing 
additional variability.

We generate the data on the data and parameters transformed in log-space, since the fit is going to be made there.

```{code-cell} ipython3
logspace_stimulus_level = np.log(stimulus_level)
log_threshold = np.log(threshold)
log_width = 2.3  # Width in log-space

perccorr = sigmoid(logspace_stimulus_level, log_threshold, log_width, gamma, lambda_)
```

This is what the generated data and threshold look like in the stimulus space

```{code-cell} ipython3
fig, ax = plt.subplots();
ax.semilogx(stimulus_level, perccorr, marker='o');
ax.set_xlabel("Stimulus Level");
ax.set_ylabel("Percent Correct");
ax.axvline(threshold, color='r');
pc = (1 - gamma - lambda_) / 2 + gamma
ax.axhline(pc, color='r');
```

We construct our data array. In this first case, we have so many trials that the average hit rate is identical to the value on the psychometric function.

Notice that we are going to give to `psignifit` the data in the log-space, so that the fit happens in that space.

```{code-cell} ipython3
ntrials = np.ones(nsteps) * num_trials
hits = (perccorr * ntrials).astype(int)

data = np.dstack([logspace_stimulus_level, hits, ntrials]).squeeze()
```

We set the options for our fit. In this case we assume a yes/no experiment and we want to estimate all parameters (i.e. fix none).

```{code-cell} ipython3
options = {}
options['sigmoid'] = sigmoid 
options['experiment_type'] = 'yes/no'
options['fixed_parameters'] = {}
```

Now we run the fitting procedure. We don't need to do anything special about the log-space: since the data lives in that space, the fit is going to happen in that space.

```{code-cell} ipython3
res = psignifit.psignifit(data, **options)
```

Lastly, we can ensure that the values in our `res.parameter_estimate` dictionary are equal to the values that we used to simulate them. Notice that, since the fit is done in log-space, the parameters that correspond to the stimulus space (threshold and width) are themselves to be interpreted in log-space.

```{code-cell} ipython3
res.parameter_estimate
```

```{code-cell} ipython3
assert np.isclose(res.parameter_estimate['threshold'], log_threshold, atol=1e-4)
assert np.isclose(res.parameter_estimate['width'], log_width, atol=1e-4)
```

```{code-cell} ipython3
assert np.isclose(res.parameter_estimate['lambda'], lambda_, atol=1e-4)
assert np.isclose(res.parameter_estimate['gamma'], gamma, atol=1e-4)
assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-4)
```

```{code-cell} ipython3
fig, ax = plt.subplots();
psigniplot.plot_psychometric_function(res, ax=ax);
ax.scatter(logspace_stimulus_level, perccorr);
ax.set_xlabel('log(Stimulus Level)');
```

If we want to interpret the parameters in the original stimulus space, we need a minimum of math:
- The threshold is just a stimulus level in log-space, and can be transformed back to stimulus space by taking its exponential
- The width is a _distance_ between stimulus levels in log-space, and cannot be tranformed back that simply. Instead, we use the definition of width: we measure the log-levels of `alpha` and `1-alpha`, transform them back to stimulus space, and then compute the width.

```{code-cell} ipython3
lin_estimated_threshold = np.exp(res.parameter_estimate['threshold'])
print('Actual threshold in stimulus space:', threshold)
print('Estimated threshold in stimlulus space:', lin_estimated_threshold)
```

```{code-cell} ipython3
lin_estimated_width = (
    np.exp(res.threshold(0.95, return_ci=False, unscaled=True)) 
    - np.exp(res.threshold(0.05, return_ci=False, unscaled=True))
)
print('Estimated width in stimlulus space:', lin_estimated_width)
```
# And now with some more realistic data...

```{code-cell} ipython3
num_trials = 50  # A smaller number of trials makes the data more noisy
eta = 0.1  # This parameter decides how overdispersed the data is
perccorr = psignifit.tools.psychometric_with_eta(logspace_stimulus_level, log_threshold, log_width, gamma, lambda_, sigmoid_name, eta)

ntrials = np.ones(nsteps, dtype=int) * num_trials
hits = np.random.binomial(ntrials, perccorr)
data = np.dstack([logspace_stimulus_level, hits, ntrials]).squeeze()
```

```{code-cell} ipython3
experimental_perccorr = hits/ntrials
fig, ax = plt.subplots();
ax.semilogx(stimulus_level, experimental_perccorr, marker='o', ls='');
ax.set_xlabel("Stimulus Level");
ax.set_ylabel("Percent Correct");
```

We run the fit again

```{code-cell} ipython3
options = {}
options['sigmoid'] = sigmoid 
options['experiment_type'] = 'yes/no'
options['fixed_parameters'] = {}

```

```{code-cell} ipython3
res = psignifit.psignifit(data, **options)
```

Plot to ensure we found a good fit


```{code-cell} ipython3
res.parameter_estimate
```

```{code-cell} ipython3
fig, ax = plt.subplots();
psigniplot.plot_psychometric_function(res, ax=ax);
```

```{code-cell} ipython3
lin_threshold = np.exp(res.parameter_estimate['threshold'])
lin_width = np.exp(res.threshold(0.95, return_ci=False, unscaled=True)) - np.exp(res.threshold(0.05, return_ci=False, unscaled=True))
lin_threshold, lin_width
```
