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

# Parameter recovery demo

In this demo, we show a parameter recovery using `psignifit`. Parameter recovery is a crucial step in validating the robustness and accuracy of any model. We simulate data with known parameters ('ground truth') and then attempt to recover those parameters using psignifit. If the fitting procedure returns the same parameter values as the true parameters that generated the data, then we can be sure that the fitting works correctly. In `tests/test_param_recovery.py` we systematically run this test.

We will cover the following steps:

  1. Simulating data with known parameters.
  
  2. Fitting the model to the simulated data.
  
  3. Comparing the recovered parameters to the original parameters.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import psignifit
from psignifit import psigniplot
from psignifit.sigmoids import sigmoid_by_name
```


First, we set parameters using which the data will be simulated

```{code-cell} ipython3
# 'ground truth' values
width = 0.3
stim_range = [0.001, 0.001 + width * 1.1]
threshold = stim_range[1]/3
lambda_ = 0.05
gamma = 0.01

sigmoid_name = "norm"
sigmoid = sigmoid_by_name(sigmoid_name)

# 
nsteps = 20
num_trials = 100

stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)
```


Using the sigmoid object, we can simulate percent correct values for each stimulus level, without introducing
additional variability.

```{code-cell} ipython3
perccorr = sigmoid(stimulus_level, threshold, width, gamma, lambda_)
```


We construct our data array

```{code-cell} ipython3
ntrials = np.ones(nsteps) * num_trials
hits = (perccorr * ntrials).astype(int)
data = np.dstack([stimulus_level, hits, ntrials]).squeeze()
print(data)
```

```{code-cell} ipython3
fig, ax = plt.subplots();
ax.scatter(stimulus_level, perccorr);
ax.set_xlabel("Stimulus Level");
ax.set_ylabel("Percent Correct");
ax.spines[['right', 'top']].set_visible(False);
```


We set the options for our fit. In this case we assume a yes/no experiment and we want to estimate all parameters (i.e. fix none of them).

```{code-cell} ipython3
options = {}
options['sigmoid'] = sigmoid 
options['experiment_type'] = 'yes/no'
options['fixed_parameters'] = {}
```


Now we run the fitting procedure

```{code-cell} ipython3
res = psignifit.psignifit(data, **options)
```


Lastly, we can ensure that the values in our `res.parameter_estimate` dictionary are equal to the values that we used to simulate them

```{code-cell} ipython3
assert np.isclose(res.parameter_estimate['lambda'], lambda_, atol=1e-2)
```

```{code-cell} ipython3
assert np.isclose(res.parameter_estimate['gamma'], gamma, atol=1e-2)
```

```{code-cell} ipython3
assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-2)
assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=1e-2)
assert np.isclose(res.parameter_estimate['width'], width, atol=1e-2)
```

```{code-cell} ipython3
psigniplot.plot_psychometric_function(res);
```


# And now with some more realistic data...

```{code-cell} ipython3
eta = 0.2 # this parameter decides how overdispersed the data is
perccorr = psignifit.tools.psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_, sigmoid_name, eta)

ntrials = np.ones(nsteps) * num_trials
hits = (perccorr * ntrials).astype(int)
data = np.dstack([stimulus_level, hits, ntrials]).squeeze()
```

```{code-cell} ipython3
fig, ax = plt.subplots();
ax.scatter(stimulus_level, perccorr);
ax.set_xlabel("Stimulus Level");
ax.set_ylabel("Percent Correct");
ax.spines[['right', 'top']].set_visible(False);
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

and plot to ensure we found a good fit

```{code-cell} ipython3
psigniplot.plot_psychometric_function(res);
```
