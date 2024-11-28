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

+++ {"nteract": {"transient": {"deleting": false}}}

# Parameter recovery demo

In this demo, we show a parameter recovery using `psignifit`. Parameter recovery is a crucial step in validating the robustness and accuracy of any model. We simulate data with known parameters ('ground truth') and then attempt to recover those parameters using psignifit. If the fitting procedure returns the same parameter values as the true parameters that generated the data, then we can be sure that the fitting works correctly. In `tests/test_param_recovery.py` we systematically run this test.

We will cover the following steps:

  1. Simulating data with known parameters.
  
  2. Fitting the model to the simulated data.
  
  3. Comparing the recovered parameters to the original parameters.

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
import numpy as np
import matplotlib.pyplot as plt

import psignifit
from psignifit import psigniplot
```

+++ {"nteract": {"transient": {"deleting": false}}}

First, we set parameters using which the data will be simulated

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
# 'ground truth' values
width = 0.3
stim_range = [0.001, 0.001 + width * 1.1]
threshold = stim_range[1]/3
lambda_ = 0.05
gamma = 0.01

sigmoid = "norm"

# 
nsteps = 20
num_trials = 100

stimulus_level = np.linspace(stim_range[0], stim_range[1], nsteps)
```

+++ {"nteract": {"transient": {"deleting": false}}}

Using the `tools.psychometric` we can simulate percent correct values for each stimulus level, without introducing additional variability.

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
perccorr = psignifit.tools.psychometric(stimulus_level, threshold, width, gamma, lambda_, sigmoid)
```

+++ {"nteract": {"transient": {"deleting": false}}}

We construct our data array

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
ntrials = np.ones(nsteps) * num_trials
hits = (perccorr * ntrials).astype(int)
data = np.dstack([stimulus_level, hits, ntrials]).squeeze()
print(data)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
  source_hidden: false
nteract:
  transient:
    deleting: false
---
fig, ax = plt.subplots()
ax.scatter(stimulus_level, perccorr)
ax.set_xlabel("Stimulus Level")
ax.set_ylabel("Percent Correct")
ax.spines[['right', 'top']].set_visible(False)
```

+++ {"nteract": {"transient": {"deleting": false}}}

We set the options for our fit. In this case we assume a yes/no experiment and we want to estimate all parameters (i.e. fix none of them).

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
options = {}
options['sigmoid'] = sigmoid 
options['experiment_type'] = 'yes/no'
options['fixed_parameters'] = {}
```

+++ {"nteract": {"transient": {"deleting": false}}}

Now we run the fitting procedure

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
res = psignifit.psignifit(data, **options)
```

+++ {"nteract": {"transient": {"deleting": false}}}

Lastly, we can ensure that the values in our `res.parameter_estimate` dictionary are equal to the values that we used to simulate them

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
assert np.isclose(res.parameter_estimate['lambda'], lambda_, atol=1e-2)
```

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
assert np.isclose(res.parameter_estimate['gamma'], gamma, atol=1e-2)
```

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
assert np.isclose(res.parameter_estimate['eta'], 0, atol=1e-2)
assert np.isclose(res.parameter_estimate['threshold'], threshold, atol=1e-2)
assert np.isclose(res.parameter_estimate['width'], width, atol=1e-2)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
  source_hidden: false
nteract:
  transient:
    deleting: false
---
psigniplot.plot_psychometric_function(res)
```

+++ {"nteract": {"transient": {"deleting": false}}}

# And now with some more realistic data...

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
eta = 0.2 # this parameter decides how overdispersed the data is
perccorr = psignifit.tools.psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_, sigmoid, eta)

ntrials = np.ones(nsteps) * num_trials
hits = (perccorr * ntrials).astype(int)
data = np.dstack([stimulus_level, hits, ntrials]).squeeze()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
  source_hidden: false
nteract:
  transient:
    deleting: false
---
fig, ax = plt.subplots()
ax.scatter(stimulus_level, perccorr)
ax.set_xlabel("Stimulus Level")
ax.set_ylabel("Percent Correct")
ax.spines[['right', 'top']].set_visible(False)
```

+++ {"nteract": {"transient": {"deleting": false}}}

We run the fit again

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
options = {}
options['sigmoid'] = sigmoid 
options['experiment_type'] = 'yes/no'
options['fixed_parameters'] = {}
options["stimulus_range"] = stim_range
```

```{code-cell} ipython3
---
nteract:
  transient:
    deleting: false
---
res = psignifit.psignifit(data, **options)
```

+++ {"nteract": {"transient": {"deleting": false}}}

and plot to ensure we found a good fit

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
  source_hidden: false
nteract:
  transient:
    deleting: false
---
psigniplot.plot_psychometric_function(res)
```

