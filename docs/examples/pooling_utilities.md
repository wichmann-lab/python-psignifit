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
mystnb:
  execution_timeout: 100
---
# Pooling Utility

*psignifit* comes with a pooling utility, which can pool trials of very similar stimulus levels into blocks to reduce the time required for fitting, to profit from the beta-binomial model and to improve interpretability of data-plots.

```{note}
In contrast to the MATLAB implementation, *psignifit* does not pool implicitly. Instead a warning is printed if pooling might be useful. Then pooling can be run as a separate call using `psignifit.tools.pool_blocks`
```

The general form of the manual pooling utility looks like this:
```
pooled_data = pool_blocks(data, max_tol=0.4, max_gap=10, max_length=10)
```

It pools together trials which differ at most `max_tol`, are separated by
maximally `max_gap` trials from other levels, and are maximum `max_length`
trials apart overall.


# Example
For illustration we will use a dataset obtained from running a Quest run with 400 trials.

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt

import psignifit as ps
import psignifit.psigniplot as psp

data = np.genfromtxt("pooling_example_data.csv", delimiter=',', usecols=(0,1,2))
print(data)
```

First we fit this dataset using the settings for a 2AFC experiment and a normal distribution.


```{code-cell} ipython3
options = {'sigmoid': 'norm',
           'experiment_type': '2AFC'
           }

res = ps.psignifit(data, **options)
```
Note that this takes a bit longer than usual and it gives us a warning about the number per bloc. Let's have a look at what psignifit did automatically here:

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1);
psp.plot_psychometric_function(res, ax=ax);
```
Each block contains only very few trials. Thus the beta-binomial model cannot help much to correct for overdispersion. Furthermore the many lines of data slow psignifit down.


By default psignifit pools only trials collected at the exact same stimulus level. We can pool the data manually to make each bloc contain all trials which differ by up to 0.01.


```{code-cell} ipython3
from psignifit.tools import pool_blocks
pooled = pool_blocks(data, max_tol=0.01)
print(pooled)
res = ps.psignifit(pooled, **options)
fig, ax = plt.subplots(1, 1);
psp.plot_psychometric_function(res, ax=ax);
```

Now we pooled quite strongly.

The other two options allow us to restrict which trials should be pooled again. For example we could restrict the number of trials to 25 per block:

```{code-cell} ipython3
pooled = pool_blocks(data, max_tol=0.01, max_length=25)
print(pooled)
res = ps.psignifit(pooled, **options)
fig, ax = plt.subplots(1, 1);
psp.plot_psychometric_function(res, ax=ax);
```

This breaks the large blocks up again allowing us to notice if there was more variability over time than expected.

The last option gives us a different rule to achieve something in a similar direction: we can enforce to pool only subsequent trials like this:

```{code-cell} ipython3
pooled = pool_blocks(data, max_tol=0.01, max_length=np.inf, max_gap=0)
res = ps.psignifit(pooled, **options)
fig, ax = plt.subplots(1, 1);
psp.plot_psychometric_function(res, ax=ax);
```
Values between 0 and infinity will allow "gaps" of maximally options.poolMaxGap trials which are not included into the block (because their stimulus level differs too much).

Of course all pooling options can be combined.
