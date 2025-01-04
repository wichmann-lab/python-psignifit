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

# Basic options

This demo explains the basic and required options that you need to set to use psignifit.

```{code-cell} ipython3
import numpy as np

import psignifit as ps

# to have some data we use the data from the first demo.
data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC')
```

## Experiment type

Depending on the experimental condition, the experimenter has different a priori knowledge about the asymptotes. In *psignifit* this is handled by an option called `experiment_type`. There are three different types available:

### nAFC Experiments
The first is meant for n-alternative-forced-choice experiments, i. e. for experiments for which n alternatives are given, an answer is enforced and exactly one alternative is right. You can choose it by setting the option `experiment_type` to the string "nAFC", where you replace *n* with the number of options. For example, if you use 2, 3 or 4 alternatives you may pass:

```
experiment_type = '2AFC'
experiment_type = '3AFC'
experiment_type = '4AFC'
```

This mode fixes the lower asymptote gamma to $1/n$ and leaves the upper asymptote free to vary.

### Yes/No Experiments ###
Intended for simple detection experiments asking subjects whether they perceived a single presented stimulus or not, or any other experiment which has two possible answers of which one is reached for "high" stimulus levels and the other for "low" stimulus levels. You choose it with:

```
experiment_type = 'yes/no'
```

This sets both asymptotes free to vary and applies a prior to them favoring small values, e.g. asymptotes near 0 and 1 respectively.

### Equal Asymptote Experiments ###
This setting is essentially a special case of Yes/No experiments. Here the asymptotes are "yoked", i. e. they are assumed to be equally far from 0 or 1. This corresponds to the assumption that stimulus independent errors are equally likely for clear "Yes" answers as for clear "No" answers. It is chosen with:

```
experiment_type = 'equal asymptote'
```

Note that this will make fitting the psychometric function considerably faster than `yes/no` because *psignifit* has to only fit four rather than five parameters.


## Sigmoid


The default value of `sigmoid` is the cumulative Gaussian. 
```
sigmoid = 'norm'
```

Another standard alternative is the logistic function.

```
sigmoid = 'logistic'
```

The Gumbel and reversed Gumbel functions for asymmetric psychometric functions are also included. The Gumbel has a longer lower tail the reversed Gumbel a longer upper tail.  

```
sigmoid = 'gumbel'
sigmoid = 'rgumbel'
```

You can find [all sigmoids supported by psignifit here](plot_all_sigmoids).

If the standard sigmoids do not match your requirements, you may provide your own sigmoid, which has to be a subclass of `psignifit.sigmoids.Sigmoid`. 
Consult the [API reference](../reference/api) and the source code to learn how to set your own sigmoid.

