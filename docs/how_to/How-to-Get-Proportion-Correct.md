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

# Get proportion correct values

This demo shows how to get proportion correct values given from the fitted psychometric function,

We will need some fitted function for illustration. Thus we first fit our
standard data from [basic usage example](../basic-usage):


```{code-cell} ipython3
import numpy as np

import psignifit as ps

data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC')
```

Now, let's say we want to get the proportion correct values for stimulus levels `0.0012` and `0.0013`, which
were not measured during the experiment:

```{code-cell} ipython3
stimulus_levels = [0.0012, 0.0013]
prop_correct = res.proportion_correct(stimulus_levels)
print(prop_correct)
```

We can also get the proportion correct values with added noise derived from the estimated overdispersion
parameter `eta` (see the [Parameter recovery demo](../examples/parameter_recovery_demo.md#and-now-with-some-more-realistic-data)
and the [*Vision Research* paper](http://www.sciencedirect.com/science/article/pii/S0042698916000390)
for more info about the `eta` parameter). In essence, were you to draw a lot of these proportion correct values, their
variance would be compatible with that of samples from a beta binomial distribution with scale paramter equal to `eta`.

```{code-cell} ipython3
stimulus_levels = [0.0012, 0.0013]
prop_correct = res.proportion_correct(stimulus_levels, with_eta=True)
print(prop_correct)
```

If you are intersted in the proportion correct values based on the `mean` estimate instead of the default
`MAP` estimate, you can get them with:

```{code-cell} ipython3
stimulus_levels = [0.0012, 0.0013]
prop_correct = res.proportion_correct(stimulus_levels, estimate_type='mean')
print(prop_correct)
```
