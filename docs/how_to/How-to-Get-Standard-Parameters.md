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

# Get standard parameters

In *psignifit* we chose to parametrized all sigmoids to the common parameter space spanned by threshold and width. To compare these parameters to values from the literature, older fits, etc. we provide a function transforming our parameters to the common standard parametrizations. It is implemented as a method `standard_parameter_estimate` on the result object and it computes the standard values for any sigmoid. For example:

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

res = ps.psignifit(data, sigmoid='norm', experiment_type='yes/no')
```

In this example we have a Gaussian sigmoid with `alpha=0.05`, `PC=0.5` fitted. The parameters we would get out, threshold and width, correspond to the standard parameters `loc` (mean) and `scale` (standard deviation) in the following way:

```{code-cell} ipython3
# the mean (loc):
print(f'loc = {res.parameter_estimate["threshold"]}')

# the standard deviation (scale):
# 1.644853626951472 is the normal PPF at alpha=0.95
print(f'scale = {res.parameter_estimate["width"] / (2 * 1.644853626951472)}')
```

The standard parameters can be computed for any sigmoids more easily like this:

```{code-cell} ipython3
loc, scale = res.standard_parameter_estimate()
print(f'{loc=}\n{scale=}')
```
