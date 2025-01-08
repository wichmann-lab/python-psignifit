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

# Fix parameters

Fixing parameters is done by simply passing a dictionary as the option 'fixed_parameters'. 
For example, in the following example we fix the
parameters lambda and gamma to 0.02 and 0.5 respectively:

```
fixed_parameters = {'lambda': 0.02, 'gamma': 0.5}
```

The following example uses those fixed parameters to fit data

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

fixed_parameters = {'lambda': 0.02, 'gamma': 0.5}

res = ps.psignifit(data, sigmoid='norm', experiment_type='yes/no',
                   fixed_parameters=fixed_parameters)
```

We print the parameter estimates to check that lambda was indeed fixed

```{code-cell} ipython3
print(res.parameter_estimate)
```
