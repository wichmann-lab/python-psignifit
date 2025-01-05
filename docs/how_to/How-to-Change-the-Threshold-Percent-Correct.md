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

# Change the threshold percentage correct 

This option sets the proportion correct to correspond to the threshold on the *unscaled* sigmoid. Possible values are 
in the range from 0 to 1, default is 0.5. The default corresponds to 75\% in a 2AFC task (midway between the guess 
rate of 50 % and ceiling performance 100%).

To set it to a different value, for example to 90 %, you'll do:

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

res = ps.psignifit(data, sigmoid='norm', experiment_type='yes/no', thresh_PC=0.9)
```

Note that you will get a warning, because the default prior assumes that
the experimental stimulus range covers the range where the threshold likely
falls. If this doesn’t match your setup, you’ll need a custom prior. See
the [priors demo](../examples/priors)  to learn how to do it.
