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

# Result object

Here we explain what is stored in the result object returned by Psignifit

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import psignifit as ps
import psignifit.psigniplot as psp

# to have some data we use the data from demo_001
data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

# Run psignifit
res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC')
```

Now we can have a look at the res dictionary and all its fields.

## Parameter estimates
The most important result are the fitted parameters of the psychometric
function. They can be found in a dictionary format.

```{code-cell} ipython3
print(res.parameter_estimate)
```

For each of these parameters, also the confidence interval is contained
in the results as a dictionary.
For example for the threshold the confidence intervals are

```{code-cell} ipython3
print(res.confidence_intervals['threshold'])
```

and for the width parameter

```{code-cell} ipython3
print(res.confidence_intervals['width'])
```

## Options
In addition, the result contains the complete set of options passed:

```{code-cell} ipython3
print(res.configuration)
```

Because the result object contains also the options passed, it is easy to do similar fits with the same options.

```{code-cell} ipython3
# copy the data and introduce a shift in all stimulus values
otherdata = np.copy(data)
otherdata[:, 0] = otherdata[:, 0] + 0.01

# fit with exact same options
other_res = ps.psignifit(otherdata, conf=res.configuration)

# the difference in threshold should return the introduced shift
print(other_res.parameter_estimate['threshold'] - res.parameter_estimate['threshold'])
```

## Saving to JSON

This result object is by default serializable, that means that it can be saved to a JSON file

```{code-cell} ipython3
file_name = 'psignifit-result.json'
res.save_json(file_name)
```

The file contains all data, information about the fit, and the fitted parameters.
It can be loaded again to be used at a later time, for example for plotting

```{code-cell} ipython3
loaded_res = ps.Result.load_json(file_name)
print(loaded_res.parameter_estimate)
```

```{code-cell} ipython3
# plotting from loaded file
psp.plot_psychometric_function(loaded_res);
```

## Debug mode

If needed, *psignifit* can also return the whole grid of posterior probabilities by passing the option `debug=True`:

```{code-cell} ipython3
# Run psignifit in debug mode
res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC', debug=True)
```

In this mode the prior are stored as function objects. Because of this, the result object is not serializable anymore
and cannot be saved to JSON. Some diagnostic plots require the debug mode though.
