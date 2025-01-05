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

# Get thresholds and slopes

This demo shows how to get thresholds values at any proportion correct,
and similarly how to get the slope of the psychometric function
at any stimulus value as well as any proportion correct.


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

## Obtaining Threshold Values

For comparison to other estimation techniques we provide a way to
calculate thresholds at any given proportion correct.
This is done with the `threshold` method of the `result` object.

The method calculates the threshold of the function fit for a given proportion correct.
An optional argument determines if the calculation follows on the scaled
sigmoid (default), or on the original sigmoid unscaled by guessing and lapse rate (`unscaled=True`)

For example: this call will find the value at which our function reaches
90% correct:

```{code-cell} ipython3
res.threshold(0.9)  # which should be 0.0058
```

A usecase for the unscaled case might be to find the threshold for the
middle of the psychometric function independent of the guessing and lapse
rate. For this you can use the method with argument `unscaled=True`

```{code-cell} ipython3
res.threshold(0.5, unscaled=True)
```

which should be 0.0046, which is exactly the definition of the threshold we use in the fitting.

```{code-cell} ipython3
res.parameter_estimate['threshold']
```

The function also returns worst-case credible intervals for the
threshold.

```{code-cell} ipython3
threshold, CI = res.threshold(0.5, unscaled=True)
CI
```

The credible intervals are for the confidence levels given for your
function fit.
The estimates calculated by this function are very conservative when
you move far away from the original threshold, as we simply assume the
worst case for all other parameters instead of averaging over the values.

## Obtaining Slope Values

The result object also provides two methods to calculate the slope of the psychometric
function from the fits.
The first returns the slope at a given stimulus value, for example for a value of 0.006

```{code-cell} ipython3
res.slope(0.006)
```

Also you can get the slope at a particular percentage correct with
the method `slope_at_proportion_correct`. For example

```{code-cell} ipython3
res.slope_at_proportion_correct(0.6)
```

will yield the slope at the value where the psychometric function reaches
60% correct.

Like the `threshold` method, you can get the slope by giving a proportion correct
for the *unscaled* sigmoid.
For example we can calculate the slope at the midpoint of the
psychometric function using

```{code-cell} ipython3
res.slope_at_proportion_correct(0.5, unscaled=True)
```

Note that each of the two above methods do not return confidence intervals.
