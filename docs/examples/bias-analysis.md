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

# Bias analysis

For 2AFC experiments it makes sense to check whether the observers are
biased, i.e. whether they treat the two alternative answers differently.
To facilitate such checks we provide a function
biasAna(data1,data2,options), which we demonstrate in this section.

In the background this function calculates fits with changed priors
on the guessing and lapse rate, to leave the guessing rate free with only
a weak prior (beta(2,2)) to be near 0.5. To allow this in the fitting we
have to constrain the lapse rate to the range [0,0.1] leaving the range
[0,0.9] for the guessing rate.

To use the function we first have to separate our dataset to produce two
separate datasets for the two alternatives (i.e. for signal in first
interval vs. signal in second interval; signal left vs signal right, etc.)

For demonstration purposes we produce different pairs of datasets, which
combine to our standard test dataset (data11 and data12, data21 and data22,
and data31 and data32 are a pair each:

```{code-cell} ipython3
import numpy as np

import psignifit as ps
from psignifit import psigniplot
```

```{code-cell} ipython3
data11 = np.array([[0.0010, 22.0000, 45.0000], [0.0015, 27.0000, 45.0000],
                   [0.0020, 24.0000, 47.0000], [0.0025, 20.0000, 44.0000],
                   [0.0030, 27.0000, 45.0000], [0.0035, 27.0000, 44.0000],
                   [0.0040, 30.0000, 45.0000], [0.0045, 30.0000, 44.0000],
                   [0.0050, 39.0000, 43.0000], [0.0060, 40.0000, 46.0000],
                   [0.0070, 47.0000, 48.0000], [0.0080, 47.0000, 47.0000],
                   [0.0100, 42.0000, 42.0000]])

data12 = np.array([[0.0010, 23.0000, 45.0000], [0.0015, 23.0000, 45.0000],
                   [0.0020, 20.0000, 43.0000], [0.0025, 24.0000, 46.0000],
                   [0.0030, 25.0000, 45.0000], [0.0035, 26.0000, 46.0000],
                   [0.0040, 32.0000, 45.0000], [0.0045, 34.0000, 46.0000],
                   [0.0050, 37.0000, 47.0000], [0.0060, 38.0000, 44.0000],
                   [0.0070, 41.0000, 42.0000], [0.0080, 43.0000, 43.0000],
                   [0.0100, 48.0000, 48.0000]])

data21 = np.array([[0.0010, 33.0000, 45.0000], [0.0015, 37.0000, 45.0000],
                   [0.0020, 36.0000, 47.0000], [0.0025, 32.0000, 44.0000],
                   [0.0030, 36.0000, 45.0000], [0.0035, 36.0000, 44.0000],
                   [0.0040, 37.0000, 45.0000], [0.0045, 36.0000, 44.0000],
                   [0.0050, 42.0000, 43.0000], [0.0060, 43.0000, 46.0000],
                   [0.0070, 47.0000, 48.0000], [0.0080, 47.0000, 47.0000],
                   [0.0100, 42.0000, 42.0000]])

data22 = np.array([[0.0010, 12.0000, 45.0000], [0.0015, 13.0000, 45.0000],
                   [0.0020, 8.0000, 43.0000], [0.0025, 12.0000, 46.0000],
                   [0.0030, 16.0000, 45.0000], [0.0035, 17.0000, 46.0000],
                   [0.0040, 25.0000, 45.0000], [0.0045, 28.0000, 46.0000],
                   [0.0050, 34.0000, 47.0000], [0.0060, 35.0000, 44.0000],
                   [0.0070, 41.0000, 42.0000], [0.0080, 43.0000, 43.0000],
                   [0.0100, 48.0000, 48.0000]])

data31 = np.array([[0.0010, 22.0000, 45.0000], [0.0015, 25.0000, 45.0000],
                   [0.0020, 24.0000, 47.0000], [0.0025, 20.0000, 44.0000],
                   [0.0030, 20.0000, 45.0000], [0.0035, 21.0000, 44.0000],
                   [0.0040, 22.0000, 45.0000], [0.0045, 25.0000, 44.0000],
                   [0.0050, 32.0000, 43.0000], [0.0060, 35.0000, 46.0000],
                   [0.0070, 46.0000, 48.0000], [0.0080, 47.0000, 47.0000],
                   [0.0100, 42.0000, 42.0000]])

data32 = np.array([[0.0010, 23.0000, 45.0000], [0.0015, 25.0000, 45.0000],
                   [0.0020, 20.0000, 43.0000], [0.0025, 24.0000, 46.0000],
                   [0.0030, 32.0000, 45.0000], [0.0035, 32.0000, 46.0000],
                   [0.0040, 40.0000, 45.0000], [0.0045, 39.0000, 46.0000],
                   [0.0050, 44.0000, 47.0000], [0.0060, 43.0000, 44.0000],
                   [0.0070, 42.0000, 42.0000], [0.0080, 43.0000, 43.0000],
                   [0.0100, 48.0000, 48.0000]])
```

now we can check whether our different pairs show biased behaviour:

```{code-cell} ipython3
# We start with the first pair of data:
psigniplot.plot_bias_analysis(data11, data12)
```

This command will open a figure, which constains plots for the first
dataset alone (red), for the second dataset alone (blue) and for the
combined dataset (black).

The top plot show the three psychometric functions, which for the first
split of the data lie neatly on top of each other, suggesting already
that the psychometric functions obtained for the two intervals are very
similar and that no strong biases occured.

Below there are posterior marginal plots for the threshold, width, lapse
rate and guessing rate. These plots are diagnostic which aspects of the
psychometric function changed between intervals.
For our first example these plots all confirm the first impression
obtained from the first plot. It seems neither of the parameters has
changed much.

```{code-cell} ipython3
# Next, we check our second split of data:
psigniplot.plot_bias_analysis(data21, data22)
```

In this case there seems to be very strong "finger bias", i.e. the
observer is much better at guessing in one than in the other part of the
data. This can happen, when observer do not guess the two intervals
with equal probability.

This bias can be seen directly in the fitted psychometric functions, but
also in the marginal distributions, which show that the guessing rate
gamma is very different for the two passed datasets, but the other
parameters are still consistent.

As this kind of bias leads to relatively undisturbed inference for
threshold and width, the estimates from the original function might still
be usable.

```{code-cell} ipython3
# Now we have a look at our third splitting:
psigniplot.plot_bias_analysis(data31, data32)
```

In this case the guessing rate does not seem to differ between intervals,
but the psychometric functions are shifted, i.e. the task was easier in
one than in the other case.

This can be observed in the plotted functions again or by observing that
the posterior clearly favours different thresholds for the two parts of
the data.

If this type of bias occurs one should be careful in interpreting the
results, as it seems that the two allegedly equal variants of the
experiment would not yield equal results.
Also the width estimate of the combined function will not be the width of
the two functions in the two intervals, contaminating this measure as
well.


In summary: Use these plots to find if the psychometric functions for the
two alternatives in 2AFC differ. This should allow you to find the
relatively harmless biases in the guessing of observers and the much more
harmful biases in true performance.
This is especially important as all biases we demonstrated here cannot be
detected by looking at the combined psychometric function only.
In real datsets the biases we demonstrated can be combined. Nonetheless
the plots of the marginals should allow a separation of the different
biases.

