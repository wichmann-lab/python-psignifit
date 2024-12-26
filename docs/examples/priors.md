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

# Priors

This demo covers how we set the priors for different situations.
This gives you effective control over which parameters of the
psychometric function are considered for fitting and all confidence
statements.
There is no way to do Bayesian statistics without a prior.


## Staying with the standard

First let's have a look what psignifit does if you do not specify a
prior explicitly. In this case psignifit uses an heuristic and chooses
a prior which assumes that you somehow sampled the whole psychometric function.

Specifically, it assumes that the threshold is within the range of
the data, and with decreasing probability up to half the range above or
below the measured data.
For the width we assume that it is somewhere between two times the
minimal distance of two measured stimulus levels and the range of the
data; and with decreasing probability up to 3 times the range of the data.
This default priors can be easily visualized, see below.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import json

import psignifit as ps
from psignifit import psigniplot
```

To illustrate this we plot the priors from our original example from the [basic usage](../basic-usage) page.

```{code-cell} ipython3
data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])


res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC', debug=True)

# notice that we need to pass 'debug=True'
psigniplot.plot_prior(res);
```

The panels in the top row show the prior densities for threshold, width
and lapse rate respectively. In the second row psychometric functions
corresponding to the 0, 25, 75 and 100% quantiles of the prior are
plotted in colour. The other parameters of the functions are left at
the prior mean value, which is also marked in all panels in black.
As an orientation the small black dots in the lower panels mark the
levels at which data were sampled.


You should check that the assumptions we make for the heuristic to
work are actually true in the case of your data.
e.g. check whether (at least) one of the following statements holds:


- You understand what our priors mean exactly and judge them to be appropriate.
- You are sure you recorded a trial well above and a trial well below threshold.
- Your posterior concentrates on an area for which the prior was (nearly) constant.


You can evaluate your priors with above plot. With these color-coded sigmoids
you can evaluate the adequacy of the prior:

For a threshold prior to be adequate (first column), it should be flat along all your stimulus levels; this will ensure that the threshold estimation is driven exclusively by the data. The stimulus levels of your data are shown as black-dots in the x-axis.
The prior is flat (constant) for all threshold values between the red and blue sigmoids; both are equally likely. The yellow and green sigmoids are very unlikely, and this makes sense as they are outside of the range of stimulus levels. Here you can see that `psignifit` assumes that you choose stimulus values along the whole range the psychometric function.

Similarly, for the width prior (middle column) the sigmoids varing between the red and blue cases are more or less likely; contrarily the yellow and green sigmoids are very unlikely. Again this makes sense, as `psignifit` assumes that you sampled the whole range of the psychometric function and not just around the threshold.

Finally for the lapse prior it assumes a decaying prior (right column), which in simulations have been shown to be a reasonable assumption. You don't expect that observers lapse more than 20%; if so, then the observer is the problem (e.g. they were not following instructions, falling asleep, lapsing in attention, etc). For these cases maybe consider to repeat the experiment.



## Adjusting the realistic range

There are situations for which the assumptions for our standard prior
do not hold. For example when **adaptive methods** are used or you fit
incomplete datasets. To fit these correctly psignifit allows you to set
the realistic range for the threshold/ the range of data you expect to
measure manually. In this part we show how to do this.

For example consider the followind dataset, which is a simulation of a
3-down-1-up staircase procedure with 50 trials on a yes-no experiment.
This samples considerably above threshold. In this case the true
threshold and width were 1.
Thus the assumption that we know that the threshold
(threshold as defined in psignifit 4, not what 3-down-1-up defines as threshold!)
is in the range of the data is clearly violated.


```{code-cell} ipython3
data = np.array([[1.5000, 3.0000, 3.0000], [1.3500, 3.0000, 3.0000],
                 [1.2150, 1.0000, 2.0000], [1.3365, 2.0000, 3.0000],
                 [1.4702, 3.0000, 3.0000], [1.3231, 3.0000, 3.0000],
                 [1.1908, 1.0000, 2.0000], [1.3099, 3.0000, 3.0000],
                 [1.1789, 1.0000, 2.0000], [1.2968, 2.0000, 3.0000],
                 [1.4265, 3.0000, 3.0000], [1.2838, 1.0000, 2.0000],
                 [1.4122, 3.0000, 3.0000], [1.2710, 1.0000, 2.0000],
                 [1.3981, 1.0000, 2.0000], [1.5379, 1.0000, 2.0000],
                 [1.6917, 3.0000, 3.0000], [1.5225, 3.0000, 3.0000],
                 [1.3703, 2.0000, 3.0000]])
```

We fit the data assuming the same lapse rate for yes and for no.

```{code-cell} ipython3
res = ps.psignifit(data,
                   experiment_type='equal asymptote',
                   debug=True)
```
by default this uses a cumulative normal fit, which is fine for now.
The prior for this function looks like this:

```{code-cell} ipython3
psigniplot.plot_prior(res);
```

which is not particularly conspicuous. Comparing the functions to the
stimulus range shows that we still expect a reasonably sampled
psychometric function with threshold in the sampling range.

But now, have a look at the fitted function:

```{code-cell} ipython3
plt.figure();
psigniplot.plot_psychometric_function(res, data_size=0.1);
```

You should notice that the proportion correct is larger than 50 and we did
not measure a stimulus level clearly below threshold (as defined in psignifit 4).
Thus it might be that the threshold is below our data, as it is the case actually in our
example.
This is a common problem with adaptive procedures, which do not explore
the full possible stimulus range. Then our heuristic for the prior may
easily fail.

You can see how the prior influences the result by looking at the
marginal plot for the threshold as well

```{code-cell} ipython3
psigniplot.plot_marginal(res, 'threshold');
```

Note that the dashed grey line, which marks the prior, decreases where
there is still posterior probability. This shows that the prior has an
influence on the outcome.



To "heal" this, psignifit allows you to pass another range, for which you
believe in the assumptions of our prior.
You pass this range with argument `stimulus_range`, and `psignifit`
will be set the prior considering this range instead.
For our example dataset we might give a generous range and assume the
possible range is .5 to 1.5.

```{code-cell} ipython3
res_range = ps.psignifit(data,
                         experiment_type='equal asymptote',
                         debug=True,
                         stimulus_range=[.5, 1.5])
```

We can now take a look how the prior changed

```{code-cell} ipython3
psigniplot.plot_prior(res_range);
```

By having a look at the marginal plot we can see that there is no area
where the prior dominates the posterior anymore. Thus our result for the
threshold is now driven by the data.
Note also that the credible interval now extends considerably further down as well.

```{code-cell} ipython3
plt.figure();
psigniplot.plot_marginal(res_range, 'threshold');
```

Finally we can also compare our new fitted psychometric function.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1);
psigniplot.plot_psychometric_function(res_range, ax=ax,
                                      data_size=0.1, plot_parameter=False);
psigniplot.plot_psychometric_function(res, ax=ax, line_color='gray',
                                      plot_data=False, plot_parameter=False);
```

The function the new prior (in black) is shifted to the left in
comparison to the original one (in gray), indicating
that the threshold was influenced by the prior


## The prior on the beta-binomial variance — adjusting how conservative one wants to be

With the betabinomial model we have an additional parameter which
represents how stationary the observer was.
The prior on this parameter can be adjusted with a single parameter of
psignifit: `beta_prior`.

Larger values for this parameter represent a stronger prior, e.g. stronger
believe in a stationary observer. Smaller values represent a more
conservative inference, giving non-stationary observers a higher prior
probability.

`beta_prior=1` represents a flat prior, e.g. maximally conservative inference. Our
default is 10, which fitted our simulations well, around several hundred
the analysis becomes very similar to the binomial analysis.
This will barely influence the point estimate you find for your
psychometric function. Its main effect is on the confidence intervals
which will grow or shrink.

For example we will fit the data from above once much more
conservative, once more progressively:

```{code-cell} ipython3
# first again with standard settings:
data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

res = ps.psignifit(data, experiment_type='2AFC', confP=[0.95])
```

First lets have a look at the results with the standard prior strength:

```{code-cell} ipython3
print('Fit:')
print(json.dumps(res.parameter_estimate, indent=2))
# (here we use json to show the dictionary in a pretty format)
```

The credible intervals are

```{code-cell} ipython3
print('Confidence intervals:')
print(json.dumps(res.confidence_intervals, indent=2))
```

Now we recalculate with the smallest most conservative prior:

```{code-cell} ipython3
res1 = ps.psignifit(data, beta_prior=1, experiment_type='2AFC', confP=[0.95])
```

and with a very strong prior of 200

```{code-cell} ipython3
res200 = ps.psignifit(data, beta_prior=200, experiment_type='2AFC', confP=[0.95])
```

First see that the only parameter whose fit changes by this is the
beta-variance parameter $\eta$ (eta)

```{code-cell} ipython3
print('Fit with beta prior = 1: ')
print(json.dumps(res1.parameter_estimate, indent=2))

print('Fit with beta prior = 200: ')
print(json.dumps(res200.parameter_estimate, indent=2))
```

Now we have a look at the confidence intervals (here only for the 95% ones)

```{code-cell} ipython3
print('Confidence Intervals for beta prior = 1: ')
print(json.dumps(res1.confidence_intervals, indent=2))

print('Confidence Intervals for beta prior = 200: ')
print(json.dumps(res200.confidence_intervals, indent=2))
```

They also do not change dramatically, but they are smaller for the 200
prior than for the 1 prior.

Our recommendation based on the simulations is to keep the 10 prior. If
you have questions contact us.

## Passing custom priors

This part explains how to use custom priors, when you do not want to use
our standard set, or it is wrong even for a corrected stimulus range.

```{warning}
To do this you should know what you are doing, and everything is on your
own risk.
```

As an example we will fix the prior on the lapse rate parameter $\lambda$
to a constant between 0 and .1, and zero
elsewhere, as it was done in the psignifit 2 toolbox.

To use custom priors, first define the priors you want to use as function
and include it in a dictionary with the key value corresponding to
the parameter name. For our example this works as follows:

```{code-cell} ipython3
def prior_lambda(x):
    # we add a small value because prior should not return 0
    return ((x >= 0) * (x <= .1)).astype('float') + 1e-10

custom_priors = {'lambda': prior_lambda}
```

Note that we did not normalize this prior. This is internally done by
psignifit.

Most of the times you then have to adjust the bounds of integration as
well. This confines the region psignifit operates on. All values outside
the bounds implicitly have prior probability of 0.

For our example we set manually the bounds for the lambda parameter

```{code-cell} ipython3
custom_bounds = {
    'lambda': (0.0, 0.2),
}
```

We fit now with the custom priors and bounds

```{code-cell} ipython3
res = ps.psignifit(data,
                   priors=custom_priors,
                   bounds=custom_bounds,
                   experiment_type='2AFC', debug=True)
```

You can have a look at priors and the fitted function as follows

```{code-cell} ipython3
plt.figure();
ps.psigniplot.plot_prior(res);

plt.figure();
ps.psigniplot.plot_psychometric_function(res);
```
