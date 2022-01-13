# -*- coding: utf-8 -*-
"""
4. Priors
=========

This demo covers how we set the priors for different situations.
This gives you effective control over which parameters of the
psychometric function are considered for fitting and all confidence
statements.
There is no way to do Bayesian statistics without a prior.


Staying with the standard
-------------------------

First: let's have a look what psignifit does if you do not specify a
prior explicitly:
Then psignifit chooses a prior which assumes that you somehow sampled the
whole psychometric function.
Specifically it assumes that the threshold is within the range of
the data and with decreasing probability up to half the range above or
below the measured data.
For the width we assume that it is somewhere between two times the
minimal distance of two measured stimulus levels, and the range of the
data or with decreasing probability up two 3 times the range of the data.
"""

import matplotlib.pyplot as plt
import numpy as np

import psignifit as ps

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# To illustrate this we plot the priors from our original example from
# demo_001:

data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])


res = ps.psignifit(data, sigmoid_name='norm', experiment_type='2AFC')

ps.psigniplot.plot_prior(res)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# You should check that the assumptions we make for the heuristic to work
# are actually true in the case of your data.
# e.g. check, whether one of the following statements holds:
# (1) You understand what our priors mean exactly and judge them to be
# appropriate.
# (2) You are sure you recorded a trial well above and a trial well below
# threshold.
# (3) Your posterior concentrates on an area for which the prior was
# constant.
#
# adjusting the realistic range
# -----------------------------
#
# There are situations for which the assumptions for our standard prior
# do not hold. For example when adaptive methods are used or you fit
# incomplete datasets. To fit these correctly psignifit allows you to set
# the realistic range for the threshold/ the range of data you expect to
# measure manually. In this part we show how to do this.
#
# For example consider the followind dataset, which is a simulation of a
# 3-down-1-up staircase procedure with 50 trials on a yes-no experiment.
# This samples considerably above threshold. In this case the true
# threshold and width were 1.
# Thus the assumption that we know that the threshold is in the range of
# the data is clearly violated.

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

# We fit this assuming the same lapse rate for yes and for no.
# By default this uses a cumulative normal fit, which is fine for now.
res = ps.psignifit(data, experiment_type='equal asymptote')

# We first have a look at the fitted function
plt.figure()
ps.psigniplot.plot_psych(res)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# You should notice that the percent correct is larger than 50 and we did
# not measure a stimulus level clearly below threshold. Thus it might be
# that the theshold is below our data, as it is the case actually in our
# example.
# This is a common problem with adaptive procedures, which do not explore
# the full possible stimulus range. Then our heuristic for the prior may
# easily fail.
#
# You can see how the prior influences the result by looking at the
# marginal plot for the threshold as well:

ps.psigniplot.plot_marginal(res, 0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# note that the dashed grey line, which marks the prior goes down where
# there is still posterior probability. This shows that the prior has an
# influence on the outcome.
#
# To "heal" this psignifit allows you to pass another range, for which you
# believe in the assumptions of our prior. The prior will be set as for the
# true data range, but for the provided range.
# For our example dataset we might give a generous range and assume the
# possible range is .5 to 1.5

res_range = ps.psignifit(data, stimulus_range=[.5, 1.5], experiment_type='equal asymptote')

# We can now have a look how the prior changed:
ps.psigniplot.plotPrior(res_range)

# By having a look at the marginal plot we can see that there is no area
# where the prior dominates the posterior anymore. Thus our result for the
# threshold is now dominated by the data everywhere.

plt.figure()
ps.psigniplot.plotMarginal(res_range, 0)

# Finally we can also compare our new fitted psychometric function,
# to see that even the point estimate for the psychometric function was
# influenced by the prior here:
plt.figure()
ps.psigniplot.plotPsych(res_range)
ps.psigniplot.plotPsych(res)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# The prior on the betabinomial variance- adjusting how conservative to be
# With the betabinomial model we have an additional parameter which
# represents how stationary the observer was.
# The prior on this parameter can be adjusted with a single parameter of
# psignifit:
# options['betaPrior']
# Larger values for this parameter represent a stronger prior, e.g. stronger
# believe in a stationary observer. Smaller values represent a more
# conservative inference, giving nonstationary observers a higher prior
# probability.
# 1 represents a flat prior, e.g. maximally conservative inference. Our
# default is 10, which fitted our simulations well, around several hundred
# the analysis becomes very similar to the binomial analysis to which the
# analysis converges when options.betaPrior goes to infinity.
# This will barely influence the point estimate you find for your
# psychometric function. It's main effect is on the confidence intervals
# which will grow or shrink.
#
# For an example we will fit the data from above once much more
# conservative, once more progressively:

# first again with standard settings:
data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

res = ps.psignifit(data, experiment_type='2AFC')

# first lets have a look at the results with the standard prior strength:
print('Fit:', res['Fit'])
print('confidence Intervals:', res['conf_Intervals'])

# now we recalculate with the smallest most conservative prior:
res1 = ps.psignifit(data, beta_prior=1, experiment_type='2AFC')

# and with a very strong prior of 200
res200 = ps.psignifit(data, beta_prior=200, experiment_type='2AFC')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# First see that the only parameter whose fit changes by this is the
# beta-variance parameter eta (the 5th)

print('Fit with beta prior = 1: ', res1.parameter_estimate)
print('Fit with beta prior = 200: ', res200.parameter_estimate)

# Now we have a look at the confidence intervals
# TODO: uncomment once confidence intervals are implemented
# print('confidence Intervals for beta prior = 1: ', res1.confidence_intervals)
# print('confidence Intervals for beta prior = 200: ', res200.confidence_intervals)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# They also do not change dramatically, but they are smaller for the 200
# prior than for the 1 prior.
#
# Our recommendation based on the simulations is to keep the 10 prior. If
# you have questions contact us.
#
# Passing Custom Priors
# ---------------------
#
# This part explains how to use custom priors, when you do not want to use
# our standard set, or it is wrong even for a corrected stimulus range.
# To do this you should know what you are doing, and everything is on your
# own risk.
#
# As an example we will fix the prior on lambda the lapse rate parameter
# of the psychometric funtion to a constant between 0 and .1 and zero
# elsewhere as it was done in the psignifit 2 toolbox.
#
# To use custom priors, first define the priors you want to use as function
# handles.
#
# For our example this works as follows:


def prior_lambda(x):
    return ((x >= 0) * (x <= .1)).astype('float')


custom_priors = {'lambda': prior_lambda}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Note that we did not normalize this prior. This is internally done by
# psignifit.
#
# If you are not familiar with function handles and anonymous functions
# in Python3 you can find an introduction to them here:
# http://www.programiz.com/python-programming/anonymous-function
#
# Most of the times you then have to adjust the bounds of integration as
# well. This confines the region psignifit operates on. All values outside
# the bounds implicitly have prior probability 0!
# For our example we set all bounds to NaN, which means they are set
# automatically and state only the bounds for lambda.

zero_bounds = {
    'threshold' (np.nan, np.nan),
    'width' (np.nan, np.nan),
    'lambda' (0, 1.),
    'gamma' (np.nan, np.nan),
    'eta' (np.nan, np.nan),
}

res = ps.psignifit(data, priors=custom_priors, bounds=zero_bounds, experiment_type='2AFC')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# There will be a warning that the prior chosen here is zero at some
# values. This is true, but we intend it to be like this, constraining our
# analysis stronger than the standard priors do.
#
# With these commands you have set the priors manually: Have a look at
# them:

plt.figure()
ps.psigniplot.plot_prior(res)

plt.figure()
ps.psigniplot.plot_psych(res)
