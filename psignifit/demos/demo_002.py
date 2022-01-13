# -*- coding: utf-8 -*-
"""
 2. More Options
 ===============

 Which options can one set?

 This demo explains all fields of the options dictionary, e.g. which options
 you can set for the fitting process as a user.
"""

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

# initializing options dictionary
config = {
    'sigmoid_name': 'norm',
    'experiment_type': '2AFC'
}

# now or at any later time you can run a fit with this command.
res = ps.psignifit(data, **config)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Sigmoid
# --------
# The standard options of `sigmoid_name` and `experiment_type` are described in
# :ref:`Demo 1 <sphx_glr_generated_examples_demo_001.py>`.
#
# If the standard sigmoids do not match your requirements, you
# may provide a reference to your own sigmoid, which has to be
# a subclass of :class:`psignifit.sigmoids.Sigmoid`.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Estimation Type
# ---------------
# How you want to estimate your fit from the posterior
# 'mean' The posterior mean. In a Bayesian sence a more suitable estimate.
# the expected value of the Posterior.
# 'MAP' The MAP estimator is the maximum a posteriori computed from
# the posterior.

config['estimate_type'] = 'MAP'
config['estimate_type'] = 'mean'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Optimization Steps
# ------------------
# This sets the number of grid points on each dimension in the final
# fitting (grid_steps) and in the moving of bounds (steps_moving_bounds)
# as a dictionary with keys threshold, width, lambda (upper asymptote),
# gamma (lower asymptote), and eta (variance scaling).
#
# You may change this if you need more accurate estimates on the sparsely
# sampled parameters or if you want to play with them to save time.
#
# For example to get an even more exact estimate on the
# lapse rate/upper asymptote, set lambda to 50.
# Such, the lapse rate is sampled at 50 places giving you a much more exact
# and smooth curve for comparisons.
#
config['grid_steps'] = {'lambda': 50}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Confidence intervals
# --------------------
# The confidence level for the computed confidence intervals.
# This may be set to any number between 0 and 1 excluding.

# for example to get 99% confidence intervals try
config['confP'] = .99

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# You may specify a vector with N elements as well.
# If you do the conf_intervals in the
# result will be a 5x2xN array containing the values for the different
# confidence levels in the 3rd dimension.

# will return 4 confidence intervals for each parameter for example.
config['confP'] = [.95, .9, .68, .5]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# `CI_method` sets how the confidence intervals are computed in getConfRegion
# possible variants are:
#
# 'stripes': Find a threshold with (1-alpha) above it
#      This will disregard intervals of low posterior probability and then move
#      in from the sides to adjust the exact CI size.
#      This can handle bounds and asymmetric distributions slightly better, but
#      will introduce slight jumps of the confidence interval when confp is
#      adjusted depending on when the gridpoints get too small posterior
#      probability.
# 'project':
#      Project the confidence region on each axis
# 'percentiles': find alpha/2 and 1-alpha/2 percentiles (alpha = 1-confP)
#      This cuts at the estimated percentiles-> always tries to place alpha/2
#      posterior probability above and below the credible interval.
#      This has no jumping but will exclude bound values even when they have
#      the highest posterior. Additionally it will not choose the area of
#      highest posterior density if the distribution is skewed.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters
# ----------
# Which percent correct correspond to the threshold?
# Given in Percent correct on the unscaled sigmoid (reaching from 0 to 1).
# For example to define the threshold as 90% correct try:

config['thresh_PC'] = .9

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# How the width of a psychometric function is defined can be changed by `width_alpha`:
#   `width= psi^(-1)(1-alpha) - psi^(-1)(alpha)`
#   where `psi^(-1)` is the inverse of the sigmoid function.
#
# `width_alpha` must be between 0 and .5 excluding.
#
# Thus this would enable the useage of the interval from .1 to .9 as the
# width for example:

config['width_alpha'] = .1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Priors
# ------
#
# If you want to set your priors manually, you can set `priors`
# as a dict with the parameter name as key and the custom prior function
# as the value.
#
# For details on how do change these refer to
# https://github.com/wichmann-lab/psignifit/wiki/Priors and
# :ref:`Demo 4 <sphx_glr_generated_examples_demo_004.py>`
#
# The strength of the prior in favor of a binomial observer ca be set with
# the `beta_prior` option.
# Larger values correspond to a stronger prior. We choose this value after
# a rather large number of simulations. Refer to the paper to learn more
# about this.


config['beta_prior'] = 15

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Bounds
# ------
# You may provide your own bounds for the parameters.
# This should be a dict of the parameter name and tuples of
# start and end of the range.
#
# For example this would set the bounds to threshold between 1 and 2,
# width between .1 and 5, a fixed lapse rate of .05,
# a fixed lower asymptote at .05, and a maximum on the variance scale of .2
#
# .. note::
#     By this you artificially exclude all values out of this range. Only
#     exclude parameter values, which are truely impossible!'''

config['bounds'] = {
    'threshold': (1, 2),
    'width': (.1, 5),
    'lambda': (.05, .05),
    'gamma': (.5, .5),
    'eta': (np.exp(-20), .2)
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parts of the grid which produce marginal values below `max_bound_values`
# are considered 0 and are excluded calculations.
# It should be a very small value and at least smaller than `1/(max(grid_steps))`.
#
# This for example would exclude fewer values and more conservative
# movement of the bounds:

config['max_bound_values'] = np.exp(-20)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# By default, the bounds are moved together during optimization.
# Usually this is good to concentrate on the right area
# in the parameter space.
#
# This can be turned off.
# Your posterior will always use the initial setting for the bounds.
# This is usefull if you set the bounds by hand and do not want
# psignifit to move them after this.

config['move_bounds'] = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Pooling
# -------
#
# These options set how your data is pooled into blocks.
# Then we pool together a maximum of poolMaxLength trials,
# which are separated by a maximum of poolMaxGap trials of other stimulus levels.
# If you want you may specify a tolerance in stimulus level to pool trials,
# but by default we only pool trials with exactly the same stimulus level.
#
# .. note::
#     In contrast to the matlab implementation, python-psignifit does not
#     pool implicitly. Instead a warning is printed, if pooling might be useful.
#     Then pooling can be run as the separate function.
#
# Dynamic Grid
# ------------
# Toggles the useage of a dynamic/adaptive grid.
#
# .. note::
#   There was hope for a more exact estimate by this, but although the curves
#   look smoother the confidence intervals were not more exact.
#   This is why this was not ported to python-psignifit.
