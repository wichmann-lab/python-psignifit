# -*- coding: utf-8 -*-
"""
5. Plotting
===========

 Here the basic plot functions which come with the toolbox are explained.
 Most of the functions return the handle of the axis they plotted in
 to enable you to plot further details and change axis properties after the plot.

 To have something to plot we use the example data as provided in
 :ref:`Demo 1 <sphx_glr_generated_examples_demo_001.py>`.
"""


import matplotlib.pyplot as plt
import numpy as np

import psignifit as ps

data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])

res = ps.psignifit(data, sigmoid_name='norm', experiment_type='2AFC')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot Psychometric Function
# --------------------------
#
# This funciton plots the fitted psychometric function with the measured data.
#  It takes the result dict you want to plot. You can also set plotting options.
#
# 'dataColor': np.array([0,round(105/255,3),round(170/255,3)]),
# 'plotData':    True,
# 'lineColor': np.array([0,0,0],dtype='float'),
# 'lineWidth': 2,
# 'xLabel' : 'Stimulus Level',
# 'yLabel' : 'PercentCorrect',
# 'labelSize' : 15,
# 'fontSize' : 10,
# 'fontName' : 'Helvetica',
# 'tufteAxis' : False,
# 'plotPar' : True,
# 'aspectRatio': False,
# 'extrapolLength': .2,
# 'CIthresh': False

plt.figure()
ps.psigniplot.plot_psych(res, lineWidth=5)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot Marginal Posterior
# -----------------------
#
# This function plots the marginal posterior density for a single parameter.
# As input it requires a results dictionary, the parameter to plot and optionally
# plotting options and a handle to an axis to plot in.
# (As usual 1 = threshold, 2 = width, 3 = lambda, 4 = gamma, 5 = eta)

plt.figure()
ps.psigniplot.plot_marginal(res)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# The gray shadow corresponds to the chosen confidence interval and the black
# line shows the point estimate for the plotted parameter.
# The prior is also included in the plot as a gray dashed line.
#
# You may set the following options again with their
#  respective default values assigned to change the behaviour of the plot:
#
# ::
#     'dim' = 0
#     'lineColor' = [0,round(105/255,3),round(170/255,3)]      # color of the density
#     'lineWidth'      = 2                   # width of the plotline
#     'xLabel'         = '[parameter name] '   # X-Axis label
#     'yLabel'         = 'Marginal Density'  # Y-Axis label
#     'labelSize'      = 15                  # Font size for the label
#     'tufteAxis'      = False               # custom axis drawing enabled
#     'prior'          = True;               # include the prior as a dashed weak line
#     'priorColor'     = [.7,.7,.7]          # color of the prior distibution
#     'CIpatch'        = True                # draw the patch for the confidence interval
#     'plotPE'         = True                # plot the point estimate?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot Posterior in 2-D
# ---------------------
#
# This plots 2 dimensional posterior marginals.
# As input this function expects the result dict, two numbers for the two parameters
# to plot against each other and optionally a handle h to the axis to plot in
# and plotting options.

plt.figure()
ps.psigniplot.plot2D(res, 0, 1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# As options the following fields in plotOptions can be set:
# 'axisHandle'  = plt.gca()    # axes handle to plot in
# 'colorMap'  = getColorMap()         # A colormap for the posterior
# 'labelSize' = 15                   # FontSize for the labels
# 'fontSize'  = 10                   # FontSize for the ticks
# 'label1'    = '[parameter name]'   # label for the first parameter
# 'label2'    = '[parameter name]'   # label for the second parameter

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot Priors
# -----------
# As a tool this function plots the actually used priors of the provided
# result dictionary.

plt.figure()
ps.psigniplot.plot_prior(res)
