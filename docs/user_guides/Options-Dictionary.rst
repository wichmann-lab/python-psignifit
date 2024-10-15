.. warning::
   This documentation page is still work in progress! Some information might be outdated.

.. _options-dictionary:

Options Dictionary
==================
The configuration/options can be passed directly as dictionary of parameters to the psignifit function.

***

**options['sigmoid']    = 'norm'**
This sets the type of sigmoid you fit to your data.

The default value _'norm'_ fits a cumulative Gaussian to your data.

    options['sigmoid']    = 'norm'

Another standard alternative is the logistic function.

    options['sigmoid']    = 'logistic'


We also included the Gumbel and reversed Gumbel functions for asymmetric psychometric functions. The Gumbel has a longer lower tail the reversed Gumbel a longer upper tail.  

    options['sigmoid']    = 'gumbel';
    options['sigmoid']    = 'rgumbel';

Note that the `weibull` function is defined as `gumbel`, except with log-scaled data. If you invoke

    options['sigmoid']    = 'weibull

it will be necessary to manually convert your data to log-space before fitting.
***

For passing a custom sigmoid, simply pass a custom sigmoid as the `options['sigmoid']` parameter. The sigmoid could be implemented as a class that inherits from the `Sigmoid` class in the `sigmoids` module. Please refer to the `sigmoids` module for inspiration on how to implement a custom sigmoid.

***

**options['experiment_type'] = 'yes/no'**  
This sets the type of experiment you are conducting. Please refer to the `experiment-types` section for more information.


***

**options['experiment_type'] = 'MAP'**  
which point estimate you want from the analysis

_'MAP'_ The MAP estimator is the maximum a posteriori computed from the posterior.

    options['experiment_type'] = 'MAP';

_'mean'_ The posterior mean. For strict Bayesians the best estimate: the expected value of the Posterior. However, in our paper we show that for psychometric function fitting and the small datasets used in experimental psychology and the neurosciences, and for the our priors the MAP is preferable as default!

    options['experiment_type'] = 'mean';

***
# TODO:
**options.stepN   = [40,40,20,20,20]**  
**options.mbStepN = [25,20,10,10,20]**  
This sets the number of grid points on each dimension in the final fitting (_stepN_) and in the moving of borders (_mbStepN_). The order is _[threshold,width,upper asymptote,lower asymptote,variance scaling]_

You may change this if you need more accurate estimates on the sparsely sampled parameters.
For example to get an more exact estimate on the lapse rate/upper asymptote plug in: 

    options.stepN=[40,40,50,20,20];  

Now the lapse rate is sampled at 50 places giving you a much more exact and smooth curve for comparisons.

***

**options['confP']          = .95**  
The confidence level for the computed confidence intervals. This may be set to any number between 0 and 1 excluding.

For example to get 99% confidence intervals try:  

    options['confP']          = .99;

You may specify a vector as well. If you do, the _conf_intervals_ in the result will be a 5x2xN array containing the values for the different confidence levels in the 3rd dimension. 

    options['confP'] = [.95,.9,.68,.5];

will return 4 confidence intervals for each parameter corresponding to the 95%, 90%, 68% and 50% credible intervals.

***

**options['CI_method']       ='percentiles'**
This sets how the confidence intervals are computed in _getConfRegion.m_. Possible variants are:
       _'project'_ -> project the confidence region on each axis
   _'percentiles'_ -> find alpha/2 and 1-alpha/2 percentiles
                    (alpha = 1-confP)

***

**options['thresh_PC']       = .5**
Which percent correct correspond to the threshold? Given in percent correct on the unscaled sigmoid (reaching from 0 to 1):

For example to define the threshold as 90% correct try:

    options['thresh_PC']        = .9;

For details have a look at :ref:`How to Change the Threshold Percent Correct <how_to/How-to-Change-the-Threshold-Percent-Correct>`.
# TODO: fix this link
***

**options['priors']  = dict**
# TODO: is this correct?
This field contains a dictionary of function handles, which define the priors for each parameter.
If you want to set your priors manually, here is the place for it.
#TODO: fix the link
For details on how to change these refer to :ref:`Priors <priors>`.

***

**options['beta_prior']      = 10**
This sets the strength of the Prior in favour of a binomial observer. Larger values correspond to a stronger prior. We choose this value after a rather large number of simulations. Refer to [Priors](Priors) to learn more about this.
#TODO: fix the link
***

# TODO: these do not exist...
**options.nblocks        = 35;**  
**options.poolMaxGap     = inf;**  
**options.poolMaxLength  = inf;**      
**options.poolxTol       = 0;**          
These options set how your data is pooled into blocks. Your data is only pooled if your data Matrix has more than _nblocks_ lines. Then we pool together a maximum of _poolMaxLength_ trials, which are separated by a maximum of _poolMaxGap_ trial of other stimulus levels. If you want you may specify a tolerance in stimulus level _poolxTol_ to pool trials, but by default we only pool trials with exactly the same number of trials.

***

**options['bounds']**
In this field you may provide your own bounds for the parameters. This should be a dictionary with the name of all the parameters as keys and the lower and upper bounds as a tuple as the values.

For example to set the bounds for the threshold to be between 0.1 and 0.9 you would use:

    options['bounds'] = {'threshold': (0.1, 0.9)}
# TODO: does this work???
NOTE: By this you artificially exclude all values out of this range. Only exclude parameter values, which are impossible!


***

**options['max_bound_value'] = exp(-10)**  
Parts of the grid which produce marginal values below this are considered 0 and are excluded from the calculation ; it should be a very small value and at least smaller than 1/(max(stepN)).

This for example would exclude fewer values and more conservative movement of the borders:  

    options['max_bound_value'] = exp(-20)

***

**options['move_bounds']    = 1**
# TODO: what does this do?
toggles the movement of borders by _moveBorders.m_. Usually this is good to concentrate on the right area in the parameter space.  

    options['move_bounds']     = 1

If you set  

    options['move_bounds']     = 0

your posterior will always use the initial setting for the borders. This is useful if you set options['bounds'] by hand and do not want psignifit to move them after this.

***


**options['width_alpha']     = .05**  
This changes how the width of a psychometric function is defined _width= psi^(-1)(1-alpha) - psi^(-1)(alpha)_
where _psi^(-1)_ is the inverse of the sigmoid function. _width_alpha_ must be between 0 and .5 excluding

Thus this would enable the usage of the interval from .1 to .9 as the width for example:

    options['width_alpha']     = .1;

***


Refer  to :ref:`Demo 2 <sphx_glr_generated_examples_demo_002.py>` for information on the meaning of individual fields.
