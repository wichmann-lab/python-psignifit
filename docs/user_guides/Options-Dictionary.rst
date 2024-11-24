WARNING: OLD DOCUMENTATION

The options that are still not well described are left here
to be double checked!



Options Dictionary
==================

# TODO:
**options.stepN   = [40,40,20,20,20]**  
**options.mbStepN = [25,20,10,10,20]**  
This sets the number of grid points on each dimension in the final fitting (_stepN_) and in the moving of borders (_mbStepN_). The order is _[threshold,width,upper asymptote,lower asymptote,variance scaling]_

You may change this if you need more accurate estimates on the sparsely sampled parameters.
For example to get an more exact estimate on the lapse rate/upper asymptote plug in: 

    options.stepN=[40,40,50,20,20];  

Now the lapse rate is sampled at 50 places giving you a much more exact and smooth curve for comparisons.

***


**options['thresh_PC']       = .5**
Which percent correct correspond to the threshold? Given in percent correct on the unscaled sigmoid (reaching from 0 to 1):

For example to define the threshold as 90% correct try:

    options['thresh_PC']        = .9;



**options['priors']  = dict**
# TODO: is this correct?
This field contains a dictionary of function handles, which define the priors for each parameter.
If you want to set your priors manually, here is the place for it.
#TODO: fix the link
For details on how to change these refer to :ref:`Priors <priors>`.

***


# TODO: these do not exist...
**options.nblocks        = 35;**  
**options.poolMaxGap     = inf;**  
**options.poolMaxLength  = inf;**      
**options.poolxTol       = 0;**          
These options set how your data is pooled into blocks. Your data is only pooled if your data Matrix has more than _nblocks_ lines. Then we pool together a maximum of _poolMaxLength_ trials, which are separated by a maximum of _poolMaxGap_ trial of other stimulus levels. If you want you may specify a tolerance in stimulus level _poolxTol_ to pool trials, but by default we only pool trials with exactly the same number of trials.

***



**options['width_alpha']     = .05**  
This changes how the width of a psychometric function is defined _width= psi^(-1)(1-alpha) - psi^(-1)(alpha)_
where _psi^(-1)_ is the inverse of the sigmoid function. _width_alpha_ must be between 0 and .5 excluding

Thus this would enable the usage of the interval from .1 to .9 as the width for example:

    options['width_alpha']     = .1;

***

