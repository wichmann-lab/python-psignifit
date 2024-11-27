.. warning::
   This documentation page is still work in progress! Some information might be outdated.

.. _priors:

Priors
======

Priors are discussed extensively for the MATLAB version
`here <https://github.com/wichmann-lab/psignifit/wiki/Priors>`__ and all
settings work described for MATLAB work in python as well.

The relevant fields of the options dictionary are:

::

   options['stimulus_range'] = np.array([a,b])  # set the range used for setting the prior to [a,b]
   options['beta_prior'] = k                    # set the strength of the prior on the beta-binomial to k
   options['borders'] = {'threshold': (np.nan, np.nan)}  # setting fixed borders for threshold parameter


Also the python version has the plotPrior function to show the used
prior on at least the first 3 parameters. It is available using the
following command:

::

   ps.psigniplot.plotPrior(res)

To pass arbitrary hand chosen priors in python first remember that “you
should know what you are doing, and everything is on your own risk” as
stated in the MATLAB wiki. To pass a function as a prior use the
following syntax:

::

   options['priors'] = [list of priors];

where your priors may be any lambda or normal functions. This is simpler
than in MATLAB as python allows functions to be passed around as objects
removing the need to use handles as in MATLAB.
