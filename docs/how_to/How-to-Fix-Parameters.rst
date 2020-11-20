.. _how-to-fix-parameters:

How to: Fix Parameters
======================

Fixing parameters is as easy as in the MATLAB version (see
`here <https://github.com/wichmann-lab/psignifit/wiki/How-to-Fix-Parameters>`__).
Just remember that python has all indices shifted compared to MATLAB
starting at 0. Thus the following example code fixes eta, the
overdispersion parameter, **not** gamma.

::

   options['fixedPars'] = np.ones([5,1])*np.nan
   options['fixedPars'][4] = 0
