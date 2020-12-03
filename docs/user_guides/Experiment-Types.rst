.. _experiment-types:

Experiment Types
================

Setting experiment types works as it does in MATLAB, see
`here <https://github.com/wichmann-lab/psignifit/wiki/Experiment-Types>`__.

The only difference is that a dictionary in python is accessed slightly
different from a struct in MATLAB. Instead of the code displayed for
MATLAB you should use commands of the following syntax to change fields
in the options:

::

   options['expType']   = '2AFC'
   options['expType']   = 'YesNo'
   options['expType']   = 'equalAsymptote'
