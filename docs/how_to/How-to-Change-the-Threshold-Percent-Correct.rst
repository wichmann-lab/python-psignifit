.. WARNING::
   This documentation page is still work in progress! Some information might be outdated.
   
.. _how-to-change-threshold-percent-correct:

How to: Change the Threshold Percent Correct
============================================

Changing the percent correct defining the threshold works as it does in
MATLAB, see
`here <https://github.com/wichmann-lab/psignifit/wiki/How-to-Change-the-Threshold-Percent-Correct>`__.

The only difference is that a dictionary in python is accessed slightly
different from a struct in MATLAB. Instead of the code displayed for
MATLAB you should use commands of the following syntax to change fields
in the options:

::

   options['threshPC']   = 0.9
