.. warning::
   This documentation page is still work in progress! Some information might be outdated.

.. _how-to-change-threshold-percent-correct:

How to: Change the Threshold Percent Correct
============================================


Psignifit usually defines the threshold as the point where the unscaled sigmoid is 0.5, i.e. half-way up its range. Sometimes one wants to calculate thresholds for another percent correct level. To do so psignifit has an entry of the options struct:

Changing the percent correct defining the threshold works as it does in
MATLAB, see
`here <https://github.com/wichmann-lab/psignifit/wiki/How-to-Change-the-Threshold-Percent-Correct>`__.

The only difference is that a dictionary in python is accessed slightly
different from a struct in MATLAB. Instead of the code displayed for
MATLAB you should use commands of the following syntax to change fields
in the options:

::

   options['threshPC']   = 0.9
