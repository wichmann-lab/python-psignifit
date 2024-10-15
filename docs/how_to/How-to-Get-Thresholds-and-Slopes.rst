.. WARNING::
   This documentation page is still work in progress! Some information might be outdated.
   
.. _how-to-get-thresholds-and-slopes:

How To: Get Thresholds and Slopes
=================================

As we describe in detail for the MATLAB version
`here <https://github.com/wichmann-lab/psignifit/wiki/How-to-Get-Thresholds-and-Slopes>`__
psignifit contains utility functions to extract thresholds for different
percent correct and slopes at given percent correct or stimulus levels.

The python versions these functions are directly loaded when you import
psignifit and can be called in the analogue syntax to MATLAB as follows:

::

   ps.getThreshold(res,percentCorrect)
   ps.getSlope(res,stimLevel)
   ps.getSlopePC(res,percentCorrect)

The only change you may observe is that the getThreshold function always
returns a tuple with the point estimate and confidence intervals. This
could not be avoided as python functions are not told which of their
outputs are used by the calling program. The warnings about these
confidence intervals being a rough worst case approximation still
applies as spelled out in detail for the MATLAB version.

To get only the threshold call:

::

   ps.getThreshold(res,percentCorrect)[0]
