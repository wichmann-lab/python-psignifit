.. _experiment-types:

Experiment Types
================

Setting experiment types works as it does in MATLAB, see
`here <https://github.com/wichmann-lab/psignifit/wiki/Experiment-Types>`__.

The only difference is the naming of the experiment types and that they can be passed
as parameters of the psignifit function.
Find an interactive example in :ref:`Demo 1 <sphx_glr_generated_examples_demo_001.py>`.

Apart from the *nAFC* (2AFC, 3AFC, … ) we provide two other options
for *expType*: First *‘YesNo’* which enables a free upper and lower
Asymptote and, second, *‘equalAsymptote’*, which assumes that the upper
and the lower asymptote are equal. You find a more detailed description
of the types :ref:`here <experiment-types>`.

::

   experiment_type = '2AFC'
   experiment_type = '5AFC'
   experiment_type = 'yes/no'
   experiment_type = 'equal asymptote'
