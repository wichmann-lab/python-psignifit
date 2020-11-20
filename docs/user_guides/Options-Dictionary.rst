.. _options-dictionary:

Options Dictionary
==================


The `options struct <https://github.com/wichmann-lab/psignifit/wiki/Options-Struct>`__
from MATLAB was implemented as a dictionary in python. Thus to set any
options in the python version use the following syntax:

::

   options = dict()              % Initialization as an empty dictionary
   options['fieldName'] = value  % Setting individual values

The options available have the same name and function as the ones in the
MATLAB version. Refer to the MATLAB wiki
`here <https://github.com/wichmann-lab/psignifit/wiki/Options-Struct>`__
or to demo_002 for information on the meaning of individual fields.
