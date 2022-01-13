.. _options-dictionary:

Options Dictionary
==================


Instead of the `options struct <https://github.com/wichmann-lab/psignifit/wiki/Options-Struct>`__
from MATLAB, the configuration can be passed directly as parameters to the psignifit function.
In addition, they could be specified as a dictionary as in the following:

::

   config = dict()              % Initialization as an empty dictionary
   config['fieldName'] = value  % Setting individual values
   result = ps.psignifit(data, **config)

The options available have the same name and function as the ones in the
MATLAB version. Refer to the MATLAB wiki
`here <https://github.com/wichmann-lab/psignifit/wiki/Options-Struct>`__
or to :ref:`Demo 2 <sphx_glr_generated_examples_demo_002.py>`
for information on the meaning of individual fields.
