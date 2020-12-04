.. _result-struct:

Result Struct
=============

The result of your computations in python is a object mirroring the
struct in MATLAB the same way the options dictionary mirrors the options
struct.

Thus entries can be accessed as follows:

::

   res.field_name


Consult `print(res)`,  :ref:`Demo 3 <sphx_glr_generated_examples_demo_003.py>`, or :class:`psignifit.Result`
for a full list of fields and the meaning of the different entries.
