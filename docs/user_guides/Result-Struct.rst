.. _result-struct:
Result Struct
=============

The result of your computations in python is a dictionary mirroring the
struct in MATLAB the same way the options dictionary mirrors the options
struct.

Thus entries can be accessed as follows:

::

   res['fieldName']

Consult res.keys(), demo_003 or the MATLAB
`wiki <https://github.com/wichmann-lab/psignifit/wiki/Result-Struct>`__
for a full list of fields and the meaning of the different entries.
