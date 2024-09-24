.. _how-to-fix-parameters:

How to: Fix Parameters
======================

Fixing Parameters is done by simply adding a dictionary to the options using the key 'fixed_parameters'.
In the following example we fix the parameters lambda and gamma to 0.02 and 0.5 respectively.

::

   options['fixed_parameters'] = {'lambda': 0.02, 'gamma': 0.5}
