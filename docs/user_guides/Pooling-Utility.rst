.. _pooling-utility:

Pooling Utility
===============

Pooling works the same way in python as it does in
`MATLAB <https://github.com/wichmann-lab/psignifit/wiki/Pooling-Utility>`__.
The corresponding fields can be set in python with commands using the
following syntax:

::

   options['nblocks']        = 25;      # number of blocks required to start pooling
   options['poolMaxGap']     = np.inf;  # maximal number of trials with other stimulus levels between pooled trials
   options['poolMaxLength']  = np.inf;  # maximal number of trials per block
   options['poolxTol']       = 0;       # maximal difference in stimulus level from the first trial in a block 
