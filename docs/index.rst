.. psignifit documentation master file

Welcome to psignifit's documentation!
=====================================

This wiki hosts a manual and comments on our python clone of our MATLAB
toolbox `psignifit
4 <https://github.com/wichmann-lab/psignifit/wiki>`__.

This code was tested against the MATLAB version and yields the same
results up to numerical accuracy of the optimization algorithms for all
fits and credible intervals (see :ref:`test_matlab` for details on our tests).
However, this version is considerably less
tested than the original MATLAB version.

The python version exclusively supports python 3. In this we follow the
whole numpy/scipy etc. environment, which is scheduled to stop
supporting python 2 soon.

Where to start?
~~~~~~~~~~~~~~~

First, :ref:`install_guide` the toolbox.

Then, check out the :ref:`getting_started`
or the :ref:`demo_index` files.

More information
~~~~~~~~~~~~~~~~

The `wiki on the MATLAB version <https://github.com/wichmann-lab/psignifit/wiki>`__
covers a broader discussion on how to apply our toolbox. Anything said there
applies equally to the python version.

A paper describing our method in detail and showing tests for the
congruency of our method is published at *Vision Research*: `Painfree
and accurate Bayesian estimation of psychometric functions for
(potentially) overdispersed
data <http://www.sciencedirect.com/science/article/pii/S0042698916000390>`__
by `Heiko H.
Schütt <http://www.nip.uni-tuebingen.de/people/members.html>`__, `Stefan
Harmeling <http://www.cs.hhu.de/lehrstuehle-und-arbeitsgruppen/computer-vision-computer-graphics-and-pattern-recognition/unser-team/team/harmeling.html>`__,
`Jakob H. Macke <http://www.mackelab.org/people/>`__, and `Felix A.
Wichmann <http://www.nip.uni-tuebingen.de/people/members.html>`__.

If you require additional information, or to report errors or questions
please contact us: heiko.schuett@uni-tuebingen.de or
felix.wichmann@uni-tuebingen.de

Acknowledgements
~~~~~~~~~~~~~~~~

First, we would like to thank Sophie Laturnus and Ole Fortmann who
programmed this python clone based on our MATLAB m-files.

Also, we would like to thank previous members of the Wichmann-lab who
were involved in developing MCMC based methods of Bayesian inference for
the psychometric function, most notably Ingo Fründ, Valentin Hänel,
Frank Jäkel and Malte Kuss.

Furthermore, our thanks go to the reviewers of our manuscript and the
students and colleagues who read the paper or tested the software and
provided feedback: Nicole Eichert, Frank Jäkel, David Janssen, Britta
Lewke, Lars Rothkegel, Joshua Solomon, Tom Wallis, Uli Wannek, Christian
Wolf and Bei Xiao

Contents
~~~~~~~~
.. toctree::
   :maxdepth: 2

   install_guide.rst
   getting_started.rst
   user_guides/index.rst
   how_to/index.rst
   generated_examples/index.rst
   test_MATLAB_version.rst
   references/index.rst


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
