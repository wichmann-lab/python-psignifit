# Welcome to psignifit\'s documentation!

*Psignifit* is a Python toolbox for Bayesian psychometric function estimation.

This library is a [Python implementation](https://github.com/wichmann-lab/python-psignifit) of the method described in
detail in the paper published at *Vision Research*:
[Painfree and accurate Bayesian estimation of psychometric functions for (potentially) overdispersed data](http://www.sciencedirect.com/science/article/pii/S0042698916000390)
by [Heiko H. Schütt](http://www.nip.uni-tuebingen.de/people/members.html),
[Stefan Harmeling](http://www.cs.hhu.de/lehrstuehle-und-arbeitsgruppen/computer-vision-computer-graphics-and-pattern-recognition/unser-team/team/harmeling.html),
[Jakob H. Macke](http://www.mackelab.org/people/),
and [Felix A. Wichmann](http://www.nip.uni-tuebingen.de/people/members.html).

The original [MATLAB implementation](https://github.com/wichmann-lab/psignifit/wiki) is also available. 

The psignifit Python library contains extensive [tests](https://github.com/wichmann-lab/python-psignifit/tree/main/tests), 
including [parameter recovery tests](https://github.com/wichmann-lab/python-psignifit/blob/main/tests/test_param_recovery.py),
which ensure that the fitting procedure is correctly implemented.

If you require additional information or to report errors or ask questions
please [open an issue on github](https://github.com/wichmann-lab/python-psignifit/issues).

## Where to start?

First [install](install_guide) the package. 

Then, check out the [basic usage](basic-usage) page.

## Acknowledgements

The first implementation, as Python-clone of the original MATLAB library, was done by Sophie Laturnus and Ole Fortmann. 

The current completely rewritten implementation is the result of a programming sprint in September 2024 and later work. The participants were David-Elias Künstle, Felix Wichmann, Guillermo Aguilar, Lisa Schwetlick, Pietro Berkes and Tiziano Zito. The complete list of [contributors](https://github.com/wichmann-lab/python-psignifit/blob/main/CONTRIBUTORS).

We would like to thank previous members of the Wichmann-lab who
were involved in developing MCMC based methods of Bayesian inference for
the psychometric function, most notably Ingo Fründ, Valentin Hänel,
Frank Jäkel and Malte Kuss.

Furthermore, our thanks go to the reviewers of our manuscript and the
students and colleagues who read the paper or tested the software and
provided feedback: Nicole Eichert, Frank Jäkel, David Janssen, Britta
Lewke, Lars Rothkegel, Joshua Solomon, Tom Wallis, Uli Wannek, Christian
Wolf and Bei Xiao.
