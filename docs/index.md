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

## A brief history of *psignifit*

The development of *psignifit* (Psychometric SIGNIficant FITting) began in the mid 1990's by Felix Wichmann when
he worked for his doctoral degree at the University of Oxford.

The first version (*psignifit* `1`) was programmed in Mathematica, but was soon ported
to MATLAB. Shortly thereafter Jeremy Hill joined the development and wrote
the C code (MEX files), executing the numerically intensive parts of the code
and thereby speeding up *psignifit* `2` enormously. This version implemented
maximum likelihood (ML) parameter estimation and a parametric bootstrap for
confidence intervals, based on two papers by Felix Wichmann and Jeremy Hill:

  - Wichmann, F. A. and Hill, N. J. (2001). [The psychometric function: I. Fitting, sampling and goodness-of-fit](https://link.springer.com/article/10.3758/BF03194544). *Perception and Psychophysics*, 63(8), 1293-1313.
  - Wichmann, F. A. and Hill, N. J. (2001). [The psychometric function: II. Bootstrap-based confidence intervals and sampling](https://link.springer.com/article/10.3758/BF03194545). *Perception and Psychophysics*, 63(8), 1314-1329.

A copy of the original website and code is still [available](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/neuronale-informationsverarbeitung/research/software/psignifit-legacy/).

*psignifit* `3` was developed by Ingo Fründ, Valentin Hänel and Felix Wichmann at the
TU Berlin between 2009 and 2012 using a mix of Python and C/C++, inspired by Malte Kuss, Frank Jäkel and
Felix Wichmann's Bayesian psychometric function fitting package PsychoFun written in R:

  - Kuss, M., Jäkel, F. and Wichmann, F. A. (2005). [Bayesian inference for psychometric functions](http://jov.arvojournals.org/article.aspx?articleid=2192844). *Journal of Vision*, 5, 478-492
  - Fründ, I., Haenel, N. V. and Wichmann, F.A. (2011). [Inference for psychometric functions in the presence of nonstationary behavior](http://jov.arvojournals.org/article.aspx?articleid=2121082). *Journal of Vision*, 11(6:16), 1-19.

The goal of the re-write was twofold: First, to move from ML parameter
estimation and bootstrapping to Bayesian inference. Second, to move from MATLAB
to Python as programming language. However, it is important to note that the
MCMC sampling used for Bayesian inference in *psignifit* `3` is not always reliable,
and further development of *psignifit* `3` was stopped. The original website is still
[online](https://psignifit.sourceforge.net/) though.


For the latest *psignifit* `4` version, development was shifted back to MATLAB.
Heiko Schütt implemented Bayesian inference based on numerical integration rather than MCMC
sampling, helped by Stefan Harmeling and Felix Wichmann as well as statistical
support by Jakob Macke. Numerical integration avoids the reliability issues
plaguing *psignifit* `3`. Furthermore, *psignifit* `4` is fast enough to be used without
MEX files. These two changes–avoiding the pain of MCMC chain diagnosis and
the pain of MEX file compilation–are alluded to in the title of the paper
describing the method:

- Schütt, H. H., Harmeling, S., Macke, J. H. and Wichmann, F. A. (2016). [Painfree and accurate Bayesian estimation of psychometric functions for (potentially) overdispersed data](http://www.sciencedirect.com/science/article/pii/S0042698916000390). *Vision Research* 122, 105-123.

A final important statistical development of *psignifit* `4` was the move from
a purely binomial to a beta-binomial model, allowing the estimation of 
overdispersion, i.e. of the degree to which human (or animal) observers are more "noisy"
than an idealised binomial decision maker. The original *psignifit* `4` [MATLAB implementation](https://github.com/wichmann-lab/psignifit)
is available on [GitHub](https://github.com/wichmann-lab/psignifit).

The first implementation of *psignifit* `4` in Python started in 2016 as a pure
Python-clone of the original MATLAB library, implemented by Sophie Laturnus, Ole Fortmann
and Heiko Schütt. In 2017 Tiziano Zito started a complete rewrite, with the intent
of making the code more "pythonic" and making use of the SciPy statistical libraries switching
away from self-written ones. The rewrite was taken over by David-Elias Künstle in 2020, who
did most of the hard work of refactoring and implementing the missing functionality. Künstle's
and Zito's work lived for a long time unfinished in a `refactor-api` branch on the
[public GitHub repo](https://github.com/wichmann-lab/python-psignifit) without seeing the light
of the day.

In 2024 David-Elias Künstle, Guillermo Aguilar, Lisa Schwetlick, Pietro Berkes
and Tiziano Zito got together at a programming sprint and with a few bits of advise from Heiko
Schütt and Felix Wichmann after a couple of days of sweat and tears finally published the first
official release of the completely rewritten [implementation](https://github.com/wichmann-lab/python-psignifit/releases/tag/v4.1)
of the [*Vision Research* paper](http://www.sciencedirect.com/science/article/pii/S0042698916000390).

Since then the *psignifit* Python library is maintained by several [contributors](https://github.com/wichmann-lab/python-psignifit/blob/main/CONTRIBUTORS). 

## Acknowledgements

Our thanks go to the reviewers of our manuscript and the
students and colleagues who read the paper or tested the software and
provided feedback: Nicole Eichert, Frank Jäkel, David Janssen, Britta
Lewke, Lars Rothkegel, Joshua Solomon, Tom Wallis, Uli Wannek, Christian
Wolf and Bei Xiao.
