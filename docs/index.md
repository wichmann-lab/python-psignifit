# Welcome to psignifit\'s documentation!

*Psignifit 4* is a python toolbox for Bayesian psychometric function estimation.

This documentation contains a manual and usage guides on our python equivalent of
[psignifit 4](https://github.com/wichmann-lab/psignifit/wiki), originally
implemented in MATLAB. This code was tested against the MATLAB version and yields the same
results up to numerical accuracy of the optimization algorithms for all
fits and credible intervals. 

In addition, this python implementation contains extensive [tests](https://github.com/wichmann-lab/python-psignifit/tree/main/psignifit/tests), 
including [parameter recovery tests](https://github.com/wichmann-lab/python-psignifit/blob/main/psignifit/tests/test_param_recovery.py),
which ensure that the fitting procedure is correctly implemented.
This is important, as psignifit external dependencies (`scipy`, `numpy`)
might change in the future. 
If these changes break our *psignifit* implementation, 
these tests will detect it automatically,
[as we run them continuously](https://github.com/wichmann-lab/python-psignifit/actions).


## Where to start?

First, [install](install_guide) the package. 

Then, check out the [basic usage](basic-usage) page.

## More information


A paper describing our method in detail and showing tests for the
congruency of our method is published at *Vision Research*: [Painfree
and accurate Bayesian estimation of psychometric functions for
(potentially) overdispersed
data](http://www.sciencedirect.com/science/article/pii/S0042698916000390)
by [Heiko H.
Schütt](http://www.nip.uni-tuebingen.de/people/members.html), [Stefan
Harmeling](http://www.cs.hhu.de/lehrstuehle-und-arbeitsgruppen/computer-vision-computer-graphics-and-pattern-recognition/unser-team/team/harmeling.html),
[Jakob H. Macke](http://www.mackelab.org/people/), and [Felix A.
Wichmann](http://www.nip.uni-tuebingen.de/people/members.html).

If you require additional information, or to report errors or questions,
please [open an issue in github](https://github.com/wichmann-lab/python-psignifit/issues)
or contact us: <heiko.schuett@uni-tuebingen.de> or
<felix.wichmann@uni-tuebingen.de>

## Acknowledgements

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
