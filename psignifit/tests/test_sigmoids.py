import numpy as np
import pytest

import psignifit.sigmoids as sg


# fixed parameters for simple sigmoid sanity checks
X = np.linspace(1e-12, 1-1e-12, num=10000)
M = 0.5
WIDTH = 0.9
PC = 0.5
ALPHA = 0.05

# with the above parameters, we expect:
# threshold = 0.5
# width = 0.9
# gamma = 0
# lambda = 1
# eta = 0

# list of all sigmoids (after having removed aliases)
ALL_SIGS = { getattr(sg, name) for name in dir(sg) if not name.startswith('_') }

LOG_SIGS = ('weibull', 'logn', 'neg_weibull', 'neg_logn')

@pytest.mark.parametrize('sigmoid', ALL_SIGS)
def test_sigmoid_sanity_check(sigmoid):
    #if sigmoid.__name__ in LOG_SIGS:
    #    x = np.log(X)
    #else:
    #    x = X
    x = X
    out = sigmoid(x, M, WIDTH, PC=PC, alpha=ALPHA)
    #print(sigmoid.__name__, out.min(), out.max())
    assert np.isclose(sigmoid(M, M, WIDTH, PC=PC, alpha=ALPHA), PC)

