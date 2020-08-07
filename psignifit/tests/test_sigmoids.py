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
LOG_SIGS = ('weibull', 'logn', 'neg_weibull', 'neg_logn')
ALL_SIGS = [ (getattr(sg, name), name in LOG_SIGS, 'neg' in name)
             for name in dir(sg) if not name.startswith('_')]


@pytest.mark.parametrize('sigmoid,is_log,is_neg', ALL_SIGS)
def test_sigmoid_sanity_check(sigmoid, is_log, is_neg):
    x_M = M
    x = np.linspace(1e-8, 1, 100)
    if is_neg:
        x = 1 - x
    if is_log:
        x_M = np.exp(x_M)
        x = np.exp(x)

    # sigmoid(M) == PC
    np.testing.assert_allclose(sigmoid(x_M, M, WIDTH, PC=PC, alpha=ALPHA), PC)

    # |X_L - X_R| == WIDTH, with
    # with sigmoid(X_L) == ALPHA
    # and  sigmoid(X_R) == 1 - ALPHA
    s = sigmoid(x, M, WIDTH, PC=PC, alpha=ALPHA)
    idx_alpha, idx_nalpha =  np.abs(s - ALPHA).argmin(), np.abs(s - (1 - ALPHA)).argmin()
    np.testing.assert_allclose(s[idx_nalpha] - s[idx_alpha], WIDTH, atol=0.02)
