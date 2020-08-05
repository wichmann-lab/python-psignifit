import numpy as np
import pytest

import psignifit.sigmoids as sg


# fixed parameters for simple sigmoid sanity checks
X = np.linspace(0, 1, num=15)
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


ALL_SIGS = [getattr(sg, name) for name in dir(sg) if not name.startswith('_')]
print(ALL_SIGS)

@pytest.mark.parametrize('sigmoid', ALL_SIGS)
def test_sigmoid_sanity_check(sigmoid):
    out = sigmoid(X, M, WIDTH, PC=PC, alpha=ALPHA)


