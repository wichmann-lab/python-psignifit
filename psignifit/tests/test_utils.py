import numpy as np
from psignifit.utils import normalize

def test_normalize():
    # a constant function
    func = lambda x: np.ones_like(x)
    # the integral is length of x, so the normalized function should return 1/len(x)
    x = np.arange(11)
    norm = normalize(func, (0, 10))
    assert np.allclose(1./10, norm(x))

def test_normalize_sin():
    # for sin the integral in (0, pi/2) is 1, so the norm(sin) == sin
    x = np.linspace(0,np.pi/2,100)
    norm = normalize(np.sin, (0, np.pi/2))
    assert np.allclose(np.sin(x), norm(x))
