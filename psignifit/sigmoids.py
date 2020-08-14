import numpy as np

from .utils import normcdf as _normcdf
from .utils import norminv as _norminv
from .utils import norminvg as _norminvg
from .utils import t1cdf as _t1cdf
from .utils import t1icdf as _t1icdf


class Sigmoid:
    """ Base class for sigmoid implementation.

    Handels logarithmic input and negative output
    for the specific sigmoid implementations.

    Sigmoid classes should derive from this class and implement
    the methods '_value', '_slope', and '_threshold'.

    The stimulus levels, threshold and width are parameters of method calls.
    They correspond to the object attributes PC and alpha in the following way:

    threshold: threshold is the stimulus level at which the sigmoid has value PC (float)
         psi(m) = PC , typically PC=0.5
    width: the difference of stimulus levels where the sigmoid has value alpha and 1-alpha
         width = X_(1-alpha) - X_(alpha)
         psi(X_(1-alpha)) = 0.95 = 1-alpha
         psi(X_(alpha)) = 0.05 = alpha
    """
    logspace = False
    negate = False

    def __init__(self, PC=0.5, alpha=0.05, negative=False, logspace=False):
        self.PC = PC
        self.alpha = alpha
        self.negative = negative
        self.logspace = logspace

    def __call__(self, stimulus_level, threshold, width):
        if self.logspace:
            stimulus_level = np.log(stimulus_level)
        value = self._value(stimulus_level, threshold, width)

        if self.negative:
            return 1 - value
        else:
            return value

    def _value(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray):
        """ Calculate the sigmoid value at specified stimulus levels. """
        raise NotImplementedError("This should be overwritten by an implementation.")

class Gaussian(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = width / (_norminv(1 - self.alpha) - _norminv(self.alpha))
        return _normcdf(stimulus_level, (threshold - _norminvg(self.PC, 0, C)), C)


class Logistic(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        return 1 / (1 + np.exp(-2 * np.log(1 / self.alpha - 1) / width * (stimulus_level - threshold)
                               + np.log(1 / self.PC - 1)))

class Gumbel(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return 1 - np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(1 - self.PC))))


class ReverseGumbel(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(self.PC))))


class Student(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = (_t1icdf(1 - self.alpha) - _t1icdf(self.alpha))
        return _t1cdf(C * (stimulus_level - threshold) / width + _t1icdf(self.PC))


_CLASS_BY_NAME = {
    'norm': Gaussian,
    'gauss': Gaussian,
    'logistic': Logistic,
    'gumbel': Gumbel,
    'rgumbel': ReverseGumbel,
    'tdist': Student,
    'student': Student,
    'heavytail': Student,
    'weibull': Gumbel,
    'logn': Gaussian,
}

_LOGSPACE_NAMES = [
    'weibull',
    'logn'
]

ALL_SIGMOID_NAMES = set(_CLASS_BY_NAME.keys())
ALL_SIGMOID_NAMES |= { 'neg_' + name for name in ALL_SIGMOID_NAMES }

def sigmoid_by_name(name, PC=None, alpha=None):
    kwargs = dict()
    name = name.lower().strip()
    if PC is not None:
        kwargs['PC'] = PC
    if alpha is not None:
        kwargs['alpha'] = alpha
    if name.startswith('neg_'):
        name = name[4:]
        kwargs['negative'] = True
    if name in _LOGSPACE_NAMES:
        kwargs['logspace'] = True

    return _CLASS_BY_NAME[name](**kwargs)

