""" All sigmoid functions.

If you add a new sigmoid type, add it to the CLASS_BY_NAME constant
and to the _LOGSPACE_NAMES, if it expects stimulus on an exponential scale.
"""
import numpy as np
from typing import Optional

from .utils import normcdf as normcdf
from .utils import norminv as norminv
from .utils import norminvg as norminvg
from .utils import t1cdf as t1cdf
from .utils import t1icdf as t1icdf


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
        """ Calculate the sigmoid value at specified stimulus levels. """
        if self.logspace:
            stimulus_level = np.log(stimulus_level)
        value = self._value(stimulus_level, threshold, width)

        if self.negative:
            return 1 - value
        else:
            return value

    def inverse(self, perc_correct: np.ndarray, threshold: float, width: float,
                gamma: Optional[float] = None, lambd: Optional[float] = None) -> np.ndarray:
        """ Finds the stimulus value for given parameters at different percent correct.

        See Sigmoid class description for a discussion of the parameters.

        Args:
            perc_correct: Percentage correct at the threshold to calculate.
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
            gamma: Parameter value for the lower offset of the sigmoid
            lambd: Parameter value for the upper offset of the sigmoid
        Returns:
            Threshold at the percentage correct values.
        """
        perc_correct = np.asarray(perc_correct)
        if lambd is not None and gamma is not None:
            if (perc_correct < gamma).any() or (perc_correct > (1 - lambd)).any():
                raise ValueError(f'perc_correct={perc_correct} has to be between {gamma} and {1 - lambd}.')
            perc_correct = (perc_correct - gamma) / (1 - lambd - gamma)
        PC = self.PC
        if self.negative:
            PC = 1 - PC
            perc_correct = 1 - perc_correct

        result = self._inverse(perc_correct, threshold, width, PC)
        if self.logspace:
            return np.exp(result)
        else:
            return result

    def _value(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray, PC: float) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")


class Gaussian(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = width / (norminv(1 - self.alpha) - norminv(self.alpha))
        return normcdf(stimulus_level, (threshold - norminvg(self.PC, 0, C)), C)

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray, PC: float) -> np.ndarray:
        C = norminv(1 - self.alpha) - norminv(self.alpha)
        return norminvg(perc_correct, threshold - norminvg(PC, 0, width / C), width / C)


class Logistic(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        return 1 / (1 + np.exp(-2 * np.log(1 / self.alpha - 1) / width * (stimulus_level - threshold)
                               + np.log(1 / self.PC - 1)))

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray, PC: float) -> np.ndarray:
        return threshold - width * ( np.log(1 / perc_correct - 1) - np.log(1 / PC - 1)) / 2 / np.log(1 / self.alpha - 1)

class Gumbel(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return 1 - np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(1 - self.PC))))

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray, PC: float) -> np.ndarray:
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return threshold + (np.log(-np.log(1 - perc_correct)) - np.log(-np.log(1 - PC))) * width / C

class ReverseGumbel(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(self.PC))))

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray, PC: float) -> np.ndarray:
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return threshold + (np.log(-np.log(perc_correct)) - np.log(-np.log(PC))) * width / C

class Student(Sigmoid):
    def _value(self, stimulus_level, threshold, width):
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        return t1cdf(C * (stimulus_level - threshold) / width + t1icdf(self.PC))

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray, PC: float) -> np.ndarray:
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        return (t1icdf(perc_correct) - t1icdf(PC)) * width / C + threshold

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

