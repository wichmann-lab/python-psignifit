""" All sigmoid functions.

If you add a new sigmoid type, add it to the CLASS_BY_NAME constant
"""
from typing import Optional, TypeVar

import numpy as np
import scipy as sp
import scipy.stats

# Alias common distribution to be reused all over the place.

# - Normal distribution:
#   - This one is useful when we want mean=0, std=1
#     Percent point function -> inverse of cumulative normal distribution function
#     returns percentiles
norminv = scipy.stats.norm(loc=0, scale=1).ppf
#   - also instantiate a generic version
norminvg = scipy.stats.norm.ppf
#   - Cumulative normal distribution function
normcdf = scipy.stats.norm.cdf

# T-Student with df=1
t1cdf = scipy.stats.t(1).cdf
t1icdf = scipy.stats.t(1).ppf

# sigmoid can be calculated on single floats, or on numpy arrays of floats
N = TypeVar('N', float, np.ndarray)


class Sigmoid:
    """ Base class for sigmoid implementation.

    Handles logarithmic input and negative output
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

    def __init__(self, PC=0.5, alpha=0.05, negative=False):
        """
        Args:
             PC: Proportion correct (sigmoid function value) at threshold
             alpha: Scaling parameter
             negative: Flip sigmoid such proportion correct is decreasing.
        """
        self.alpha = alpha
        self.negative = negative
        self.PC = PC
        if negative:
            self._PC = 1 - PC
        else:
            self._PC = self.PC

    def __eq__(self, o: object) -> bool:
        return (isinstance(o, self.__class__)
                and o.PC == self.PC
                and o.alpha == self.alpha
                and o.negative == self.negative)

    def __call__(self, stimulus_level: N, threshold: N, width: N) -> N:
        """ Calculate the sigmoid value at specified stimulus levels.

        Args:
            prop_correct: Proportion correct at the threshold to calculate.
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
            gamma: Parameter value for the lower offset of the sigmoid
            lambd: Parameter value for the upper offset of the sigmoid
        Returns:
            Proportion correct at the stimulus values.
        """
        value = self._value(stimulus_level, threshold, width)

        if self.negative:
            return 1 - value
        else:
            return value

    def slope(self, stimulus_level: N, threshold: N, width: N, gamma: N = 0, lambd: N = 0) -> N:
        """ Calculate the slope at specified stimulus levels.

        Args:
            prop_correct: Proportion correct at the threshold to calculate.
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
            gamma: Parameter value for the lower offset of the sigmoid
            lambd: Parameter value for the upper offset of the sigmoid
        Returns:
            Slope at the stimulus values.
        """

        slope = (1 - gamma - lambd) * self._slope(stimulus_level, threshold, width)

        if self.negative:
            return -slope
        else:
            return slope

    def inverse(self, prop_correct: N, threshold: N, width: N,
                gamma: Optional[N] = None, lambd: Optional[N] = None) -> np.ndarray:
        """ Finds the stimulus value for given parameters at different proportion correct.

        See :class:psignifit.sigmoids.Sigmoid for a discussion of the parameters.

        Args:
            prop_correct: Proportion correct at the threshold to calculate.
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
            gamma: Parameter value for the lower offset of the sigmoid
            lambd: Parameter value for the upper offset of the sigmoid
        Returns:
            Threshold at the proportion correct values.
        """
        prop_correct = np.asarray(prop_correct)
        if lambd is not None and gamma is not None:
            if (prop_correct < gamma).any() or (prop_correct > (1 - lambd)).any():
                raise ValueError(f'prop_correct={prop_correct} has to be between {gamma} and {1 - lambd}.')
            prop_correct = (prop_correct - gamma) / (1 - lambd - gamma)
        if self.negative:
            prop_correct = 1 - prop_correct

        result = self._inverse(prop_correct, threshold, width)
        return result

    def _value(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _inverse(self, prop_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")


class Gaussian(Sigmoid):
    """ Sigmoid based on the Gaussian distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C =(norminv(1 - self.alpha) - norminv(self.alpha))
        return normcdf(stimulus_level, (threshold - norminvg(self._PC, 0,  width / C)),  width / C)

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = norminv(1 - self.alpha) - norminv(self.alpha)
        m = (threshold - norminvg(self._PC, 0, width / C))
        normalized_stimulus_level = (stimulus_level - m) / width * C
        normalized_slope = sp.stats.norm.pdf(normalized_stimulus_level)
        return normalized_slope * C / width

    def _inverse(self, prop_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = norminv(1 - self.alpha) - norminv(self.alpha)
        return norminvg(prop_correct, threshold - norminvg(self._PC, 0, width / C), width / C)

class Logistic(Sigmoid):
    """ Sigmoid based on the Logistic distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        return 1 / (1 + np.exp(-2 * np.log(1 / self.alpha - 1) / width * (stimulus_level - threshold)
                               + np.log(1 / self._PC - 1)))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = 2 * np.log(1 / self.alpha - 1) / width
        unscaled_slope = np.exp(-C * (stimulus_level - threshold) + np.log(1 / self._PC - 1))
        return C * unscaled_slope / (1 + unscaled_slope)**2

    def _inverse(self, prop_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        return (threshold - width * (np.log(1 / prop_correct - 1) - np.log(1 / self._PC - 1)) / 2
                / np.log(1 / self.alpha - 1))


class Gumbel(Sigmoid):
    """ Sigmoid based on the Gumbel distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return 1 - np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(1 - self._PC))))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        unscaled_stimulus_level = np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(1 - self._PC)))
        return C / width * np.exp(-unscaled_stimulus_level) * unscaled_stimulus_level

    def _inverse(self, prop_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return threshold + (np.log(-np.log(1 - prop_correct)) - np.log(-np.log(1 - self._PC))) * width / C


class ReverseGumbel(Sigmoid):
    """ Sigmoid based on the reversed Gumbel distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(self._PC))))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        unscaled_stimulus_level = np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(self._PC)))
        return -C / width * np.exp(-unscaled_stimulus_level) * unscaled_stimulus_level

    def _inverse(self, prop_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return threshold + (np.log(-np.log(prop_correct)) - np.log(-np.log(self._PC))) * width / C


class Student(Sigmoid):
    """ Sigmoid based on the Student-t distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        return t1cdf(C * (stimulus_level - threshold) / width + t1icdf(self._PC))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        stimLevel = (stimulus_level - threshold) * C / width + t1icdf(self._PC)
        return C / width * sp.stats.t.pdf(stimLevel, df=1)

    def _inverse(self, prop_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        return (t1icdf(prop_correct) - t1icdf(self._PC)) * width / C + threshold


class Weibull(Gumbel):
    """ Sigmoid based on the Weibull function.

    IMPORTANT: All the sigmoids in `psignifit` work in linear space. This sigmoid class is an
    alias for the `Gumbel` class. It is left to the user to transform stimulus values to
    logarithmic space.
    """
    pass


_CLASS_BY_NAME = {
    'norm': Gaussian,
    'gauss': Gaussian,
    'logistic': Logistic,
    'gumbel': Gumbel,
    'rgumbel': ReverseGumbel,
    'tdist': Student,
    'student': Student,
    'heavytail': Student,
    'weibull': Weibull,
}


ALL_SIGMOID_NAMES = set(_CLASS_BY_NAME.keys())
ALL_SIGMOID_NAMES |= {'neg_' + name for name in ALL_SIGMOID_NAMES}


def sigmoid_by_name(name, PC=None, alpha=None):
    """ Find and initialize a sigmoid from the name.

    The list of supported name can be found in the global
    variable :const:`psignifit.sigmoids.ALL_SIGMOID_NAMES`.

    Note, that some supported names are synonymes, such
    equal sigmoids might be returned for different names.

    Names starting with `neg_` indicate, that the
    sigmoid is decreasing instead of increasing.

    See :meth:`psignifit.sigmoids.Sigmoid.__init__` for
     a description of the arguments.
    """
    kwargs = dict()
    name = name.lower().strip()
    if PC is not None:
        kwargs['PC'] = PC
    if alpha is not None:
        kwargs['alpha'] = alpha
    if name.startswith('neg_'):
        name = name[4:]
        kwargs['negative'] = True

    return _CLASS_BY_NAME[name](**kwargs)
