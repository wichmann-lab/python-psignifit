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
             PC: Percentage correct (sigmoid function value) at threshold
             alpha: Scaling parameter
             negative: Flip sigmoid such percentage correct is decreasing.
        """
        self.alpha = alpha
        self.negative = negative
        if negative:
            self.PC = 1 - PC
        else:
            self.PC = PC

    def __eq__(self, o: object) -> bool:
        return (isinstance(o, self.__class__)
                and o.PC == self.PC
                and o.alpha == self.alpha
                and o.negative == self.negative)

    def __call__(self, stimulus_level: N, threshold: N, width: N) -> N:
        """ Calculate the sigmoid value at specified stimulus levels.

        Args:
            perc_correct: Percentage correct at the threshold to calculate.
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
            gamma: Parameter value for the lower offset of the sigmoid
            lambd: Parameter value for the upper offset of the sigmoid
        Returns:
            Percentage correct at the stimulus values.
        """
        value = self._value(stimulus_level, threshold, width)

        if self.negative:
            return 1 - value
        else:
            return value

    def slope(self, stimulus_level: N, threshold: N, width: N, gamma: N = 0, lambd: N = 0) -> N:
        """ Calculate the slope at specified stimulus levels.

        Args:
            perc_correct: Percentage correct at the threshold to calculate.
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

    def inverse(self, perc_correct: N, threshold: N, width: N,
                gamma: Optional[N] = None, lambd: Optional[N] = None) -> np.ndarray:
        """ Finds the stimulus value for given parameters at different percent correct.

        See :class:psignifit.sigmoids.Sigmoid for a discussion of the parameters.

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

        result = self._inverse(perc_correct, threshold, width)
        return result

    def _value(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should be overwritten by an implementation.")

    def assert_sanity_checks(self, n_samples: int, threshold: float, width: float):
        """ Assert multiple sanity checks on this sigmoid implementation.

        These checks cannot completely assure the correct implementation of a sigmoid,
        but try to catch common and obvious mistakes.
        We recommend to use these for custom sigmoid subclasses.

        The checks are performed on linear spaced stimulus levels between 0 and 1
        and the provided sigmoid parameters.

        Two checks for relations between parameters:
          - `sigmoid(threshold_stimulus_level) == threshold_percent_correct`
          - `|X_L - X_R| == width`
            with `sigmoid(X_L) == alpha`
            and  `sigmoid(X_R) == 1 - alpha`

        Two checks for the inverse:
          - `inverse(PC) == threshold_stimulus_level`
          - `inverse(inverse(stimulus_levels) == stimulus_levels`

        Two checks for the slope:
          - `maximum(|slope(stimulus_levels)|)` close to `|slope(0.5)|`
          - `slope(stimulus_levels) > 0`  (or < 0 for negative sigmoid)

        Args:
             n_samples: Number of stimulus levels between 0 (exclusive) and 1 for tests
             threshold: Parameter value for threshold at PC
             width: Width of the sigmoid
        Raises:
              AssertionError if a sanity check fails.
        """
        stimulus_levels = np.linspace(1e-8, 1, n_samples)
        threshold_stimulus_level = threshold
        if self.negative:
            stimulus_levels = 1 - stimulus_levels

        # sigmoid(threshold_stimulus_level) == threshold_percent_correct
        np.testing.assert_allclose(self(threshold_stimulus_level, threshold, width), self.PC)
        # |X_L - X_R| == WIDTH, with
        # with sigmoid(X_L) == ALPHA
        # and  sigmoid(X_R) == 1 - ALPHA
        perc_correct = self(stimulus_levels, threshold, width)
        idx_alpha, idx_nalpha = (np.abs(perc_correct - self.alpha).argmin(),
                                 np.abs(perc_correct - (1 - self.alpha)).argmin())
        np.testing.assert_allclose(perc_correct[idx_nalpha] - perc_correct[idx_alpha], width, atol=0.02)

        # Inverse sigmoid at threshold percentage correct (y-axis)
        # Expects the threshold stimulus level (x-axis).
        stimulus_level_from_inverse = self.inverse(self.PC,
                                                   threshold=threshold,
                                                   width=width)
        np.testing.assert_allclose(stimulus_level_from_inverse, threshold_stimulus_level)
        # Expects inverse(value(x)) == x
        y = self(stimulus_levels, threshold=threshold, width=width)
        np.testing.assert_allclose(stimulus_levels,
                                   self.inverse(y, threshold=threshold, width=width),
                                   atol=1e-9)

        slope = self.slope(stimulus_levels, threshold=threshold, width=width, gamma=0, lambd=0)
        # Expects maximal slope at a medium stimulus level
        assert 0.4 * len(slope) < np.argmax(np.abs(slope)) < 0.6 * len(slope)
        # Expects slope to be all positive/negative for standard/decreasing sigmoid
        if self.negative:
            assert np.all(slope < 0)
        else:
            assert np.all(slope > 0)


class Gaussian(Sigmoid):
    """ Sigmoid based on the Gaussian distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = width / (norminv(1 - self.alpha) - norminv(self.alpha))
        return normcdf(stimulus_level, (threshold - norminvg(self.PC, 0, C)), C)

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = norminv(1 - self.alpha) - norminv(self.alpha)
        normalized_stimulus_level = (stimulus_level - threshold) / threshold * C
        normalized_slope = sp.stats.norm.pdf(normalized_stimulus_level)
        return normalized_slope * C / width

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = norminv(1 - self.alpha) - norminv(self.alpha)
        return norminvg(perc_correct, threshold - norminvg(self.PC, 0, width / C), width / C)


class Logistic(Sigmoid):
    """ Sigmoid based on the Logistic distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        return 1 / (1 + np.exp(-2 * np.log(1 / self.alpha - 1) / width * (stimulus_level - threshold)
                               + np.log(1 / self.PC - 1)))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = 2 * np.log(1 / self.alpha - 1) / width
        unscaled_slope = np.exp(-C * (stimulus_level - threshold) + np.log(1 / self.PC - 1))
        return C * unscaled_slope / (1 + unscaled_slope)**2

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        return threshold - width * (np.log(1 / perc_correct - 1) - np.log(1 / self.PC - 1)) / 2 / np.log(1 / self.alpha - 1)


class Gumbel(Sigmoid):
    """ Sigmoid based on the Gumbel distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return 1 - np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(1 - self.PC))))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        unscaled_stimulus_level = np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(1 - self.PC)))
        return C / width * np.exp(-unscaled_stimulus_level) * unscaled_stimulus_level

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(self.alpha)) - np.log(-np.log(1 - self.alpha))
        return threshold + (np.log(-np.log(1 - perc_correct)) - np.log(-np.log(1 - self.PC))) * width / C


class ReverseGumbel(Sigmoid):
    """ Sigmoid based on the reversed Gumbel distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return np.exp(-np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(self.PC))))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        unscaled_stimulus_level = np.exp(C / width * (stimulus_level - threshold) + np.log(-np.log(self.PC)))
        return -C / width * np.exp(-unscaled_stimulus_level) * unscaled_stimulus_level

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = np.log(-np.log(1 - self.alpha)) - np.log(-np.log(self.alpha))
        return threshold + (np.log(-np.log(perc_correct)) - np.log(-np.log(self.PC))) * width / C


class Student(Sigmoid):
    """ Sigmoid based on the Student-t distribution's CDF. """
    def _value(self, stimulus_level, threshold, width):
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        return t1cdf(C * (stimulus_level - threshold) / width + t1icdf(self.PC))

    def _slope(self, stimulus_level: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        stimLevel = (stimulus_level - threshold) * C / width + t1icdf(self.PC)
        return C / width * sp.stats.t.pdf(stimLevel, df=1)

    def _inverse(self, perc_correct: np.ndarray, threshold: np.ndarray, width: np.ndarray) -> np.ndarray:
        C = (t1icdf(1 - self.alpha) - t1icdf(self.alpha))
        return (t1icdf(perc_correct) - t1icdf(self.PC)) * width / C + threshold


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
