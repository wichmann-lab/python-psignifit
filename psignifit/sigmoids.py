""" All sigmoid functions.

If you add a new sigmoid type, add it to the CLASS_BY_NAME constant
"""
from typing import Optional, TypeVar

import numpy as np
import scipy.stats


# sigmoid can be calculated on single floats, or on numpy arrays of floats
N = TypeVar('N', float, np.ndarray)


class Sigmoid:
    """ Base class for sigmoid implementation.

    Handles negative output for the specific sigmoid implementations.

    Sigmoid classes should derive from this class and implement the methods '_value', '_slope', '_threshold',
    and `_standard_parameters`.

    In many cases, a sigmoid corresponds to the CDF of a probability distribution. This is the case
    for all the sigmoids included in `psignifit`. If the probability distribution
    is one of the distributions defined in `scipy.stats`, it is possible to define a `Sigmoid` subclass
    by implementing only the methods `_scipy_distr` and `_standard_parameters`. The methods
    '_value', '_slope', '_threshold' will be automatically defined in such a case.

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
            stimulus_level: Stimulus level value at which to calculate the slope
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
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
            stimulus_level: Stimulus level value at which to calculate the slope
            threshold: Parameter value for threshold at PC
            width: Parameter value for width of the sigmoid
            gamma: Parameter value for the lower offset of the sigmoid
            lambd: Parameter value for the upper offset of the sigmoid
        Returns:
            Slope at the stimulus values.
        """

        raw_slope = self._slope(stimulus_level, threshold, width)
        slope = (1 - gamma - lambd) * raw_slope

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

    def standard_parameters(self, threshold: N, width: N) -> tuple:
        """ Transforms parameters threshold and width to a standard parametrization.

        The interpretation of the standard parameters, location and scale, depends on the sigmoid class used.
        For instance, for a Gaussian sigmoid, the location corresponds to the mean and the scale to the standard
        deviation of the distribution.

        For negative slope sigmoids, we return the same parameters as for the positive ones.

        Args:
            threshold: Threshold value at PC
            width: Width of the sigmoid
        Returns:
            Standard parameters (loc, scale) for the sigmoid subclass.
        """
        return self._standard_parameters(threshold, width)

    # --- Private interface

    def _value(self, stimulus_level: N, threshold: N, width: N) -> N:
        """ Compute the sigmoid value at specified stimulus levels.

        This method can be overwritten by subclasses to define new sigmoids. The base implementation uses the
        CDF of a continuous distribution function defined in `scipy.stats` and returned by `_scipy_distr`.

        Args:
            stimulus_level: Stimulus level values at which to calculate the sigmoid value
            threshold: Threshold value at PC
            width: Width of the sigmoid
        Returns:
            Proportion correct at the stimulus values.
        """
        loc, scale = self._standard_parameters(threshold=threshold, width=width)
        value = self._cdf(stimulus_level, loc=loc, scale=scale)
        return value

    def _slope(self, stimulus_level: N, threshold: N, width: N) -> N:
        """ Compute the slope of the sigmoid at a given stimulus level.

        This method can be overwritten by subclasses to define new sigmoids. The base implementation uses the
        PDF of a continuous distribution function defined in `scipy.stats` and returned by `_scipy_distr`.

        Args:
            stimulus_level: Stimulus level value at which to calculate the slope
            threshold: Threshold value at PC
            width: Width of the sigmoid
        Returns:
            Slope at the stimulus level value
        """
        loc, scale = self._standard_parameters(threshold=threshold, width=width)
        raw_slope = self._pdf(stimulus_level, loc=loc, scale=scale)
        return raw_slope

    def _inverse(self, prop_correct: N, threshold: N, width: N) -> N:
        """ Compute the stimulus value at different proportion correct values.

        This method can be overwritten by subclasses to define new sigmoids. The base implementation uses the
        PPF of a continuous distribution function defined in `scipy.stats` and returned by `_scipy_distr`.

        Args:
            prop_correct: Proportion correct values at which to calculate the stimulus level values.
            threshold: Threshold value at PC
            width: Width of the sigmoid
        Returns:
            Stimulus values corresponding to the proportion correct values.
        """
        loc, scale = self._standard_parameters(threshold=threshold, width=width)
        result = self._ppf(prop_correct, loc=loc, scale=scale)
        return result

    def _standard_parameters(self, threshold: N, width: N) -> list:
        """ Transforms parameters threshold and width to a standard parametrization.

        This method must be overwritten by subclasses to define new sigmoids.

        The interpretation of the standard parameters, location and scale, depends on the sigmoid class used.
        For instance, for a Gaussian sigmoid, the location corresponds to the mean and the scale to the standard
        deviation of the distribution.

        For negative slope sigmoids, we return the same parameters as for the positive ones.

        Args:
            threshold: Threshold value at PC
            width: Width of the sigmoid
        Returns:
            Standard parameters (loc, scale) for the sigmoid subclass.
        """
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _scipy_distr(self):
        """ Get the scipy.stats implementation of the underlying cdf.

        Returns
            scipy_distr: scipy.stats implementation of the distribution
            distr_kwarg: A dictionary of extra keyword arguments needed to initialize the scipy.stats distribution
        """
        raise NotImplementedError("This should be overwritten by an implementation.")

    def _ppf(self, x, loc=0, scale=1):
        distr, distr_kwargs = self._scipy_distr()
        return distr.ppf(x, loc=loc, scale=scale, **distr_kwargs)

    def _pdf(self, x, loc=0, scale=1):
        distr, distr_kwargs = self._scipy_distr()
        return distr.pdf(x, loc=loc, scale=scale, **distr_kwargs)

    def _cdf(self, x, loc=0, scale=1):
        distr, distr_kwargs = self._scipy_distr()
        return distr.cdf(x, loc=loc, scale=scale, **distr_kwargs)


class Gaussian(Sigmoid):
    """ Sigmoid based on the Gaussian distribution's CDF. """

    def _scipy_distr(self):
        return scipy.stats.norm, {}

    def _standard_parameters(self, threshold: N, width: N) -> list:
        if self.alpha == 0.0:
            scale = width
        else:
            scale = width / (self._ppf(1 - self.alpha) - self._ppf(self.alpha))
        loc = threshold - self._ppf(self._PC, 0, scale)
        return [loc, scale]


class Logistic(Sigmoid):
    """ Sigmoid based on the Logistic distribution's CDF. """

    def _scipy_distr(self):
        return scipy.stats.logistic, {}

    def _standard_parameters(self, threshold: N, width: N) -> list:
        if self.alpha == 0.0:
            scale = width
        else:
            scale = width / (2 * np.log(1 / self.alpha - 1))
        loc = threshold - self._ppf(self._PC, loc=0, scale=scale)
        return [loc, scale]


class Gumbel(Sigmoid):
    """ Sigmoid based on the Gumbel distribution's CDF. """

    def _scipy_distr(self):
        return scipy.stats.gumbel_l, {}

    def _standard_parameters(self, threshold: N, width: N) -> list:
        if self.alpha == 0.0:
            scale = width
        else:
            scale = width / (np.log(-np.log(self.alpha)) - np.log(-np.log(1-self.alpha)))
        loc = threshold - self._ppf(self._PC, loc=0, scale=scale)
        return [loc, scale]


class ReverseGumbel(Sigmoid):
    """ Sigmoid based on the reversed Gumbel distribution's CDF. """

    def _scipy_distr(self):
        return scipy.stats.gumbel_r, {}

    def _standard_parameters(self, threshold: N, width: N) -> list:
        if self.alpha == 0.0:
            scale = width
        else:
            scale = width / (np.log(-np.log(self.alpha)) - np.log(-np.log(1-self.alpha)))
        loc = threshold - self._ppf(self._PC, loc=0, scale=scale)
        return [loc, scale]


class Student(Sigmoid):
    """ Sigmoid based on the Student-t distribution's CDF. """

    def _scipy_distr(self):
        return scipy.stats.t, {'df': 1}

    def _standard_parameters(self, threshold: N, width: N) -> list:
        if self.alpha == 0.0:
            scale = width
        else:
            scale = width / (self._ppf(1 - self.alpha) - self._ppf(self.alpha))
        loc = threshold - self._ppf(self._PC, loc=0, scale=scale)
        return [loc, scale]


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


def assert_sigmoid_sanity_checks(sigmoid, n_samples: int, threshold: float, width: float):
    """ Assert multiple sanity checks on this sigmoid implementation.

    This is support code to have a first sanity check for  custom sigmoid subclasses.
    These checks cannot completely assure the correct implementation of a sigmoid,
    but try to catch common and obvious mistakes.

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

    # sigmoid(threshold_stimulus_level) == threshold_percent_correct
    np.testing.assert_allclose(sigmoid(threshold_stimulus_level, threshold, width), sigmoid.PC)
    # |X_L - X_R| == WIDTH, with
    # with sigmoid(X_L) == ALPHA
    # and  sigmoid(X_R) == 1 - ALPHA
    prop_correct = sigmoid(stimulus_levels, threshold, width)
    # When the sigmoid is negative, it is decreasing, so we compute the width on 1-prop_correct
    # (Alternatively, we could have used `argmax` and swapped the indices)
    if sigmoid.negative:
        stimulus_levels = 1 - stimulus_levels
    idx_alpha, idx_nalpha = (np.abs(prop_correct - sigmoid.alpha).argmin(),
                             np.abs(prop_correct - (1 - sigmoid.alpha)).argmin())
    np.testing.assert_allclose(
        stimulus_levels[idx_nalpha] - stimulus_levels[idx_alpha],
        width, atol=0.02)

    # Inverse sigmoid at threshold proportion correct (y-axis)
    # Expects the threshold stimulus level (x-axis).
    stimulus_level_from_inverse = sigmoid.inverse(sigmoid.PC,
                                                  threshold=threshold,
                                                  width=width)
    np.testing.assert_allclose(stimulus_level_from_inverse, threshold_stimulus_level)
    # Expects inverse(value(x)) == x
    y = sigmoid(stimulus_levels, threshold=threshold, width=width)
    np.testing.assert_allclose(stimulus_levels,
                               sigmoid.inverse(y, threshold=threshold, width=width),
                               atol=1e-8)

    slope = sigmoid.slope(stimulus_levels, threshold=threshold, width=width, gamma=0, lambd=0)
    # Expects maximal slope at a medium stimulus level
    assert 0.3 * len(slope) < np.argmax(np.abs(slope)) < 0.7 * len(slope)
    # Expects slope to be all positive/negative for standard/decreasing sigmoid
    if sigmoid.negative:
        assert np.all(slope < 0)
    else:
        assert np.all(slope > 0)
