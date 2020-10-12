"""This module defines the basic configuration object for psignifit.
"""
import re

import numpy as np

from . import sigmoids
from .utils import PsignifitException
from .typing import ExperimentType


class Conf:
    """The basic configuration object for psignifit.

    This class contains a set of valid options and the corresponding sanity
    checks.

    It raises `PsignifitException` if an invalid option is specified or
    if a valid option is specified with a value outside of the allowed range.

    Note: attributes and methods starting with `_` are considered private and
    used internally. Do not change them unlsess you know what you are doing

    Note for the developer: if you want to add a new valid option `foobar`,
    expand the `Conf.valid_opts` tuple (in alphabetical order) and add any
    check in a newly defined method `def check_foobar(self, value)`, which
    raises `PsignifitException` if `value` is outside of the accepted range
    for `foobar`.
    """
    # set of valid options for psignifit. Add new attributes to this tuple
    _valid_opts = (
        'beta_prior',
        'bounds',
        'CI_method',
        'confP',
        'dynamic_grid',
        'estimate_type',
        'experiment_type',
        'experiment_choices',
        'fast_optim',
        'fixed_parameters',
        'grid_eval',
        'grid_set_type',
        'grid_steps',
        'instant_plot',
        'max_bound_value',
        'move_bounds',
        'parameters',
        'pool_max_blocks',
        'pool_max_gap',
        'pool_max_length',
        'pool_xtol',
        'priors',
        'sigmoid',
        'steps_moving_bounds',
        'stimulus_range',
        'thresh_PC',
        'uniform_weight',
        'verbose',
        'width_alpha',
        'width_min',
    )

    def __init__(self, **kwargs):
        # we only allow keyword arguments
        # set private attributes defaults
        self._parameters = {'threshold', 'width', 'lambda', 'gamma', 'eta'}

        # set public defaults
        self.beta_prior = 10
        self.bounds = None
        self.CI_method = 'percentiles'
        self.confP = (.95, .9, .68)
        self.dynamic_grid = False
        self.estimate_type = 'MAP'
        self.experiment_type = ExperimentType.YES_NO
        self.experiment_choices = None
        self.fast_optim = False
        self.fixed_parameters = None
        self.grid_eval = None
        self.grid_set_type = 'cumDist'
        self.grid_steps = None
        self.instant_plot = False
        self.max_bound_value = 1e-05
        self.move_bounds = True
        self.pool_max_blocks = 25
        self.pool_max_gap = np.inf
        self.pool_max_length = np.inf
        self.pool_xtol = 0
        self.priors = None
        self.sigmoid = 'norm'
        self.stimulus_range = None
        self.thresh_PC = 0.5
        self.uniform_weight = None
        self.verbose = True
        self.width_alpha = 0.05
        self.width_min = None

        # overwrite defaults with user preferences
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif name in self._valid_opts:
            # first run checks for the supplied option, if any are available
            if hasattr(self, 'check_' + name):
                # run the check
                # the check method should raise if value is not valid
                value = getattr(self, 'check_' + name)(value)
            super().__setattr__(name, value)
        else:
            raise PsignifitException(f'Invalid option "{name}"!')

    # template for an option checking method
    # def check_foobar(self, value):
    #    if value > 10:
    #       raise PsignifitException(f'Foobar must be < 10: {value} given!')

    def __repr__(self):
        # give an nice string representation of ourselves
        _str = []
        for name in sorted(self._valid_opts):
            # if name is not defined, returns None
            value = getattr(self, name, None)
            _str.append(f'{name}: {value}')
        return '\n'.join(_str)

    def check_bounds(self, value):
        if value is not None:
            # bounds is a dict in the form {'parameter_name': (left, right)}
            if type(value) != dict:
                raise PsignifitException(
                    f'Option bounds must be a dictionary ({type(value).__name__} given)!'
                )
            # now check that the keys in the dictionary are valid
            vkeys = self._parameters
            if vkeys < set(value.keys()):
                raise PsignifitException(
                    f'Option bounds keys must be in {vkeys}. Given {list(value.keys())}!'
                )
            # now check that the values are sequences of length 2
            for v in value.values():
                try:
                    correct_length = (len(v) == 2)
                except TypeError:
                    correct_length = False
                if not correct_length:
                    raise PsignifitException(
                        f'Bounds must be a sequence of 2 items: {v} given!')
            return value

    def check_fixed_parameters(self, value):
        if value is not None:
            # fixed parameters is a dict in the form {'parameter_name': value}
            if type(value) != dict:
                raise PsignifitException(
                    f'Option fixed_parameters must be a dictionary ({type(value).__name__} given)!'
                )
            # now check that the keys in the dictionary are valid
            vkeys = self._parameters
            if vkeys < set(value.keys()):
                raise PsignifitException(
                    f'Option fixed_paramters keys must be in {vkeys}. Given {list(value.keys())}!'
                )
        return value

    def check_experiment_type(self, value):
        valid_values = [type.value for type in ExperimentType]
        if not isinstance(value, ExperimentType):
            cond1 = value in valid_values
            cond2 = re.match('[0-9]+AFC', value)
            if not (cond1 or cond2):
                raise PsignifitException(
                    f'Invalid experiment type: "{value}"\nValid types: {valid_values},' +
                    ', or "2AFC", "3AFC", etc...')
            if cond2:
                self.experiment_choices = int(value[:-3])
                value = ExperimentType.N_AFC
            else:
                value = ExperimentType(value)
        if value is ExperimentType.N_AFC and self.experiment_choices is None:
            raise PsignifitException("For nAFC experiments, expects 'experiment_choices' to be a number, got None.\n"
                                     "Can be specified in the experiment type, e.g. 2AFC, 3AFC, … .")

        self.grid_steps = {param: None for param in self._parameters}
        self.steps_moving_bounds = {param: None for param in self._parameters}
        if value == ExperimentType.YES_NO.value:
            self.grid_steps['threshold'] = 40
            self.grid_steps['width'] = 40
            self.grid_steps['lambda'] = 20
            self.grid_steps['gamma'] = 20
            self.grid_steps['eta'] = 20
            self.steps_moving_bounds['threshold'] = 25
            self.steps_moving_bounds['width'] = 30
            self.steps_moving_bounds['lambda'] = 10
            self.steps_moving_bounds['gamma'] = 10
            self.steps_moving_bounds['eta'] = 15
        else:
            self.grid_steps['threshold'] = 40
            self.grid_steps['width'] = 40
            self.grid_steps['lambda'] = 20
            self.grid_steps['gamma'] = 1
            self.grid_steps['eta'] = 20
            self.steps_moving_bounds['threshold'] = 30
            self.steps_moving_bounds['width'] = 40
            self.steps_moving_bounds['lambda'] = 10
            self.steps_moving_bounds['gamma'] = 1
            self.steps_moving_bounds['eta'] = 20

        return value

    def check_sigmoid(self, value):
        try:
            sigmoids.sigmoid_by_name(value)
        except KeyError:
            raise PsignifitException('Invalid sigmoid name "{value}", use one of {sigmoids.ALL_SIGMOID_NAMES}')
        return value

    def check_dynamic_grid(self, value):
        if value:
            if self.grid_eval is None:
                self.grid_eval = 10000
            if self.uniform_weigth is None:
                self.uniform_weigth = 1.

        return value

    def check_stimulus_range(self, value):
        if value:
            try:
                len_ = len(value)
                wrong_type = False
            except TypeError:
                wrong_type = True
            if wrong_type or len_ != 2:
                raise PsignifitException(
                    f"Option stimulus range must be a sequence of two items!")

        return value

    def check_width_alpha(self, value):
        try:
            # check that it is a number:
            diff = 1 - value
            wrong_type = False
        except TypeError:
            wrong_type = True
        if wrong_type or not (0 < diff < 1):
            raise PsignifitException(
                f"Option width_alpha must be between 0 and 1 ({value} given)!")

        return value

    def check_width_min(self, value):
        if value:
            try:
                _ = value + 1
            except TypeError:
                raise PsignifitException("Option width_min must be a number")

        return value
