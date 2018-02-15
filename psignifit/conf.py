"""This module defines the basic configuration object for psignifit.
"""
import re

import numpy as np

from .utils import PsignifitException
from . import sigmoids


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
             'borders',
             'CI_method',
             'confP',
             'dynamic_grid',
             'estimate_type',
             'experiment_type',
             'fast_optim',
             'fixed_pars',
             'grid_eval',
             'grid_set_type',
             'instant_plot',
             'max_border_value',
             'move_borders',
             'pool_max_blocks',
             'pool_max_gap',
             'pool_max_length',
             'pool_xtol',
             'priors',
             'sigmoid',
             'steps',
             'steps_moving_borders',
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
        self._logspace = False

        # set public defaults
        self.beta_prior = 10
        self.CI_method = 'percentiles'
        self.confP = (.95, .9, .68)
        self.dynamic_grid = False
        self.estimate_type = 'MAP'
        self.experiment_type = 'yes/no'
        self.fast_optim = False
        self.fixed_pars = (None, )*5
        self.grid_eval = None
        self.grid_set_type = 'cumDist'
        self.instant_plot = False
        self.max_border_value = 1e-05
        self.move_borders = True
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
            if hasattr(self, 'check_'+name):
                # run the check
                # the check method should raise if value is not valid
                getattr(self, 'check_'+name)(value)
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

    def check_borders(self, value):
        if value is not None:
            # borders is a dict in the form {'parameter_name': (left, right)}
            if type(value) != dict:
                raise PsignifitException(
         f'Option borders must be a dictionary ({type(value).__name__} given)!')
            # now check that the keys in the dictionary are valid
            vkeys = {'threshold', 'width', 'lambda', 'gamma', 'eta'}
            if vkeys < set(value.keys()):
                raise PsignifitException(
         f'Option borders keys must be in {vkeys}. Given {list(value.keys())}!')
            # now check that the values are sequences of length 2
            for v in value.values():
                try:
                    correct_length = (len(v) == 2)
                except Exception:
                    correct_length = False
                if not correct_length:
                    raise PsignifitException(
                           f'Borders must be a sequence of 2 items: {v} given!')

    def check_experiment_type(self, value):
        cond1 = value in ('yes/no', 'equal asymptote')
        cond2 = re.match('[0-9]AFC', value)
        if not (cond1 or cond2):
            raise PsignifitException(
        f'Invalid experiment type: "{value}"\nValid types: "yes/no",'+
         ' "equal asymptote", "2AFC", "3AFC", etc...')

        self.steps = [40,40,20,20,20] if value=='yes/no' else [40,40,20,1,20]
        self.steps_moving_borders = [25,30, 10,10,15] if value=='YesNo' else [30,40,10,1,20]

    def check_sigmoid(self, value):
        cond1 = value in dir(sigmoids)
        cond2 = value.startswith('_') or value.startswith('my_')
        if (not cond1) or (cond2):
            raise PsignifitException(f'Invalid sigmoid: "{value}"!')
        # set logspace when appropriate
        if value in ('weibull', 'logn', 'neg_weibull', 'neg_logn'):
            self._logspace = True

    def check_dynamic_grid(self, value):
        if value:
            if self.grid_eval is None:
                self.grid_eval = 10000
            if self.uniform_weigth is None:
                self.uniform_weigth = 1.

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

    def check_width_alpha(self, value):
        try:
            # check that it is a number:
            diff = 1 - value
            wrong_type = False
        except Exception:
            wrong_type = True
        if wrong_type or not ( 0 < diff < 1):
            raise PsignifitException(
             f"Option width_alpha must be between 0 and 1 ({value} given)!")

    def check_width_min(self, value):
        if value:
            try:
                _ = value + 1
            except Exception:
                raise PsignifitException("Option width_min must be a number")



