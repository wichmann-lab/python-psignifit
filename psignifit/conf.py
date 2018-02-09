"""This module defines the basic configuration object for psignifit.
"""
import re

import numpy as np

from . import sigmoids

class PsignifitConfException(Exception):
    pass

class Conf:
    """The basic configuration object for psignifit.

    This class contains a set of valid options and the corresponding sanity
    checks.

    It raises `PsignifitConfException` if an invalid option is specified or
    if a valid option is specified with a value outside of the allowed range.

    Note: attributes and methods starting with `_` are considered private and
    used internally. Do not change them unlsess you know what you are doing

    Note for the developer: if you want to add a new valid option `foobar`,
    expand the `Conf.valid_opts` tuple (in alphabetical order) and add any
    check in a newly defined method `def check_foobar(self, value)`, which
    raises `PsignifitConfException` if `value` is outside of the accepted range
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
             'nblocks',
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
        self.experiment_type = 'YesNo'
        self.fast_optim = False
        self.fixed_pars = (None, )*5
        self.grid_eval = None
        self.grid_set_type = 'cumDist'
        self.instant_plot = False
        self.max_border_value = 1e-05
        self.move_borders = True
        self.nblocks = 25
        self.pool_max_gap = np.inf
        self.pool_max_length = np.inf
        self.pool_xtol = 0
        self.sigmoid = 'norm'
        self.stimulus_range = False
        self.thresh_PC = 0.5
        self.uniform_weight = None
        self.verbose = False
        self.width_alpha = 0.05

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
            raise PsignifitConfException(f'Invalid option "{name}"!')

    # template for an option checking method
    # def check_foobar(self, value):
    #    if value > 10:
    #       raise PsignifitConfException(f'Foobar must be < 10: {value} given!')

    def __repr__(self):
        # give an nice string representation of ourselves
        _str = []
        for name in sorted(self._valid_opts):
            # if name is not defined, returns None
            value = getattr(self, name, None)
            _str.append(f'{name}: {value}')
        return '\n'.join(_str)

    def check_experiment_type(self, value):
        cond1 = value in ('YesNo', 'equalAsymptote')
        cond2 = re.match('[0-9]AFC', value)
        if not (cond1 or cond2):
            raise PsignifitConfException(f'Invalid experiment type: "{value}"!')
        self.steps = [40,40,20,20,20] if value=='YesNo' else [40,40,20,1,20]
        self.steps_moving_borders = [25,30, 10,10,15] if value=='YesNo' else [30,40,10,1,20]

    def check_sigmoid(self, value):
        cond1 = value in dir(sigmoids)
        cond2 = value.startswith('_') or value.startswith('my_')
        if (not cond1) or (cond2):
            raise PsignifitConfException(f'Invalid sigmoid: "{value}"!')
        # set logspace when appropriate
        if value in ('weibull', 'logn', 'neg_weibull', 'neg_logn'):
            self._logspace = True
