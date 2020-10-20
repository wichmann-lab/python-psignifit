"""This module defines the basic configuration object for psignifit.
"""
import re
import dataclasses
from typing import Dict, Tuple, Optional

import numpy as np

from . import sigmoids
from .utils import PsignifitException
from .typing import ExperimentType, Prior


_PARAMETERS = {'threshold', 'width', 'lambda', 'gamma', 'eta'}


@dataclasses.dataclass(frozen=True)
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

    beta_prior: int = 10
    CI_method: str = 'percentiles'
    confP: Tuple[float, float, float] = (.95, .9, .68)
    dynamic_grid: bool = False
    estimate_type: str = 'MAP'
    experiment_type: ExperimentType = ExperimentType.YES_NO
    experiment_choices: Optional[int] = None
    fast_optim: bool = False
    fixed_parameters: Optional[Dict[str, float]] = None
    grid_set_type: str = 'cumDist'
    instant_plot: bool = False
    max_bound_value: float = 1e-05
    move_bounds: bool = True
    pool_max_blocks: int  = 25
    pool_max_gap: float = np.inf
    pool_max_length: float = np.inf
    pool_xtol: float = 0
    priors: Optional[Dict[str, Prior]] = dataclasses.field(default=None, hash=False)
    sigmoid: str = 'norm'
    stimulus_range: Optional[Tuple[float, float]] = None
    thresh_PC: float = 0.5
    verbose: bool = True
    width_alpha: float = 0.05
    width_min: Optional[float] = None

    # parameters, if not specified, will be initialize based on others
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
    grid_eval: Optional[int] = None
    uniform_weight: Optional[float] = None

    # parameters which are always initialized based on other parameters
    grid_steps: Dict[str, int] = dataclasses.field(init=False)
    steps_moving_bounds: Dict[str, int] = dataclasses.field(init=False)

    def __post_init__(self):
        self.check_parameters()

    def check_parameters(self):
        for field in dataclasses.fields(self):
            checker_name = 'check_' + field.name
            if hasattr(self, checker_name):
                checker = getattr(self, checker_name)
                old_value = getattr(self, field.name)
                object.__setattr__(self, field.name, checker(old_value))

    # template for an option checking method
    # def check_foobar(self, value):
    #    if value > 10:
    #       raise PsignifitException(f'Foobar must be < 10: {value} given!')
    def check_bounds(self, value):
        if value is not None:
            # bounds is a dict in the form {'parameter_name': (left, right)}
            if type(value) != dict:
                raise PsignifitException(
                    f'Option bounds must be a dictionary ({type(value).__name__} given)!'
                )
            # now check that the keys in the dictionary are valid
            vkeys = _PARAMETERS
            if vkeys < set(value.keys()):
                raise PsignifitException(
                    f'Option bounds keys must be in {vkeys}. Given {list(value.keys())}!'
                )
            # now check that the values are sequences of length 2
            for v in value.values():
                try:
                    correct_length = (len(v) == 2)
                except Exception:
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
            vkeys = _PARAMETERS
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
                object.__setattr__(self, 'experiment_choices', int(value[:-3]))
                value = ExperimentType.N_AFC
            else:
                value = ExperimentType(value)
        if value is ExperimentType.N_AFC and self.experiment_choices is None:
            raise PsignifitException("For nAFC experiments, expects 'experiment_choices' to be a number, got None.\n"
                                     "Can be specified in the experiment type, e.g. 2AFC, 3AFC, … .")

        # because the attributes are immutable (frozen)
        # we have to set them with this special syntax
        object.__setattr__(self, 'grid_steps', {
            'threshold': 40,
            'width': 40,
            'lambda': 20,
            # 'gamma' will be set below
            'eta': 20,
        })
        if value == ExperimentType.YES_NO.value:
            self.grid_steps['gamma'] = 20
            object.__setattr__(self, 'steps_moving_bounds', {
                'threshold': 25,
                'width': 30,
                'lambda': 10,
                'gamma': 10,
                'eta': 15,
            })
        else:
            self.grid_steps['gamma'] = 1
            object.__setattr__(self, 'steps_moving_bounds', {
                'threshold': 30,
                'width': 40,
                'lambda': 10,
                'gamma': 1,
                'eta': 20,
            })

        return value

    def check_sigmoid(self, value):
        try:
            sigmoid = sigmoids.sigmoid_by_name(value)
        except:
            raise PsignifitException('Invalid sigmoid name "{value}", use one of {sigmoids.ALL_SIGMOID_NAMES}')
        return value

    def check_dynamic_grid(self, value):
        if value:
            if self.grid_eval is None:
                object.__setattr__(self, 'grid_eval', 10000)
            if self.uniform_weigth is None:
                object.__setattr__(self, 'uniform_weight', 1.)

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
        except Exception:
            wrong_type = True
        if wrong_type or not (0 < diff < 1):
            raise PsignifitException(
                f"Option width_alpha must be between 0 and 1 ({value} given)!")

        return value

    def check_width_min(self, value):
        if value:
            try:
                _ = value + 1
            except Exception:
                raise PsignifitException("Option width_min must be a number")

        return value
