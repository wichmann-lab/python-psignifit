"""This module defines the basic configuration object for psignifit.
"""
import re
import dataclasses
from typing import Any, Dict, Tuple, Optional, Union
import warnings

from . import sigmoids
from ._utils import PsignifitException
from ._typing import ExperimentType, Prior


_PARAMETERS = {'threshold', 'width', 'lambda', 'gamma', 'eta'}


@dataclasses.dataclass
class Configuration:
    """The basic configuration object for psignifit.

    This class contains a set of valid options and the corresponding sanity
    checks.
    These checks raise `PsignifitException` if an invalid option is specified or
    if a valid option is specified with a value outside of the allowed range.

    The sanity checks are only run during initialization.
    Changing attributes is highly discouraged and should be followed
    by rerunning sanity checks with `config.check_attributes()`.

    Note: attributes and methods starting with `_` are considered private and
    used internally. Do not change them unlsess you know what you are doing

    Note for the developer: if you want to add a new valid option `foobar`,
    expand the `Conf.valid_opts` tuple (in alphabetical order) and add any
    check in a newly defined method `def check_foobar(self, value)`, which
    raises `PsignifitException` if `value` is outside of the accepted range
    for `foobar`.
    """
    beta_prior: int = 10
    CI_method: str = 'project'
    confP: Tuple[float, float, float] = (.95, .9, .68)
    estimate_type: str = 'MAP'
    experiment_type: str = ExperimentType.YES_NO.value
    experiment_choices: Optional[int] = None
    fast_optim: bool = False
    fixed_parameters: Optional[Dict[str, float]] = None
    grid_set_type: str = 'cumDist'
    instant_plot: bool = False
    max_bound_value: float = 1e-05
    move_bounds: bool = True
    pool_max_blocks: int = 25
    priors: Optional[Dict[str, Prior]] = dataclasses.field(default=None, hash=False)
    sigmoid: Union[str, sigmoids.Sigmoid] = 'norm'
    stimulus_range: Optional[Tuple[float, float]] = None
    thresh_PC: float = 0.5
    verbose: bool = True
    width_alpha: float = 0.05
    width_min: Optional[float] = None

    # attributes, if not specified, will be initialize based on others
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
    grid_steps: Dict[str, int] = dataclasses.field(default_factory=dict)
    steps_moving_bounds: Dict[str, int] = dataclasses.field(default_factory=dict)


    def __post_init__(self):
        self.check_attributes()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        config_dict = config_dict.copy()
        return cls(confP=tuple(config_dict.pop('confP')),
                   **config_dict)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def check_attributes(self):
        """ Run sanity checks.

        For each attribute named NAME, this function
        searches for a method check_NAME and executes
        it if found.

        Example for a sanity check of a `foobar` attribute:
        .. code-block:: python

            def check_foobar(self, value):
                if value > 10:
                    raise PsignifitException(f'Foobar must be < 10: {value} given!')
         """
        for attribute in dataclasses.fields(self):
            sanity_check_name = 'check_' + attribute.name
            if hasattr(self, sanity_check_name):
                sanity_check_method = getattr(self, sanity_check_name)
                attribute_value = getattr(self, attribute.name)
                sanity_check_method(attribute_value)
        self.check_experiment_type_matches_fixed_parameters(self.fixed_parameters, self.experiment_type)

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

    def check_fixed_parameters(self, value):
        if value is not None:
            # fixed parameters is a dict in the form {'parameter_name': value}
            if isinstance(type(value), dict):
                raise PsignifitException(
                    f'Option fixed_parameters must be a dictionary ({type(value).__name__} given)!'
                )
            # now check that the keys in the dictionary are valid
            vkeys = _PARAMETERS
            if vkeys < set(value.keys()):
                raise PsignifitException(
                    f'Option fixed_paramters keys must be in {vkeys}. Given {list(value.keys())}!'
                )


    def check_experiment_type_matches_fixed_parameters(self, fixed_params, experiment_type):
        if experiment_type == ExperimentType.N_AFC.value:
            if fixed_params is not None and 'gamma' in fixed_params:
                warnings.warn(
                    f'The parameter gamma was fixed to {fixed_params["gamma"]}. In {ExperimentType.N_AFC.value} experiments gamma must be fixed to 1/n. Ignoring fixed gamma.')


    def check_experiment_type(self, value):
        valid_values = [type.value for type in ExperimentType]
        is_valid = value in valid_values
        is_nafc = re.match('[0-9]+AFC', value)
        if not (is_valid or is_nafc):
            raise PsignifitException(
                f'Invalid experiment type: "{value}"\nValid types: {valid_values},' +
                ', or "2AFC", "3AFC", etc...')
        if is_nafc:
            self.experiment_choices = int(value[:-3])
            self.experiment_type = ExperimentType.N_AFC.value
            value = ExperimentType.N_AFC.value
        if value == ExperimentType.N_AFC.value and self.experiment_choices is None:
            raise PsignifitException("For nAFC experiments, expects 'experiment_choices' to be a number, got None.\n"
                                     "Can be specified in the experiment type, e.g. 2AFC, 3AFC, … .")

        default_grid_steps = {
            'threshold': 40,
            'width': 40,
            'lambda': 20,
            # 'gamma' will be set below
            'eta': 20,
        }
        if value == ExperimentType.YES_NO.value:
            default_grid_steps['gamma'] = 20
            self.steps_moving_bounds = {
                'threshold': 25,
                'width': 30,
                'lambda': 10,
                'gamma': 10,
                'eta': 15,
                **self.steps_moving_bounds
            }
        else:
            default_grid_steps['gamma'] = 1
            self.steps_moving_bounds = {
                'threshold': 30,
                'width': 40,
                'lambda': 10,
                'gamma': 1,
                'eta': 20,
                **self.steps_moving_bounds
            }
        self.grid_steps = {**default_grid_steps,
                           **self.grid_steps}

    def check_sigmoid(self, value):
        try:
            self.make_sigmoid()
        except KeyError:
            raise PsignifitException('Invalid sigmoid name "{value}", use one of {sigmoids.ALL_SIGMOID_NAMES}')

    def check_stimulus_range(self, value):
        if value:
            try:
                len_ = len(value)
                wrong_type = False
            except TypeError:
                wrong_type = True
            if wrong_type or len_ != 2:
                raise PsignifitException(
                    "Option stimulus range must be a sequence of two items!")

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

    def check_width_min(self, value):
        if value:
            try:
                _ = value + 1
            except Exception:
                raise PsignifitException("Option width_min must be a number")

    def make_sigmoid(self) -> sigmoids.Sigmoid:
        """ Construct sigmoid according to this configuration.

        Returns:
             Sigmoid object with percentage correct and alpha according to config.
        """
        if isinstance(self.sigmoid, sigmoids.Sigmoid):
            self.sigmoid.PC = self.thresh_PC
            self.sigmoid.alpha = self.width_alpha
            return self.sigmoid
        else:
            return sigmoids.sigmoid_by_name(self.sigmoid, PC=self.thresh_PC, alpha=self.width_alpha)
