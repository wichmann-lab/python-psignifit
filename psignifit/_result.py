import dataclasses
from typing import Any, Dict, Tuple, List, TextIO, Union, Optional
import json
from pathlib import Path

import numpy as np

from ._configuration import Configuration
from ._typing import ParameterGrid


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclasses.dataclass
class Result:
    parameter_estimate: Dict[str, float]
    configuration: Configuration
    confidence_intervals: Dict[str, List[Tuple[float, float]]]
    data: Tuple[List[float], List[float], List[float]]
    parameter_values: Dict[str, List[float]]
    prior_values: Dict[str, List[float]]
    marginal_posterior_values: Dict[str, List[float]]
    posterior_mass: Optional[np.ndarray] = dataclasses.field(compare=False, default=None)

    # If future attributes should contain numpy arrays,
    # run np.asarray in __post_init__.
    # Otherwise, load_json may result in nested lists instead.
    def __post_init__(self):
        if self.posterior_mass is not None:
            self.posterior_mass = np.asarray(self.posterior_mass)

    def _equal_numpy_dict(first, second):
        """ Test if two dicts of numpy arrays are equal"""
        if first.keys() != second.keys():
            return False
        return all(np.array_equal(first[key], second[key]) for key in first)

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]):
        result_dict = result_dict.copy()
        config_dict = result_dict.pop('configuration')
        return cls(configuration=Configuration.from_dict(config_dict),
                   **result_dict)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save_json(self, file: Union[TextIO, str, Path], **kwargs: Any):
        if 'cls' not in kwargs:
            kwargs['cls'] = NumpyEncoder

        result_dict = self.as_dict()
        if hasattr(file, 'write'):
            json.dump(result_dict, file, **kwargs)
        else:
            with open(file, 'w') as f:
                json.dump(result_dict, f, **kwargs)

    @classmethod
    def load_json(cls, file: Union[TextIO, str, Path], **kwargs: Any):
        if hasattr(file, 'read'):
            result_dict = json.load(file, **kwargs)
        else:
            with open(file, 'r') as f:
                result_dict = json.load(f, **kwargs)
        return cls.from_dict(result_dict)

    def threshold(self, percentage_correct: np.ndarray, unscaled: bool = False, return_ci: bool = True
                  ) -> Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """ Threshold stimulus value and confidence interval for a different percent correct cutoff.

        The CIs you may obtain from this are calculated based on the confidence
        intervals only, e.g. with the shallowest and the steepest psychometric
        function and may thus broaden if you move away from the standard
        threshold of unscaled sigmoid. These CI are only upper bounds.

        For a more accurate inference, refit psignifit using the percentage correct in the configuration.

        Args:
            percentage_correct: percent correct at the threshold you want to calculate
            unscaled: If the percent correct you provide are for the unscaled
                      sigmoid or for the one scaled by lambda and gamma.
                      By default this function returns the one for the scaled sigmoid.
            return_ci: If the confidence bounds should be returned along with the stimulus value
        Returns:
            thresholds: stimulus values for all provided percentage_correct (if return_ci=False)
            (thresholds, ci): stimulus values along with confidence intervals

        """
        percentage_correct = np.asarray(percentage_correct)
        sigmoid = self.configuration.make_sigmoid()

        if unscaled:  # set asymptotes to 0 for everything.
            lambd, gamma = 0, 0
        else:
            lambd, gamma = self.parameter_estimate['lambda'], self.parameter_estimate['gamma']
        new_threshold = sigmoid.inverse(percentage_correct, self.parameter_estimate['threshold'],
                                        self.parameter_estimate['width'], lambd, gamma)
        if not return_ci:
            return new_threshold

        param_cis = [self.confidence_intervals[param] for param in ('threshold', 'width', 'lambda', 'gamma')]
        if unscaled:  # set asymptotes to 0
            param_cis[2] = np.zeros_like(param_cis[2])
            param_cis[3] = np.zeros_like(param_cis[3])

        new_ci = []
        for (thres_ci, width_ci, lambd_ci, gamma_ci) in zip(*param_cis):
            ci_min = sigmoid.inverse(percentage_correct, thres_ci[0], width_ci[0], lambd_ci[0], gamma_ci[0])
            ci_max = sigmoid.inverse(percentage_correct, thres_ci[1], width_ci[1], lambd_ci[1], gamma_ci[1])
            new_ci.append([ci_min, ci_max])

        return new_threshold, new_ci

    def slope(self, stimulus_level: np.ndarray) -> np.ndarray:
        """ Slope of the psychometric function at a given stimulus levels.

        Args:
            stimulus_level: stimulus levels at where to evaluate the slope.
        Returns:
            Slopes of the psychometric function at the stimulus levels.
        """
        stimulus_level, param = np.asarray(stimulus_level), self.parameter_estimate
        sigmoid = self.configuration.make_sigmoid()
        return sigmoid.slope(stimulus_level, param['threshold'], param['width'], param['gamma'], param['lambda'])

    def slope_at_percentage_correct(self, percentage_correct: np.ndarray, unscaled: bool = False):
        """ Slope of the psychometric function at a given performance level in percent correct.

        Args:
            percentage_correct: percent correct at the threshold you want to calculate
            unscaled: If the percent correct you provide are for the unscaled
                      sigmoid or for the one scaled by lambda and gamma.
                      By default this function returns the one for the scaled sigmoid.
        Returns:
            Slopes of the psychometric function at the stimulus levels which
            correspond to the given percentage correct.
        """
        stimulus_levels = self.threshold(percentage_correct, unscaled, return_ci=False)
        return self.slope(stimulus_levels)
