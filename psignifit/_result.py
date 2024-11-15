import dataclasses
import json
from typing import Any, Dict, Tuple, List, Optional, TextIO, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._configuration import Configuration
from ._typing import EstimateType


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclasses.dataclass
class Result:
    parameters_estimate_MAP: Dict[str, float]
    parameters_estimate_mean: Dict[str, float]
    configuration: Configuration
    confidence_intervals: Dict[str, List[Tuple[float, float]]]
    data: NDArray[float]
    parameter_values: Dict[str, NDArray[float]]
    prior_values: Dict[str, NDArray[float]]
    marginal_posterior_values: Dict[str, NDArray[float]]
    debug: Dict[Any, Any]

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
            # converts data automatically to lists of lists
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

        result_dict['parameter_values'] = {
            k: np.asarray(v) for k, v in result_dict['parameter_values'].items()}
        result_dict['prior_values'] = {
            k: np.asarray(v) for k, v in result_dict['prior_values'].items()}
        result_dict['marginal_posterior_values'] = {
            k: np.asarray(v) for k, v in result_dict['marginal_posterior_values'].items()}
        result_dict['data'] = np.asarray(result_dict['data'])
        return cls.from_dict(result_dict)

    def get_parameters_estimate(self, estimate_type: Optional[EstimateType]=None):
        """ Get the estimate of the parameters by type.

        Args:
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
        Returns:
            A dictionary mapping parameter names to parameter estimate.
        """
        if estimate_type is None:
            estimate_type = self.configuration.estimate_type

        if estimate_type == 'MAP':
            estimate = self.parameters_estimate_MAP
        elif estimate_type == 'mean':
            estimate = self.parameters_estimate_mean
        else:
            raise ValueError("`estimate_type` must be either 'MAP' or 'mean'")

        return estimate

    def threshold(self, proportion_correct: np.ndarray, unscaled: bool = False, return_ci: bool = True,
                  estimate_type: Optional[EstimateType]=None) -> Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """ Threshold stimulus value and confidence interval for a different proportion correct cutoff.

        The CIs you may obtain from this are calculated based on the confidence
        intervals only, e.g. with the shallowest and the steepest psychometric
        function and may thus broaden if you move away from the standard
        threshold of unscaled sigmoid. These CI are only upper bounds.

        For a more accurate inference, refit psignifit using the proportion correct in the configuration.

        Args:
            proportion_correct: proportion correct at the threshold you want to calculate
            unscaled: If the proportion correct you provide are for the unscaled
                      sigmoid or for the one scaled by lambda and gamma.
                      By default this function returns the one for the scaled sigmoid.
            return_ci: If the confidence bounds should be returned along with the stimulus value
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
        Returns:
            thresholds: stimulus values for all provided proportion_correct (if return_ci=False)
            (thresholds, ci): stimulus values along with confidence intervals

        """
        proportion_correct = np.asarray(proportion_correct)
        sigmoid = self.configuration.make_sigmoid()

        estimate = self.get_parameters_estimate(estimate_type)
        if unscaled:  # set asymptotes to 0 for everything.
            lambd, gamma = 0, 0
        else:
            lambd, gamma = estimate['lambda'], estimate['gamma']
        new_threshold = sigmoid.inverse(proportion_correct, estimate['threshold'],
                                        estimate['width'], gamma, lambd)
        if not return_ci:
            return new_threshold

        param_cis = [self.confidence_intervals[param] for param in ('threshold', 'width', 'lambda', 'gamma')]
        if unscaled:  # set asymptotes to 0
            param_cis[2] = np.zeros_like(param_cis[2])
            param_cis[3] = np.zeros_like(param_cis[3])

        new_ci = []
        for (thres_ci, width_ci, lambd_ci, gamma_ci) in zip(*param_cis):
            ci_min = sigmoid.inverse(proportion_correct, thres_ci[0], width_ci[0], gamma_ci[0], lambd_ci[0])
            ci_max = sigmoid.inverse(proportion_correct, thres_ci[1], width_ci[1], gamma_ci[1], lambd_ci[1])
            new_ci.append([ci_min, ci_max])

        return new_threshold, new_ci

    def slope(self, stimulus_level: np.ndarray, estimate_type: Optional[EstimateType]=None) -> np.ndarray:
        """ Slope of the psychometric function at a given stimulus levels.

        Args:
            stimulus_level: stimulus levels at where to evaluate the slope.
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
        Returns:
            Slopes of the psychometric function at the stimulus levels.
        """
        stimulus_level, param = np.asarray(stimulus_level), self.get_parameters_estimate(estimate_type)
        sigmoid = self.configuration.make_sigmoid()
        return sigmoid.slope(stimulus_level, param['threshold'], param['width'], param['gamma'], param['lambda'])

    def slope_at_proportion_correct(self, proportion_correct: np.ndarray, unscaled: bool = False,
                                    estimate_type: Optional[EstimateType]=None):
        """ Slope of the psychometric function at a given performance level in proportion correct.

        Args:
            proportion_correct: proportion correct at the threshold you want to calculate
            unscaled: If the proportion correct you provide are for the unscaled
                      sigmoid or for the one scaled by lambda and gamma.
                      By default this function returns the one for the scaled sigmoid.
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
        Returns:
            Slopes of the psychometric function at the stimulus levels which
            correspond to the given proportion correct.
        """
        stimulus_levels = self.threshold(proportion_correct, unscaled, return_ci=False, estimate_type=estimate_type)
        return self.slope(stimulus_levels)
