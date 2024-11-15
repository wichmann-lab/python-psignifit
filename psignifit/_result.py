import dataclasses
import json
from typing import Any, Dict, Tuple, List, Literal, Optional, TextIO, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._configuration import Configuration


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclasses.dataclass
class Result:
    parameter_estimate: Dict[str, float]
    parameter_estimate_mean: Dict[str, float]
    configuration: Configuration
    confidence_intervals: Dict[str, List[Tuple[float, float]]]
    data: NDArray[float]
    parameter_values: Dict[str, NDArray[float]]
    prior_values: Dict[str, NDArray[float]]
    marginal_posterior_values: Dict[str, NDArray[float]]
    debug: Dict[Any, Any]
    estimate_type: Literal['MAP', 'mean'] = 'MAP'

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

    def get_parameter_estimate(self, estimate_type: Optional[str]=None):
        """ Get the estimate of the parameters by estimate type.

        Args:
            estimate_type: Type of the parameters estimate, either "MAP" or "mean". If None, the value of
            `Result.estimate_type` is used instead.
        Returns:
            A dictionary mapping parameter names to parameter estimate.
        """
        if estimate_type is None:
            estimate_type = self.estimate_type

        if estimate_type == 'MAP':
            estimate = self.parameter_estimate
        elif estimate_type == 'mean':
            estimate = self.parameter_estimate_mean
        else:
            raise ValueError("`estimate_type` must be either 'MAP' or 'mean'")

        return estimate

    def threshold(self, proportion_correct: np.ndarray, unscaled: bool = False, return_ci: bool = True
                  ) -> Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
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
        Returns:
            thresholds: stimulus values for all provided proportion_correct (if return_ci=False)
            (thresholds, ci): stimulus values along with confidence intervals

        """
        proportion_correct = np.asarray(proportion_correct)
        sigmoid = self.configuration.make_sigmoid()

        if unscaled:  # set asymptotes to 0 for everything.
            lambd, gamma = 0, 0
        else:
            lambd, gamma = self.parameter_estimate['lambda'], self.parameter_estimate['gamma']
        new_threshold = sigmoid.inverse(proportion_correct, self.parameter_estimate['threshold'],
                                        self.parameter_estimate['width'], gamma, lambd)
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

    def slope_at_proportion_correct(self, proportion_correct: np.ndarray, unscaled: bool = False):
        """ Slope of the psychometric function at a given performance level in proportion correct.

        Args:
            proportion_correct: proportion correct at the threshold you want to calculate
            unscaled: If the proportion correct you provide are for the unscaled
                      sigmoid or for the one scaled by lambda and gamma.
                      By default this function returns the one for the scaled sigmoid.
        Returns:
            Slopes of the psychometric function at the stimulus levels which
            correspond to the given proportion correct.
        """
        stimulus_levels = self.threshold(proportion_correct, unscaled, return_ci=False)
        return self.slope(stimulus_levels)
