import dataclasses
import json
from typing import Any, Dict, Tuple, List, Optional, TextIO, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._configuration import Configuration
from ._typing import EstimateType
from .tools import psychometric_with_eta

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclasses.dataclass
class Result:
    parameter_estimate_MAP: Dict[str, float]
    parameter_estimate_mean: Dict[str, float]
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

    def get_parameter_estimate(self, estimate_type: Optional[EstimateType]=None):
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
            estimate = self.parameter_estimate_MAP
        elif estimate_type == 'mean':
            estimate = self.parameter_estimate_mean
        else:
            raise ValueError("`estimate_type` must be either 'MAP' or 'mean'")

        return estimate

    parameter_estimate = property(get_parameter_estimate)

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
                      By default, this function returns the one for the scaled sigmoid.
            return_ci: If the confidence bounds should be returned along with the stimulus value
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
        Returns:
            thresholds: stimulus values for all provided proportion_correct (if return_ci=False)
            (thresholds, ci): stimulus values along with confidence intervals

        """
        proportion_correct = np.asarray(proportion_correct)
        sigmoid = self.configuration.make_sigmoid()

        estimate = self.get_parameter_estimate(estimate_type)
        if unscaled:  # set asymptotes to 0 for everything.
            lambd, gamma = 0, 0
        else:
            lambd, gamma = estimate['lambda'], estimate['gamma']
        new_threshold = sigmoid.inverse(proportion_correct, estimate['threshold'],
                                        estimate['width'], gamma, lambd)
        if not return_ci:
            return new_threshold

        new_threshold_ci = {}
        for coverage_key in self.confidence_intervals['threshold'].keys():
            thres_ci = self.confidence_intervals['threshold'][coverage_key]
            width_ci = self.confidence_intervals['width'][coverage_key]
            if unscaled:
                gamma_ci = np.array([0.0, 0.0])
                lambd_ci = np.array([0.0, 0.0])
            else:
                gamma_ci = self.confidence_intervals['gamma'][coverage_key]
                lambd_ci = self.confidence_intervals['lambda'][coverage_key]
            ci_min = sigmoid.inverse(proportion_correct, thres_ci[0], width_ci[0], gamma_ci[0], lambd_ci[0])
            ci_max = sigmoid.inverse(proportion_correct, thres_ci[1], width_ci[1], gamma_ci[1], lambd_ci[1])
            new_threshold_ci[coverage_key] = [ci_min, ci_max]

        return new_threshold, new_threshold_ci

    def slope(self, stimulus_level: np.ndarray, estimate_type: Optional[EstimateType]=None) -> np.ndarray:
        """ Slope of the psychometric function at a given stimulus levels.

        Args:
            stimulus_level: stimulus levels at where to evaluate the slope.
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
        Returns:
            Slopes of the psychometric function at the stimulus levels.
        """
        stimulus_level, param = np.asarray(stimulus_level), self.get_parameter_estimate(estimate_type)
        sigmoid = self.configuration.make_sigmoid()
        return sigmoid.slope(stimulus_level, param['threshold'], param['width'], param['gamma'], param['lambda'])

    def proportion_correct(self, stimulus_level: np.ndarray,  with_eta: bool = False,
                           estimate_type: Optional[EstimateType]=None, random_state=None) -> np.ndarray:
        """ Proportion correct corresponding to the given stimulus levels for the fitted psychometric function.

        Args:
            stimulus_level: stimulus levels.
            with_eta: if set, after computing proportion correct values, add noise
                to the data so that its variance is compatible with the estimated
                overdispersion parameter `eta`.
            estimate_type: Type of estimate, either "MAP" or "mean".
                If None, the value of `estimate_type` in `Result.configuration` is used instead.
            random_state: Random state used to generate the additional variance in the data if with_eta is True.
                If None, NumPy's default random number generator is used.
        Returns:
            Proportion correct values for the fitted psychometric function at the given stimulus levels.
        """
        stimulus_level, param = np.asarray(stimulus_level), self.get_parameter_estimate(estimate_type)
        sigmoid = self.configuration.make_sigmoid()
        if with_eta:
            out = psychometric_with_eta(stimulus_level, param['threshold'], param['width'],
                                        param['gamma'], param['lambda'], sigmoid,
                                        param['eta'], random_state=random_state)
        else:
            out = sigmoid(stimulus_level, threshold=param['threshold'], width=param['width'],
                          gamma=param['gamma'], lambd=param['lambda'])

        return out

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

    def standard_parameter_estimate(self, estimate_type: Optional[EstimateType]=None):
        """ Get the parameters of the psychometric function in the standard format.

        `psignifit` uses the same intuitive parametrization, threshold and width, for all
        sigmoid types. However, each different type of sigmoid has its own standard parametrization.

        The interpretation of the standard parameters, location and scale, depends on the sigmoid class used.
        For instance, for a Gaussian sigmoid, the location corresponds to the mean and the scale to the standard
        deviation of the distribution.

        For negative slope sigmoids, we return the same parameters as for the positive ones.

        Args:
            proportion_correct: proportion correct at the threshold you want to calculate
        Returns:
            Standard parameters (loc, scale) for the sigmoid subclass.
        """
        sigmoid = self.configuration.make_sigmoid()
        estimate = self.get_parameter_estimate(estimate_type)
        loc, scale = sigmoid.standard_parameters(estimate['threshold'], estimate['width'])
        return loc, scale

    def posterior_samples(self, n_samples, random_state=None):
        """ Get samples from the posterior over parameters.

        Return parameters values as drawn at random from the posterior of the parameters.

        The posterior information is only available if the sigmoid has been fit with `debug=True`. If the
        information is missing, an exception is raised.

        Args:
            n_samples: Number of samples to return
            random_state: np.RandomState
                Random state used to generate the samples from the posterior.
                If None, NumPy's default random number generator is used.
        Returns:
            Dictionary mapping parameter names to an array of parameter values, as drawn at random from the
            posterior over parameters.
        Raises:
            ValueError if the sigmoid has been fit with `debug=False`
        """

        if len(self.debug) == 0:
            raise ValueError("Expects `posteriors` in results, got `None`. Run the sigmoid fit with "
                             "`psignifit(..., debug=True)`.")

        if random_state is None:
            random_state = np.random.default_rng()

        values = self.parameter_values
        posterior = self.debug['posteriors']
        params_grid = np.meshgrid(*(values.values()), indexing='ij')
        params_combos = np.dstack([p.flatten() for p in params_grid]).squeeze()

        # Sample from the posterior
        n_params_combos = params_combos.shape[0]
        samples_idx = random_state.choice(
            np.arange(n_params_combos),
            size=(n_samples,),
            replace=True,
            p=posterior.flatten()
        )
        samples_params_combo = params_combos[samples_idx]

        samples = {}
        for param_idx, param in enumerate(values.keys()):
            samples[param] = samples_params_combo[:, param_idx]

        return samples
