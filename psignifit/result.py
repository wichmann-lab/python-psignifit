import dataclasses
from typing import Any, Dict, Tuple, List, TextIO, Union
import json
from pathlib import Path

import numpy as np

from .configuration import Configuration


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclasses.dataclass
class Result:
    sigmoid_parameters: Dict[str, float]
    configuration: Configuration
    confidence_intervals: Dict[str, List[Tuple[float, float]]]
    posterior: np.ndarray = dataclasses.field(compare=False)
    # If future attributes should contain numpy arrays,
    # run np.asarray in __post_init__.
    # Otherwise, load_json may result in nested lists instead.
    def __post_init__(self):
         self.posterior = np.asarray(self.posterior)

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
