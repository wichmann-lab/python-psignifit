import enum
from typing import Callable, Dict, Tuple, Optional

import numpy as np


class ExperimentType(enum.Enum):
    YES_NO = 'yes/no'
    N_AFC = 'nAFC'
    EQ_ASYMPTOTE = 'equal asymptote'


try:
    from typing import TypedDict

    class ParameterBounds(TypedDict):
        threshold: Tuple[float, float]
        width: Tuple[float, float]
        lambda: Tuple[float, float]
        gamma: Optional[Tuple[float, float]]
        eta: Tuple[float, float]

    class ParameterGrid(TypedDict):
        threshold: np.ndarray
        width: np.ndarray
        lambda: np.ndarray
        gamma: Optional[np.ndarray]
        eta: np.ndarray
except ImportError:
    # Fallback for Python < 3.8
    ParameterBounds = Dict[str, Optional[Tuple[float, float]]]
    ParameterGrid = Dict[str, Optional[np.ndarray]]


Prior = Callable[[np.ndarray], np.ndarray]
