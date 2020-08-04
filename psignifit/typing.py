import enum
from typing import Dict, Tuple, Optional


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
except ImportError:
    # Fallback for Python < 3.8
    ParameterBounds = Dict[str, Optional[Tuple[float, float]]]
