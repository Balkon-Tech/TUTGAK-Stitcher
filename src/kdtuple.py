from dataclasses import dataclass
import numpy as np


@dataclass
class KDTuple:
    """!
    @brief A dataclass that holds keypoints and descriptors
    """
    keypoints: tuple
    descriptors: np.ndarray
