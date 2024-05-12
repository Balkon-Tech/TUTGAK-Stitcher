from dataclasses import dataclass
import numpy as np


@dataclass
class KDTuple:
    keypoints: tuple
    descriptors: np.ndarray
