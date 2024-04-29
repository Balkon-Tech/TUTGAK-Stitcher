from dataclasses import dataclass
import numpy as np


@dataclass
class KDTuple:
    keypoint: tuple
    descriptor: np.ndarray
