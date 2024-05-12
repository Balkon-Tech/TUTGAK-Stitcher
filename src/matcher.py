import cv2
import numpy as np
from typing import Union
from kdtuple import KDTuple


class Matcher:
    """!
        TODO: Doc here
    """

    @staticmethod
    def match(
            matcher: Union[cv2.BFMatcher, cv2.FlannBasedMatcher],
            kd_left: KDTuple,
            kd_right: KDTuple,
            threshold: float = 0.75) -> np.ndarray:
        """
        Performs a KnnMatch between kd_left.descriptor and kd_right.descriptor.
        Filters out the bad matches and returns the matches as np.ndarray

        :param matcher: Matcher object
        :param threshold: Filter threshold. Defaults to 0.75
        """
        matches = matcher.knnMatch(
            kd_left.descriptors, kd_right.descriptors, k=2
        )

        # Apply ratio test
        good: list = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append([m])

        matches = []
        for pair in good:
            matches.append(list(
                kd_left.keypoints[pair[0].queryIdx].pt +
                kd_right.keypoints[pair[0].trainIdx].pt
            ))

        return np.array(matches)
