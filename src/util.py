import numpy as np
import cv2
from typing import Union
from kdtuple import KDTuple


class Util:
    """!
        @brief Provides some helper methods
    """
    @staticmethod
    def calculate_area(points: np.ndarray) -> float:
        """!
        @brief Calculates area from given unordered coordinate points,
               representing the coordinates of a quadrilateral

        @param points Corners of the quadrilateral

        @return The area of quadrilateral
        """
        p1, p2, p3, p4 = points
        a1: float = p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p4[1] + p4[0]*p1[1]  # noqa E226
        a2: float = p1[1]*p2[0] + p2[1]*p3[0] + p3[1]*p4[0] + p4[1]*p1[0]  # noqa E226
        return 0.5 * (a1 - a2)

    @staticmethod
    def mean_shift_gray(image: np.ndarray, dest_mean: float) -> np.ndarray:
        """!
        @brief Applies a mean shift to given image

        @param image Image to apply mean shift
        @param dest_mean Destination mean

        @return Mean shifted image
        """
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        mean: float = np.mean(gray)
        delta_mean: float = mean - dest_mean
        B, G, R, A = cv2.split(image)

        def mean_shift(x: float):
            x -= delta_mean
            if x < 0:
                x = 0
            if x > 255:
                x = 255
            return x
        mean_shift = np.vectorize(mean_shift)

        B = mean_shift(B)
        G = mean_shift(G)
        R = mean_shift(R)

        image[:, :, 0] = B
        image[:, :, 1] = G
        image[:, :, 2] = R

        return image

    @staticmethod
    def match(
            matcher: Union[cv2.BFMatcher, cv2.FlannBasedMatcher],
            kd_left: KDTuple,
            kd_right: KDTuple,
            threshold: float = 0.75) -> np.ndarray:
        """
        @brief Performs a match between two KDTuples
        Performs a KnnMatch between kd_left.descriptor and kd_right.descriptor.
        Filters out the bad matches and returns the good matches as an np.ndarray

        @param matcher Matcher object
        @param threshold Filter threshold. Defaults to 0.75

        @return Good matches
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
