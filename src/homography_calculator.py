import numpy as np
from typing import Optional


class HomographyCalculator:
    """!
        Provides methods for calculating homography matrices and
        measuring their error.
    """
    @staticmethod
    def calculate_homography_error(
            points: np.ndarray,
            H: np.ndarray) -> np.ndarray:
        """
        Calculates an error value for the given homography matrix

        :param points: Point pairs
        :param H: Homography matrix
        :return: Error of each point
        """
        num_points: int = len(points)
        all_p1: np.ndarray = np.concatenate(
            (points[:, 0:2], np.ones((num_points, 1))),
            axis=1
        )
        all_p2: np.ndarray = points[:, 2:4]

        estimate_p2: np.ndarray = np.zeros((num_points, 2))
        i = 0
        while i < num_points:
            estimated: np.ndarray = np.dot(H, all_p1[i])
            estimate_p2[i] = (estimated / estimated[2])[0:2]
            i += 1

        errors: np.ndarray = np.linalg.norm(
            all_p2 - estimate_p2,
            axis=1
        ) ** 2

        return errors

    @staticmethod
    def calculate_homography(pairs: np.ndarray) -> np.ndarray:
        """
        Calculates the homography matrix for the given pairs

        :param pairs: Matching pairs of coordinates
        :return: Homography matrix
        """
        rows_list: list[list[float]] = []

        i = 0
        while i < pairs.shape[0]:
            p1: np.ndarray = np.append(pairs[i][0:2], 1)
            p2: np.ndarray = np.append(pairs[i][2:4], 1)

            row1: list[float] = [
                0, 0, 0, p1[0], p1[1], p1[2],
                -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]
            ]
            row2: list[float] = [
                p1[0], p1[1], p1[2], 0, 0, 0,
                -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]
            ]

            rows_list.append(row1)
            rows_list.append(row2)
            i += 1

        rows: np.ndarray = np.array(rows_list)
        _, _, V = np.linalg.svd(rows)
        H = V[-1].reshape(3, 3)
        H /= H[2, 2]
        return H

    @staticmethod
    def ransac(
            matches: np.ndarray,
            random_point_count: int,
            threshold: float,
            iterations: int) -> Optional[np.ndarray]:
        """
        Use RANSAC to calculate a good homography matrix

        :param matches: Matched points
        :param random_point_count: The amount of random points used to
                                   calculate the homography matrix
        :param threshold: Error threshold
        :param iterations: Number of iterations
        :return: A good homography matrix, or None if none could be found
        """
        num_best_inliers: int = 0
        best_inliers: Optional[np.ndarray] = None

        i = 0
        while i < iterations:
            points: np.ndarray = np.random.choice(matches,
                                                  replace=False,
                                                  size=random_point_count
                                                  )
            H = HomographyCalculator.calculate_homography(points)

            if np.linalg.matrix_rank(H):
                continue

            errors: np.ndarray = HomographyCalculator.calculate_homography_error(  # noqa E501
                points, H
            )
            idx: np.ndarray = np.where(errors < threshold)[0]
            inliers: np.ndarray = matches[idx]

            num_inliers: int = len(inliers)
            if num_inliers > num_best_inliers:
                best_inliers = inliers.copy()
                num_best_inliers = num_inliers

            i += 1

        if best_inliers is None:
            return None
        H = HomographyCalculator.calculate_homography(best_inliers)
        H /= H[2, 2]
        return H
