import cv2
import numpy as np
from homography_calculator import HomographyCalculator
from image_history import ImageHistory
from image_record import ImageRecord
from typing import Optional, Union
from kdtuple import KDTuple
from matcher import Matcher


class Stitcher:
    """!
        Stitcher is responsible with stitching provided images
    """

    def __init__(self) -> None:
        """
        Creates a Stitcher object
        """
        self.history: ImageHistory = ImageHistory()
        self.full_image: Union[None, np.ndarray] = None

        # Matchers
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann_matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )
        self.bf_matcher: cv2.BFMatcher = cv2.BFMatcher.create()

        # SIFT feature extractor
        self.sift_extractor: cv2.SIFT = cv2.SIFT.create()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses image

        :param image: Input image
        :return: Preprocessed image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        return image

    def stitch(self, image: np.ndarray, whole_image=False) -> bool:
        current_image: np.ndarray = self.preprocess_image(image)
        current_gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        if len(self.history) == 0:
            # self.paste_image(image)
            self.history.append(
                ImageRecord(
                    current_image,
                    current_image,
                    current_gray,
                    0,
                    0,
                    np.identity(3),
                    None
                )
            )
            return True

        last_record: ImageRecord = self.history[-1]

        # Computes keypoints and descriptors using sift
        prev_gray: np.ndarray = last_record.warped_gray
        kd_left = self.detectAndCompute(current_gray)
        kd_right = self.detectAndCompute(prev_gray)

        initial_random_point_count: int = 16
        iterations: int = 5
        i: int = 0
        random_point_count: int = initial_random_point_count
        # homography_det: float = 1.0
        homography: Optional[np.ndarray] = None

        while i < iterations:
            if random_point_count < 4:
                random_point_count = 4

            matches: Optional[np.ndarray] = None
            if i < 2:
                matches = Matcher.match(
                    self.flann_matcher,
                    kd_left,
                    kd_right,
                    0.65
                )
            else:
                matches = Matcher.match(
                    self.bf_matcher,
                    kd_left,
                    kd_right,
                    0.65
                )

            if len(matches) < random_point_count:
                random_point_count -= 4
                continue

            homography = HomographyCalculator.ransac(
                matches,
                random_point_count,
                0.5,
                1250 + i * 500
            )

            if homography is None:
                random_point_count -= 4
                continue

            height, width, channels = current_image.shape
            corners: np.ndarray = np.array([
                [0, width,  width,      0],
                [0,     0, height, height],
                [1,     1,      1,      1]
            ])
            corners_new: np.ndarray = homography.dot(corners)
            corners_new_T: np.ndarray = corners_new.T
            corners_new_T /= corners_new_T[2]

            # TODO: Continue here

            i += 1

        return True

    def detectAndCompute(self, image: np.ndarray) -> KDTuple:
        kpts, descriptors = self.sift_extractor.detectAndCompute(image, None)
        return KDTuple(kpts, descriptors)
