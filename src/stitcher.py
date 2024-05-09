import cv2
import numpy as np
from homography_calculator import HomographyCalculator
from image_history import ImageHistory
from image_record import ImageRecord
from typing import Optional
from kdtuple import KDTuple
from matcher import Matcher
from util import Util


class Stitcher:
    """!
        Stitcher is responsible with stitching provided images
    """

    def __init__(self) -> None:
        """
        Creates a Stitcher object
        """
        self.history: ImageHistory = ImageHistory()
        self.full_image: Optional[np.ndarray] = None
        # self.last_x_offset: int = 0
        # self.last_y_offset: int = 0

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
            self.paste_image(image)
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
        i: int = -1
        random_point_count: int = initial_random_point_count
        # homography_det: float = 1.0
        homography: Optional[np.ndarray] = None

        found_match: bool = False
        new_corners: np.ndarray = np.array([])

        while i < iterations:
            i += 1
            if random_point_count < 4:
                random_point_count = 4

            # Match points
            matcher = self.flann_matcher
            if i >= 2:
                matcher = self.bf_matcher

            matches: np.ndarray = Matcher.match(
                matcher,
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
            new_corners: np.ndarray = homography.dot(corners)
            new_corners_T: np.ndarray = new_corners.T
            new_corners_T /= new_corners_T[2]

            new_area: float = Util.calculate_area(new_corners_T)
            determinant: float = new_area / (width * height)

            if determinant < 0.1:
                continue
            found_match = True
            break

        if not found_match:
            if not whole_image:
                return self.stitch(image, True)
            return False

        assert (homography is not None)
        # Calculates offsets of the image
        x_offset: int = -np.min(new_corners[0])
        y_offset: int = -np.min(new_corners[1])

        translation_matrix: np.ndarray = np.array([
            [1.0, 0.0, x_offset],
            [0.0, 1.0, y_offset],
            [0.0, 0.0, 1.0]
        ])

        destination_width: int = int(
            np.max(new_corners[0]) - np.min(new_corners[0])
        )
        destination_height: int = int(
            np.max(new_corners[1]) - np.min(new_corners[1])
        )

        homography = np.dot(translation_matrix, homography)
        warped_image: np.ndarray = cv2.warpPerspective(
            src=image,
            M=homography,
            dsize=(destination_width, destination_height)
        )

        self.paste_image(
            warped_image,
            int(x_offset),
            int(y_offset),
            whole_image
        )

        warped_gray: np.ndarray = cv2.cvtColor(warped_image, cv2.COLOR_BGRA2GRAY)

        self.history.append(
            ImageRecord(
                image,
                warped_image,
                warped_gray,
                x_offset,
                y_offset,
                homography,
                None
            )
        )

        return True

    def paste_image(
            self,
            image: np.ndarray,
            x_offset: int = 0,
            y_offset: int = 0,
            whole_image: bool = False):

        if self.full_image is None:
            # This is the first image
            self.full_image = image
            return

        # TODO: Implement here

    def detectAndCompute(self, image: np.ndarray) -> KDTuple:
        kpts, descriptors = self.sift_extractor.detectAndCompute(image, None)
        return KDTuple(kpts, descriptors)
