import cv2
import numpy as np
from homography_calculator import HomographyCalculator
from image_history import ImageHistory
from image_record import ImageRecord
from typing import Optional, Tuple
from kdtuple import KDTuple
from util import Util
import math


class Stitcher:
    """!
        @brief Stitcher is responsible with stitching provided images

        Stitching process goes like this:
        - If the provided image is the first image, set self.image to that image and stop
        - Otherwise, take the current image with last image and calculate the homography
          matrix between the images and applies it to the current image. Also calculates
          the where that image should be positioned in the final image.
        - To paste the image, stitcher first creates a new canvas and pastes the old image
          and the warped image to the correct position. There are two blending modes: legacy
          mode and the new mode. Legacy mode is just alpha blending. The new mode however,
          mixes the overlapping parts of two images by a predefined constant.
    """

    def __init__(self, *,
                 target_image_mean: int=110,
                 legacy_paste: bool=False,
                 new_image_percentage: float=0.75,
                 min_homography_determinant: float=0.1,
                 filter_threshold: float=0.65,
                 max_iterations_before_fail: int=5,
                 stitch_with_whole_image_on_fail: bool=True,
                 initial_random_point_count: int=16,
                 ransac_threshold: float=0.5,
                 initial_ransac_iterations=1250,
                 ransac_steps=500) -> None:
        """!
        @brief Creates a Stitcher object

        @param target_image_mean The target brightness value of input images
        @param legacy_paste Turn on legacy paste mode. The difference between legacy paste
               and new paste modes are explained in the class description.
        @param new_image_percentage Used when mixing two images in the new paste mode.
               Does a linear interpolation between the pixels of the old image and the new
               image. Essentially c*new + (1-c)*old where c is this constant.
        @param min_homography_determinant Minimum allowed value for the determinant of the
               homography matrix.
        @param filter_threshold A threshold value used to filter in the Util.match function.
        @param max_iterations_before_fail If a step fails, instead of erroring out, the stitcher
               tries stitching again with different values until a match is found. This is the
               number of tries before failure.
        @param stitch_with_whole_image_on_fail If stitcher fails to stitch and image with its
               predecessor image and this option is enabled, it tries to stitch it with the
               whole image and starts the whole process again.
        @param initial_random_point_count The ransac works by picking some random points and
               checking if they are "good". On each fail the stitcher decreases this value by
               4. However, the minimum point count is 4.
        @param ransac_threshold A threshold value to determine if a point is "good" in the ransac
               algorithm
        @param initial_ransac_iterations The initial iteration count for ransac
        @param ransac_steps After each failure, the iteration count increases by this value
        """
        # History related parameters
        self.history: ImageHistory = ImageHistory()
        self.full_image: Optional[np.ndarray] = None

        # Stitcher settings
        self.target_image_mean: int = target_image_mean
        self.legacy_paste: bool = legacy_paste
        self.new_image_percentage: float = new_image_percentage
        self.min_homography_determinant: float = min_homography_determinant
        self.filter_threshold: float = filter_threshold
        self.stitch_with_whole_image_on_fail: bool = stitch_with_whole_image_on_fail
        self.max_iterations_before_fail: int = max_iterations_before_fail
        self.initial_random_point_count: int = initial_random_point_count
        self.ransac_threshold: float = ransac_threshold
        self.initial_ransac_iterations: int = initial_ransac_iterations
        self.ransac_steps: int = ransac_steps

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

    def __preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """!
        @brief Preprocesses the given image.
        Converts it from BGR to BGRA and applies a mean shift to shift the
        mean of the image to the target_brightness parameter

        @param image: Input image

        @return Preprocessed image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image = Util.mean_shift_gray(image, self.target_image_mean)
        return image

    def stitch(self, image: np.ndarray, whole_image: bool = False) -> bool:
        """!
        @brief Stitches the given image to the full_image

               The process is explained in the class description.

        @param image: Image to stitch
        @param whole_image: If set to true, it uses full_image instead of the last stitched
                            image for stitching

        @return True if stitching was successful, False otherwise
        """
        current_image: np.ndarray = self.__preprocess_image(image)
        current_gray: np.ndarray = cv2.cvtColor(current_image, cv2.COLOR_BGRA2GRAY)

        if len(self.history) == 0:
            self.__paste_image(current_image)
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
        if whole_image:
            prev_gray = cv2.cvtColor(
                cv2.normalize(
                    self.full_image,
                    None, 0, 255, cv2.NORM_MINMAX
                ).astype('uint8'), cv2.COLOR_BGRA2GRAY
            )

        kd_left = self.__detect_and_compute(current_gray)
        kd_right = self.__detect_and_compute(prev_gray)

        # Homography calculation loop
        iterations: int = self.max_iterations_before_fail
        i: int = -1
        random_point_count: int = self.initial_random_point_count
        homography: Optional[np.ndarray] = None

        found_homography: bool = False
        new_corners: np.ndarray = np.array([])

        while i < iterations:
            i += 1
            if random_point_count < 4:
                random_point_count = 4

            # Match points
            matcher = self.flann_matcher
            if i >= 2:
                matcher = self.bf_matcher

            matches: np.ndarray = Util.match(
                matcher,
                kd_left,
                kd_right,
                self.filter_threshold
            )

            # If you can't find enough matches try again
            if len(matches) < random_point_count:
                random_point_count -= 4
                continue

            homography = HomographyCalculator.ransac(
                matches,
                random_point_count,
                self.ransac_threshold,
                self.initial_ransac_iterations + i * self.ransac_steps
            )

            # If you can't find a homography matrix try again
            if homography is None:
                random_point_count -= 4
                continue

            # Find where corners end up after transformation and use
            # that to calculate the ration between the area after the
            # transformation and the area before the transformation
            height, width, _ = current_image.shape
            corners: np.ndarray = np.array([
                [0, width,  width,      0],
                [0,     0, height, height],
                [1,     1,      1,      1]
            ])
            new_corners: np.ndarray = homography.dot(corners)
            new_corners /= new_corners[2]
            new_corners_T: np.ndarray = new_corners.T

            new_area: float = Util.calculate_area(new_corners_T)
            area_ratio: float = new_area / (width * height)

            # If you can't find a good homography matrix try again
            if area_ratio < self.min_homography_determinant:
                continue
            found_homography = True
            break

        # If you can't find a good homography matrix try again with whole image
        if not found_homography:
            if not whole_image and self.stitch_with_whole_image_on_fail:
                return self.stitch(image, True)
            return False

        # Calculates offsets of the image and builds the translation matrix
        x_offset: int = -np.min(new_corners[0])
        y_offset: int = -np.min(new_corners[1])

        translation_matrix: np.ndarray = np.array([
            [1.0, 0.0, x_offset],
            [0.0, 1.0, y_offset],
            [0.0, 0.0, 1.0]
        ])

        # Calculate the destination width and height
        destination_width: int = int(
            np.max(new_corners[0]) - np.min(new_corners[0])
        )
        destination_height: int = int(
            np.max(new_corners[1]) - np.min(new_corners[1])
        )

        assert homography is not None  # For typechecker
        homography = np.dot(translation_matrix, homography)

        assert homography is not None  # For typechecker
        # Warp the image
        warped_image: np.ndarray = cv2.warpPerspective(
            src=current_image,
            M=homography,
            dsize=(destination_width, destination_height)
        )

        # Paste the image and get the x and y coordinates of the image on
        # the self.full_image
        x_img, y_img = self.__paste_image(
            warped_image,
            int(x_offset),
            int(y_offset),
            whole_image
        )

        # Save image record to history
        warped_gray: np.ndarray = cv2.cvtColor(warped_image, cv2.COLOR_BGRA2GRAY)

        self.history.append(
            ImageRecord(
                current_image,
                warped_image,
                warped_gray,
                x_img,
                y_img,
                homography,
                None
            )
        )

        return True

    def __paste_image(
            self,
            image: np.ndarray,
            x_offset: int = 0,
            y_offset: int = 0,
            whole_image: bool = False) -> Tuple[int, int]:
        """!
        @brief Pastes the image to the full_image

        @param image Image to be pasted
        @param x_offset Horizontal offset from the last pasted image
        @param y_offset Vertical offset from the last pasted image
        @param whole_image If set to true, the offset applies to the whole image,
                            not to the last pasted image

        @return The total offset of the image as a tuple of ints
        """

        if self.full_image is None:
            # This is the first image
            self.full_image = image
            return 0, 0

        height, width, channels = self.full_image.shape
        img_height, img_width, img_channels = image.shape

        last_record: ImageRecord = self.history[-1]
        last_x: int = last_record.x_warped
        last_y: int = last_record.y_warped

        if whole_image:
            last_x = last_y = 0

        x_left: int = math.floor(min(0, last_x - x_offset))
        y_top: int = math.floor(min(0, last_y - y_offset))
        x_right: int = math.ceil(max(width, last_x - x_offset + img_width))
        y_bottom: int = math.ceil(max(height, last_y - y_offset + img_height))

        new_width: int = x_right - x_left
        new_height: int = y_bottom - y_top

        current_image: np.ndarray = self.full_image
        x_full_image: int = -x_left
        y_full_image: int = -y_top

        # Create a new canvas and paste the full image
        self.full_image = np.zeros(shape=(new_height, new_width, channels))
        self.full_image[
            y_full_image:y_full_image + height,
            x_full_image:x_full_image + width,
            :
        ] = current_image

        # x and y coordinates of the parameter "image"
        x_img: int = x_full_image + last_x - x_offset
        y_img: int = y_full_image + last_y - y_offset

        if self.legacy_paste:
            alpha_new = image[:, :, 3] / 255

            for c in range(channels):
                self.full_image[
                    y_img:y_img + img_height,
                    x_img:x_img + img_width,
                    c
                ] = alpha_new * image[:, :, c] + self.full_image[
                        y_img:y_img + img_height,
                        x_img:x_img + img_width,
                        c
                    ] * (1 - alpha_new)
        else:
            image_filter = image.copy()
            image_filter[image_filter > 0] = 1

            full_image_filter = self.full_image[y_img:y_img + img_height, x_img:x_img + img_width, :].copy()
            full_image_filter[full_image_filter > 0] = 1

            image_filter = (image_filter + full_image_filter)
            full_image_filter = image_filter.copy()

            image_filter[image_filter > 1] = self.new_image_percentage
            full_image_filter[full_image_filter > 1] = 1.0 - self.new_image_percentage

            self.full_image[
                y_img:y_img + img_height,
                x_img:x_img + img_width,
                :
            ] = image_filter * image[:, :, :] + self.full_image[
                    y_img:y_img + img_height,
                    x_img:x_img + img_width,
                    :
            ] * full_image_filter

        return x_img, y_img

    def __detect_and_compute(self, image: np.ndarray) -> KDTuple:
        """!
        @brief Detect and compute keypoints and descriptors of given image using SIFT

        @param image Image to find keypoints and descriptors
        @return Keypoints and descriptors as a KDTuple object
        """
        kpts, descriptors = self.sift_extractor.detectAndCompute(image, None)
        return KDTuple(kpts, descriptors)
