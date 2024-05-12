import datetime as dt
import numpy as np
from typing import Any


class ImageRecord:
    """!
        @brief Represents a state of the stitcher

        An ImageRecord object, holds some information about a
        particular state of the Stitcher, such as the homography matrix,
        the original image, etc.
    """

    def __init__(self,
                 original_image: np.ndarray,
                 warped_image: np.ndarray,
                 warped_gray: np.ndarray,
                 x_warped: int,
                 y_warped: int,
                 homography: np.ndarray,
                 metadata: Any) -> None:
        """!
        @brief Creates a new ImageRecord object

        @param original_image Original, untransformed image
        @param warped_image Image after warping
        @param warped_gray Image after warping, gray
        @param x_warped X coordinate of the warped image on the final image
        @param y_warped Y coordinate of the warped image on the final image
        @param homography The homography matrix that is used to transform
               original image to warped_image
        @param metadata Extra data to hold with the current state
        """

        self.original_image: np.ndarray = original_image
        self.warped_image: np.ndarray = warped_image
        self.warped_gray: np.ndarray = warped_gray
        self.x_warped: int = x_warped
        self.y_warped: int = y_warped
        self.homography: np.ndarray = homography
        self.metadata: str = metadata

        self.datetime: dt.datetime = dt.datetime.now()
        self.__det = np.linalg.det(self.homography[:3, :3])

    def __str__(self) -> str:
        """!
        @brief Converts an ImageRecord object to a string

        @return String representation of the ImageRecord object
        """
        return f"Determinant: {self.__det}, Warped Image Coords: " \
            f"({self.x_warped}, {self.y_warped}), Metadata: {self.metadata}"
