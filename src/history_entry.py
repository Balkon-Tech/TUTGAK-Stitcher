import datetime as dt
import numpy as np


class HistoryEntry:
    def __init__(self,
                 original_image: np.ndarray,
                 warped_image: np.ndarray,
                 x_warped: int,
                 y_warped: int,
                 homography: np.ndarray,
                 metadata: str) -> None:
        """!
        @brief Creates a new HistoryEntry object

        @param original_image Original, untransformed image
        @param warped_image Image after warping
        @param x_warped X coordinate of the warped image on the final image
        @param y_warped Y coordinate of the warped image on the final image
        @param homography The homography matrix that is used to transform
               original image to warped_image
        @param metadata Metadata that you want to associate with this object
        """

        self.original_image: np.ndarray = original_image
        self.warped_image: np.ndarray = warped_image
        self.x_warped: int = x_warped
        self.y_warped: int = y_warped
        self.homography: np.ndarray = homography
        self.metadata: str = metadata

        self.datetime: dt.datetime = dt.datetime.now()
        self.__det = np.linalg.det(self.homography[:3, :3])

    def __str__(self) -> str:
        """!
        @brief Converts an HistoryEntry object to string

        @return String representation of HistoryEntry object
        """
        return f"Determinant: {self.__det}, Warped Image Coords: " \
            "({self.x_warped}, {self.y_warped}), Metadata: {self.metadata}"
