import cv2
from image_history import ImageHistory


class Stitcher:
    """!
        A Stitcher is responsible with stitching your images
    """

    def __init__(self) -> None:
        """
        Creates a Stitcher object
        """
        self.history: ImageHistory = ImageHistory()

        # Matchers
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann_matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )
        self.bf_matcher: cv2.BFMatcher = cv2.BFMatcher.create()
