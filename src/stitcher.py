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
        self.sift_matcher: cv2.SIFT = cv2.SIFT.create()
        self.bf_matcher: cv2.BFMatcher = cv2.BFMatcher.create()
