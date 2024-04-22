import cv2
from state_history import StateHistory


class Stitcher:
    """!
        Stitcher is responsible with stitching provided images
    """

    def __init__(self) -> None:
        """
        Creates a Stitcher object
        """
        self.history: StateHistory = StateHistory()

        # Matchers
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann_matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )
        self.bf_matcher: cv2.BFMatcher = cv2.BFMatcher.create()
