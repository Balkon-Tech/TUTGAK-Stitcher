from image_record import ImageRecord


class ImageHistory(list[ImageRecord]):
    """!
        A ImageHistory object contiguously stores ImageRecord
        objects
    """

    def __init__(self, *args) -> None:
        """!
            Creates a ImageHistory object
        """
        super().__init__(*args)
