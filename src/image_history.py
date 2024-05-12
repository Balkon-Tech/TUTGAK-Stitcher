from image_record import ImageRecord


class ImageHistory(list[ImageRecord]):
    """!
        @brief An ImageHistory object contiguously stores ImageRecord
        objects
    """

    def __init__(self, *args) -> None:
        """!
            @brief Creates an ImageHistory object
        """
        super().__init__(*args)
