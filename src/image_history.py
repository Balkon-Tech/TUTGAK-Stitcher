from history_entry import HistoryEntry


class ImageHistory(list[HistoryEntry]):
    """!
        An ImageHistory object contiguously stores HistoryEntry
        Objects
    """

    def __init__(self, *args) -> None:
        """!
        Creates an ImageHistory object
        """
        super().__init__(*args)
