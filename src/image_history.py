from history_entry import HistoryEntry


class ImageHistory(list[HistoryEntry]):
    def __init__(self, *args) -> None:
        super().__init__(*args)
