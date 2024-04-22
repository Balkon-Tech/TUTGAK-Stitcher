from stitcher_state import StitcherState


class StateHistory(list[StitcherState]):
    """!
        A StateHistory object contiguously stores StitcherState
        objects
    """

    def __init__(self, *args) -> None:
        """!
            Creates a StateHistory object
        """
        super().__init__(*args)
