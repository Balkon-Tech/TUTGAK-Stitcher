import numpy as np


class Util:
    @staticmethod
    def calculate_area(points: np.ndarray) -> float:
        """
        Calculates area from given unordered coordinate points

        :param points: Corners of the polygon
        """
        p1, p2, p3, p4 = points
        a1: float = p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p4[1] + p4[0]*p1[1]  # noqa E226
        a2: float = p1[1]*p2[0] + p2[1]*p3[0] + p3[1]*p4[0] + p4[1]*p1[0]  # noqa E226
        return 0.5 * (a1 - a2)
