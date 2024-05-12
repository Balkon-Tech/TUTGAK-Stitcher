import cv2
import numpy as np


class Brightness:
    @staticmethod
    def mean_shift_gray(img: np.ndarray, dest_mean: float) -> np.ndarray:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        mean: float = np.mean(gray)
        delta_mean: float = mean - dest_mean
        B, G, R, A = cv2.split(img)

        def mean_shift(x: float):
            x -= delta_mean
            if x < 0:
                x = 0
            if x > 255:
                x = 255
            return x
        mean_shift = np.vectorize(mean_shift)

        B = mean_shift(B)
        G = mean_shift(G)
        R = mean_shift(R)

        img[:, :, 0] = B
        img[:, :, 1] = G
        img[:, :, 2] = R

        return img
