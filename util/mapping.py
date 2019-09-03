from typing import Tuple

import cv2
import numpy as np


def draw_path(map: np.ndarray, x_old: float, y_old: float, x_new: float, y_new: float, color: Tuple[int]) -> np.ndarray:
    """
    Draws a path on a map
    :param map:
    :param x_old: horizontal distance (pixels) from the left edge
    :param y_old: vertical distance (pixels) from the top edge
    :param x_new:
    :param y_new:
    :param color: BGR
    :return: a new map with the path drawn on it
    """
    x_old = int(round(x_old, 0))
    y_old = int(round(y_old, 0))
    x_new = int(round(x_new, 0))
    y_new = int(round(y_new, 0))

    cv2.line(map, (x_old, y_old), (x_new, y_new), color, 3)
    return map
