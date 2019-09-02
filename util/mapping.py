from typing import Tuple

import cv2
import numpy as np


def draw_path(map: np.ndarray, x_old: int, y_old: int, x_new: int, y_new: int, color: Tuple[int]) -> np.ndarray:
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
    cv2.line(map, (x_old, y_old), (x_new, y_new), color, 5)
    return map
