import cv2
import numpy as np


def draw_path(map: np.ndarray, x_old: int, y_old: int, x_new: int, y_new: int) -> np.ndarray:
    """
    Draws a path on a map
    :param map:
    :param x_old:
    :param y_old:
    :param x_new:
    :param y_new:
    :return: a new map with the path drawn on it
    """
    cv2.line(map, (y_old, x_old), (y_new, x_new), (255, 0, 0), 5)
    return map
