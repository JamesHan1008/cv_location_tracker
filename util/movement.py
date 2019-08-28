import numpy as np
from typing import List

from configs.camera_settings import camera_settings
from configs.classes import classes


def calculate_movement(matched_detections: List[dict], camera_id: str, image_width: int):
    """
    Calculates the movement of the observer (camera) based on changes to the locations and sizes of detected objects
    TODO: the calculations in this function is most accurate when the object is directly in front of the observer, a more accurate calculation method is needed
    :param matched_detections: TODO: make this into a class
    :param camera_id: ID of the camera used in capturing the video
    :param image_width: width of the image in pixels
    :return:
        distance (float): distance traveled in meters (positive means moving forward)
        direction (float): direction of movement in radians (positive means moving to the right)
    """
    if camera_id not in camera_settings:
        raise ValueError(f"camera ID is invalid: {camera_id}")
    focal_length = camera_settings[camera_id]["focal_length"]

    # estimate the movement of the observer based on positional changes in each matched detection
    distances = []
    directions = []
    for match in matched_detections:

        # estimated real dimensions of the detected object
        wr = classes[match["class_name"]]["mean_width"]
        hr = classes[match["class_name"]]["mean_height"]

        # coordinates of detected objects
        y_top_1, x_lft_1, y_bot_1, x_rgt_1 = match["det1_coordinates"]
        y_top_2, x_lft_2, y_bot_2, x_rgt_2 = match["det2_coordinates"]
        x_cnt_1 = (x_lft_1 + x_rgt_1) / 2
        x_cnt_2 = (x_lft_2 + x_rgt_2) / 2
        h1 = y_bot_1 - y_top_1
        h2 = y_bot_2 - y_top_2
        w1 = x_rgt_1 - x_lft_1
        w2 = x_rgt_2 - x_lft_2

        # distance from observer (estimate using both width and height)
        d1w = wr * focal_length / w1
        d2w = wr * focal_length / w2
        d1h = hr * focal_length / h1
        d2h = hr * focal_length / h2

        # distance traveled
        dw = d1w - d2w
        dh = d1h - d2h
        d = (dw + dh) / 2

        # angle of movement
        field_of_view = camera_settings[camera_id]["field_of_view"]
        x_half_of_image = image_width / 2
        x1_right_of_center = x_cnt_1 - x_half_of_image
        x2_right_of_center = x_cnt_2 - x_half_of_image

        theta1 = np.arctan((x1_right_of_center / x_half_of_image) * np.tan(field_of_view / 2))
        theta2 = np.arctan((x2_right_of_center / x_half_of_image) * np.tan(field_of_view / 2))
        theta = theta1 - theta2  # direction of observer is opposite of the direction of the object

        # keep track of the estimates based on each object
        distances.append(d)
        directions.append(theta)

    if len(distances) == 0:
        return None, None
    else:
        return np.mean(distances), np.mean(directions)
