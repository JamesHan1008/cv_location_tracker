from typing import List, Tuple

import numpy as np
import structlog

from configs.camera_settings import camera_settings
from configs.classes import classes
from configs.map_settings import map_settings
from util.detections import match_detections

structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])


def calculate_movement(matched_detections: List[dict], camera_id: str, image_width: int) -> Tuple[float, float]:
    """
    Calculates the movement of the observer (camera) based on changes to the locations and sizes of detected objects
    TODO: the calculations in this function is most accurate when the object is directly in front of the observer, a more accurate calculation method is needed
    :param matched_detections: TODO: make this into a class
    :param camera_id: ID of the camera used in capturing the video
    :param image_width: width of the image in pixels
    :return:
        distance (float): distance traveled in meters (positive means moving forward)
        direction (float): direction of movement in radians (positive means moving to the left)
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
        d = d / camera_settings[camera_id]["meters_per_pixel"]

        # angle of movement
        field_of_view = camera_settings[camera_id]["field_of_view"]
        x_half_of_image = image_width / 2
        x1_left_of_center = x_half_of_image - x_cnt_1
        x2_left_of_center = x_half_of_image - x_cnt_2

        theta1 = np.arctan((x1_left_of_center / x_half_of_image) * np.tan(field_of_view / 2))
        theta2 = np.arctan((x2_left_of_center / x_half_of_image) * np.tan(field_of_view / 2))
        theta = theta1 - theta2  # direction of observer is opposite of the direction of the object

        # keep track of the estimates based on each object
        distances.append(d)
        directions.append(theta)

    if len(distances) == 0:
        # TODO: better handling of these situations
        return 0, 0
    else:
        return np.mean(distances), np.mean(directions)


def movement_conversion_p2c(distance: float, direction: float) -> Tuple[float, float]:
    """
    Converts polar movement representation to cartesian movement representation
    :param distance: distance traveled in meters (positive means moving forward)
    :param direction: direction of movement in radians (positive means moving to the left)
    :return:
        dist_forward (float): direction of movement forward
        dist_sideways (float): direction of movement to the left
    """
    dist_forward = distance * np.cos(direction)
    dist_sideways = distance * np.sin(direction)
    return dist_forward, dist_sideways


def movement_conversion_c2p(dist_forward: float, dist_sideways: float) -> Tuple[float, float]:
    """
    Converts cartesian movement representation to polar movement representation
    :param dist_forward: direction of movement forward
    :param dist_sideways: direction of movement to the left
    :return:
        distance (float): distance traveled in meters (positive means moving forward)
        direction (float): direction of movement in radians (positive means moving to the left)
    """
    distance = np.sqrt(dist_forward ** 2 + dist_sideways ** 2)

    if dist_forward == 0:
        direction = 0
    else:
        direction = np.arctan(dist_sideways / dist_forward)

    if dist_forward < 0:
        direction += np.pi

    return distance, direction


def calculate_aggregate_movement(detections_memory: List[dict], camera_id: str, image_width: int) -> Tuple[float, float]:
    """
    Calculates the aggregate movement across 6 frames
    :param detections_memory: list of detection objects (default Mask RCNN detection schema)
    :param camera_id: ID of the camera used in capturing the video
    :param image_width: width of the image in pixels
    :return:
        distance (float): distance traveled in meters (positive means moving forward)
        direction (float): direction of movement in radians (positive means moving to the left)
    """
    step_sizes = [1, 2, 3, 6]

    agg_dist_forward_estimates = []
    agg_dist_sideways_estimates = []

    # make different estimates of aggregate movement using different step sizes
    for step_size in step_sizes:
        agg_dist_forward = 0
        agg_dist_sideways = 0
        agg_direction = 0

        # for each step size, take however many steps is required to aggregate over 6 frames of movement
        for i in range(0, 6, step_size):
            matched_detections = match_detections(detections_memory[i], detections_memory[i + step_size])
            distance, direction = calculate_movement(matched_detections, camera_id, image_width)
            agg_direction += direction

            dist_forward, dist_sideways = movement_conversion_p2c(distance, agg_direction)
            agg_dist_forward += dist_forward
            agg_dist_sideways += dist_sideways

        agg_dist_forward_estimates.append(agg_dist_forward)
        agg_dist_sideways_estimates.append(agg_dist_sideways)

    # make sure different estimates are not too far off
    if max(agg_dist_forward_estimates) - min(agg_dist_forward_estimates) > 1 or \
       max(agg_dist_sideways_estimates) - min(agg_dist_sideways_estimates) > 1:
        logger.warning(
            "Movement estimates are far apart",
            dist_forward_estimates=agg_dist_forward_estimates,
            dist_sideways_estimates=agg_dist_sideways_estimates,
        )

    mean_agg_dist_forward = np.mean(agg_dist_forward_estimates)
    mean_agg_dist_sideways = np.mean(agg_dist_sideways_estimates)

    distance, direction = movement_conversion_c2p(mean_agg_dist_forward, mean_agg_dist_sideways)

    logger.info(
        "Movement calculated",
        dist_forward=mean_agg_dist_forward,
        dist_sideways=mean_agg_dist_sideways,
        distance=distance,
        direction=direction,
    )

    return distance, direction


def calculate_new_location(map_name: str, x_old: int, y_old: int, theta_old: float, distance: float, direction: float) -> Tuple[int, int, float]:
    """
    Calculate the new location of the observer based on the previous location and the movement
    :param map_name: name of the map used to retrieve map settings
    :param x_old: distance (pixels) measured from the left edge
    :param y_old: distance (pixels) measured from the top edge
    :param theta_old: angle (radians) measured from the positive x-axis
    :param distance: distance traveled (meters) where positive means moving forward
    :param direction: direction of movement (radians) where positive means moving to the left
    :return:
        x_new: (pixels)
        y_new: (pixels)
        theta_new: (radians)
    """
    theta_new = theta_old + direction
    if theta_new < 0:
        theta_new += np.pi * 2
    if theta_new > np.pi * 2:
        theta_new -= np.pi * 2

    pixels_per_meter = 1 / map_settings[map_name]["meters_per_pixel"]
    x_moved = distance * np.cos(theta_new) * pixels_per_meter
    y_moved = distance * np.sin(theta_new) * pixels_per_meter

    x_new = int(x_old + x_moved)
    y_new = int(y_old - y_moved)

    logger.info(
        "New location calculated",
        old_location=(x_old, y_old),
        new_location=(x_new, y_new),
        old_direction=theta_old,
        new_direction=theta_new,
    )

    return x_new, y_new, theta_new
