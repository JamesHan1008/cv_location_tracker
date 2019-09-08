from typing import List, Tuple

import numpy as np
import structlog

from configs.camera_settings import camera_settings
from configs.classes import classes
from configs.map_settings import map_settings
from util.detections import DetectionSet, Match, match_detections

structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])


class Movement:
    distance: float
    direction: float
    dist_forward: float
    dist_sideways: float

    def __init__(self, distance: float = None, direction: float = None, dist_forward: float = None, dist_sideways: float = None):
        """
        :param distance: total distance traveled in meters
        :param direction: direction of movement in radians (positive means moving to the left)
        :param dist_forward: forward distance traveled in meters (negative means moving backwards)
        :param dist_sideways: leftward distance traveled in meters (negative means moving rightward)
        """
        if distance is not None and direction is not None and dist_forward is None and dist_sideways is None:
            self.distance = distance
            self.direction = direction
            self.movement_conversion_p2c()
        elif dist_forward is not None and dist_sideways is not None and distance is None and direction is None:
            self.dist_forward = dist_forward
            self.dist_sideways = dist_sideways
            self.movement_conversion_c2p()
        else:
            raise ValueError("Must provide either (distance, direction) or (dist_forward, dist_sideways), but not both")

    def movement_conversion_p2c(self) -> None:
        """
        Converts polar movement representation to cartesian movement representation
        """
        self.dist_forward = self.distance * np.cos(self.direction)
        self.dist_sideways = self.distance * np.sin(self.direction)

    def movement_conversion_c2p(self) -> None:
        """
        Converts cartesian movement representation to polar movement representation
        """
        self.distance = np.sqrt(self.dist_forward ** 2 + self.dist_sideways ** 2)

        if self.dist_forward == 0:
            self.direction = 0
        else:
            self.direction = np.arctan(self.dist_sideways / self.dist_forward)

        if self.dist_forward < 0:
            self.direction += np.pi

    def update_distance(self, new_distance):
        self.distance = new_distance
        self.movement_conversion_p2c()

    def update_direction(self, new_direction):
        self.direction = new_direction
        self.movement_conversion_p2c()

    def update_dist_forward(self, new_dist_forward):
        self.dist_forward = new_dist_forward
        self.movement_conversion_c2p()

    def update_dist_sideways(self, new_dist_sideways):
        self.dist_sideways = new_dist_sideways
        self.movement_conversion_c2p()


def calculate_movement(matched_detections: List[Match], camera_id: str, image_width: int) -> Movement:
    """
    Calculates the movement of the observer (camera) based on changes to the locations and sizes of detected objects
    TODO: the calculations in this function is most accurate when the object is directly in front of the observer, a more accurate calculation method is needed
    :param matched_detections:
    :param camera_id: ID of the camera used in capturing the video
    :param image_width: width of the image in pixels
    :return: Movement of the observer
    """
    if camera_id not in camera_settings:
        raise ValueError(f"camera ID is invalid: {camera_id}")
    focal_length = camera_settings[camera_id]["focal_length"]

    # estimate the movement of the observer based on positional changes in each matched detection
    distances = []
    directions = []
    for match in matched_detections:

        # estimated real dimensions of the detected object
        wr = classes[match.class_name]["mean_width"]
        hr = classes[match.class_name]["mean_height"]

        # coordinates of detected objects
        y_top_1, x_lft_1, y_bot_1, x_rgt_1 = match.ds1_coordinates
        y_top_2, x_lft_2, y_bot_2, x_rgt_2 = match.ds2_coordinates
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
        return Movement(distance=0, direction=0)
    else:
        return Movement(distance=float(np.mean(distances)), direction=float(np.mean(directions)))


def calculate_aggregate_movement(detections_memory: List[DetectionSet], camera_id: str, image_width: int) -> Movement:
    """
    Calculates the aggregate movement across 6 frames
    :param detections_memory: DetectionSets of the last 7 frames
    :param camera_id: ID of the camera used in capturing the video
    :param image_width: width of the image in pixels
    :return: aggregate Movement across all frames
    """
    step_sizes = [1, 2, 3, 6]

    agg_dist_forward_estimates = []
    agg_dist_sideways_estimates = []

    # make different estimates of aggregate movement using different step sizes
    for step_size in step_sizes:
        agg_movement = Movement(distance=0, direction=0)

        # for each step size, take however many steps is required to aggregate over 6 frames of movement
        for i in range(0, 6, step_size):
            matched_detections = match_detections(detections_memory[i], detections_memory[i + step_size])
            movement = calculate_movement(matched_detections, camera_id, image_width)

            agg_movement.update_dist_forward(agg_movement.dist_forward + movement.dist_forward)
            agg_movement.update_dist_sideways(agg_movement.dist_sideways + movement.dist_sideways)

        agg_dist_forward_estimates.append(agg_movement.dist_forward)
        agg_dist_sideways_estimates.append(agg_movement.dist_sideways)

    # make sure different estimates are not too far off
    if max(agg_dist_forward_estimates) - min(agg_dist_forward_estimates) > 1 or \
       max(agg_dist_sideways_estimates) - min(agg_dist_sideways_estimates) > 1:
        logger.warning(
            "Movement estimates are far apart",
            dist_forward_estimates=agg_dist_forward_estimates,
            dist_sideways_estimates=agg_dist_sideways_estimates,
        )

    best_estimate_movement = Movement(
        dist_forward=float(np.mean(agg_dist_forward_estimates)),
        dist_sideways=float(np.mean(agg_dist_sideways_estimates)),
    )

    # TODO: to_string method
    logger.info(
        "Movement calculated",
        dist_forward=best_estimate_movement.dist_forward,
        dist_sideways=best_estimate_movement.dist_sideways,
        distance=best_estimate_movement.distance,
        direction=best_estimate_movement.direction,
    )

    return best_estimate_movement


def calculate_new_location(map_name: str, x_old: int, y_old: int, theta_old: float, movement: Movement) -> Tuple[float, float, float]:
    """
    Calculate the new location of the observer based on the previous location and the movement
    :param map_name: name of the map used to retrieve map settings
    :param x_old: distance (pixels) measured from the left edge
    :param y_old: distance (pixels) measured from the top edge
    :param theta_old: angle (radians) measured from the positive x-axis
    :param movement:
    :return:
        x_new: (pixels)
        y_new: (pixels)
        theta_new: (radians)
    """
    theta_new = theta_old + movement.direction
    if theta_new < 0:
        theta_new += np.pi * 2
    if theta_new > np.pi * 2:
        theta_new -= np.pi * 2

    pixels_per_meter = 1 / map_settings[map_name]["meters_per_pixel"]
    x_moved = movement.distance * np.cos(theta_new) * pixels_per_meter
    y_moved = movement.distance * np.sin(theta_new) * pixels_per_meter

    x_new = round(x_old + x_moved, 2)
    y_new = round(y_old - y_moved, 2)

    logger.info(
        "New location calculated",
        old_location=(x_old, y_old),
        new_location=(x_new, y_new),
        old_direction=theta_old,
        new_direction=theta_new,
    )

    return x_new, y_new, theta_new
