from typing import Dict, List

import numpy as np
import structlog

from configs.classes import classes, class_lookup

structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])

MIN_OVERLAP_RATIO = 0.


class DetectionSet:
    rois: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray
    masks: np.ndarray

    def __init__(self, detection_dict: Dict[str, np.ndarray]):
        """
        :param detection_dict:
            {
                "rois": array(
                    [[y1_1, x1_1, y2_1, x2_1],
                     [y1_2, x1_2, y2_2, x2_2]], dtype=int32),
                "class_ids": array([c1, c2], dtype=int32),
                "scores": array([0.9, 0.8], dtype=float32),
                "masks": array(
                    [[[False, False, True, ... , False, False],
                      [ ... ]],
                     [[ ... ],
                      [ ... ]]])
            }
        """
        self.rois = detection_dict["rois"]
        self.class_ids = detection_dict["class_ids"]
        self.scores = detection_dict["scores"]
        self.masks = detection_dict["masks"]

    def __len__(self):
        return len(self.class_ids)


class Match:
    class_name: str
    ds1_index: int
    ds2_index: int
    ds1_coordinates: List[int]
    ds2_coordinates: List[int]
    overlap_ratio: float

    def __init__(self, class_name: str, ds1_index: int, ds2_index: int, ds1_coordinates: List[int], ds2_coordinates: List[int], overlap_ratio: float):
        """
        :param class_name:
        :param ds1_index: index of detected item in det1
        :param ds2_index: index of detected item in det2
        :param ds1_coordinates: similar to DetectionSet.rois
        :param ds2_coordinates: similar to DetectionSet.rois
        :param overlap_ratio: ratio of overlap area to the smaller area of the two matched detections
        """
        self.class_name = class_name
        self.ds1_index = ds1_index
        self.ds2_index = ds2_index
        self.ds1_coordinates = ds1_coordinates
        self.ds2_coordinates = ds2_coordinates
        self.overlap_ratio = overlap_ratio


def filter_detections(detections: DetectionSet) -> DetectionSet:
    """
    Filter out detections that do not contribute to movement calculations
    :return: a DetectionSet that only includes 'stationary' type objects
    """
    num_total_detections = len(detections)

    indexes_to_keep = []
    for i, class_id in enumerate(detections.class_ids):
        class_name = class_lookup[class_id]
        if classes[class_name]["type"] == "stationary":
            indexes_to_keep.append(i)

    # number of objects detected is the first dimension for these keys
    detections.rois = detections.rois[indexes_to_keep]
    detections.class_ids = detections.class_ids[indexes_to_keep]
    detections.scores = detections.scores[indexes_to_keep]

    # number of objects detected is the last (3rd) dimension for 'masks'
    detections.masks = detections.masks[:, :, indexes_to_keep]

    logger.info(
        "detections filtered",
        num_total_detections=num_total_detections,
        num_remaining_detections=len(detections),
    )

    return detections


def match_detections(ds1: DetectionSet, ds2: DetectionSet) -> List[Match]:
    """
    Match the two sets of detections and find those that are most likely to be the same instances.
    If two detections of an object of the same class shows up on both DetectionSets and have highly
    overlapping coordinates, then they are likely to be the same instance.
    :param ds1:
    :param ds2:
    :return: list of Matches
    """
    matches = []

    for i in range(len(ds1)):
        for j in range(len(ds2)):

            # check that the classes are the same
            if ds1.class_ids[i] != ds2.class_ids[j]:
                continue

            # check that the overlaps are large enough
            y_top_1, x_lft_1, y_bot_1, x_rgt_1 = ds1.rois[i]
            y_top_2, x_lft_2, y_bot_2, x_rgt_2 = ds2.rois[j]
            h1 = y_bot_1 - y_top_1
            h2 = y_bot_2 - y_top_2
            w1 = x_rgt_1 - x_lft_1
            w2 = x_rgt_2 - x_lft_2

            x_dist = min(x_rgt_1, x_rgt_2) - max(x_lft_1, x_lft_2)
            y_dist = min(y_bot_1, y_bot_2) - max(y_top_1, y_top_2)

            if x_dist > 0 and y_dist > 0:
                area_overlap = x_dist * y_dist
                area_largest_box = max(w1 * h1, w2 * h2)
                overlap_ratio = area_overlap / area_largest_box

                if overlap_ratio >= MIN_OVERLAP_RATIO:
                    match = Match(
                        class_name=class_lookup[ds1.class_ids[i]],
                        ds1_index=i,
                        ds2_index=j,
                        ds1_coordinates=ds1.rois[i],
                        ds2_coordinates=ds2.rois[j],
                        overlap_ratio=overlap_ratio,
                    )
                    matches.append(match)

    # deduplicate matches
    filtered_matches = []
    for match in matches:
        to_append = True
        for k in range(len(filtered_matches)):
            if match.ds1_index == filtered_matches[k].ds1_index or \
               match.ds2_index == filtered_matches[k].ds2_index:
                to_append = False
                if match.overlap_ratio > filtered_matches[k].overlap_ratio:
                    filtered_matches[k] = match
        if to_append is True:
            filtered_matches.append(match)

    return filtered_matches
