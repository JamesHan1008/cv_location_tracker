from typing import Dict, List

import numpy as np
import structlog

from configs.classes import classes, class_lookup

structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])

MIN_OVERLAP_RATIO = 0.


def filter_detections(detections: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Filter out detections that do not contribute to movement calculations
    :param detections:
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
    :return: same as 'detections', but only including 'stationary' type objects
    """
    num_total_detections = len(detections["class_ids"])

    indexes_to_keep = []
    for i, class_id in enumerate(detections["class_ids"]):
        class_name = class_lookup[class_id]
        if classes[class_name]["type"] == "stationary":
            indexes_to_keep.append(i)

    # number of objects detected is the first dimension for these keys
    for key in ["rois", "class_ids", "scores"]:
        detections[key] = detections[key][indexes_to_keep]

    # number of objects detected is the last (3rd) dimension for 'masks'
    detections["masks"] = detections["masks"][:, :, indexes_to_keep]

    logger.info(
        "detections filtered",
        num_total_detections=num_total_detections,
        num_remaining_detections=len(detections["class_ids"]),
    )

    return detections


def match_detections(det1: Dict[str, np.ndarray], det2: Dict[str, np.ndarray]) -> List[dict]:
    """
    Match the two sets of detections and find those that are most likely to be the same instances
    :param det1: same as 'detections' in 'filter_detections' above
    :param det2: same as 'detections' in 'filter_detections' above
    :return: list of dictionaries with the following fields:
        class_name: str
        det1_index: int: index of detected item in det1
        det2_index: int: index of detected item in det2
        det1_coordinates: List[int]: same as 'rois' in 'detections'
        det2_coordinates: List[int]: same as 'rois' in 'detections'
        overlap_ratio: float: ratio of overlap area to the smaller area of the two matched detections
    """
    matches = []

    for i in range(len(det1["class_ids"])):
        for j in range(len(det2["class_ids"])):

            # check that the classes are the same
            if det1["class_ids"][i] != det2["class_ids"][j]:
                continue

            # check that the overlaps are large enough
            y_top_1, x_lft_1, y_bot_1, x_rgt_1 = det1["rois"][i]
            y_top_2, x_lft_2, y_bot_2, x_rgt_2 = det2["rois"][j]
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
                    matches.append({
                        "class_name": class_lookup[det1["class_ids"][i]],
                        "det1_index": i,
                        "det2_index": j,
                        "det1_coordinates": det1["rois"][i],
                        "det2_coordinates": det2["rois"][j],
                        "overlap_ratio": overlap_ratio,
                    })

    # deduplicate matches
    filtered_matches = []
    for match in matches:
        to_append = True
        for k in range(len(filtered_matches)):
            if match["det1_index"] == filtered_matches[k]["det1_index"] or \
               match["det2_index"] == filtered_matches[k]["det2_index"]:
                to_append = False
                if match["overlap_ratio"] > filtered_matches[k]["overlap_ratio"]:
                    filtered_matches[k] = match
        if to_append is True:
            filtered_matches.append(match)

    return filtered_matches
