from typing import Dict

import numpy as np

from classes import classes, class_lookup


def filter_detections(detections: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Filter out detections that do not contribute to movement calculations
    :param detections:
        {
            "rois": array(
                [[x1, y1, w1, h1],
                 [x2, y2, w2, h2]], dtype=int32),
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

    return detections
