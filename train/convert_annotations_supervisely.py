import json
import os
from typing import Dict

import cv2
import numpy as np
import structlog
from tqdm import tqdm

from configs.classes import classes

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
RAW_DIR = os.path.join(FILE_DIR, "annotations", "raw")
CONVERTED_DIR = os.path.join(FILE_DIR, "annotations", "converted")

structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])


def convert_polygons_to_masks(raw_annotations: dict, file_name: str) -> Dict[str, list]:
    """
    Convert Supervisely generated raw annotation data into the input format expected by the Mask R-CNN model (not
    exactly the same, because the data needs to be serialized into JSON for storage)
    :param raw_annotations: data loaded from a raw annotation file generated by Supervisely
    :param file_name:
    :return: converted_annotations:
        {
            "class_ids": (instance_count)
                [100, 101]

            "masks": (H x W x instance_count)
                [[[0, 0], ... , [0, 0]],
                 ...
                 [[0, 1], ... , [1, 0]]]
        }
    """
    img_height = raw_annotations["size"]["height"]
    img_width = raw_annotations["size"]["width"]
    img_blank = np.zeros((img_height, img_width))

    class_ids = []
    masks = []

    for obj in tqdm(raw_annotations["objects"]):
        if len(obj["points"]["interior"]) > 0:
            logger.warning("Annotated object should not contain interior points", file_name=file_name)

        points = np.array(obj["points"]["exterior"])
        mask = cv2.fillPoly(img=img_blank, pts=[points], color=1)
        mask = mask.astype(int).tolist()
        masks.append(mask)

        # convert masks from (instance_count x H x W) to (H x W x instance_count)
        masks = np.asarray(masks)
        masks = np.moveaxis(masks, 0, -1)
        masks = masks.tolist()

        class_id = classes[obj["classTitle"]]["id"]
        class_ids.append(class_id)

    return {
        "class_ids": class_ids,
        "masks": masks,
    }


def main():
    if not os.path.exists(RAW_DIR):
        raise NotADirectoryError(f"[{RAW_DIR}] must exist and contain the raw annotation JSON files")
    if not os.path.exists(CONVERTED_DIR):
        os.mkdir(CONVERTED_DIR)

    for file_name in os.listdir(RAW_DIR):
        if file_name.startswith("."):
            continue

        with open(os.path.join(RAW_DIR, file_name)) as f:
            raw_annotations = json.load(f)

        converted_annotations = convert_polygons_to_masks(raw_annotations, file_name)

        with open(os.path.join(CONVERTED_DIR, file_name), "w") as f:
            json.dump(converted_annotations, f)


if __name__ == "__main__":
    main()
