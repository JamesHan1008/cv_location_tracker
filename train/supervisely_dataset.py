import json
import os
import sys
from typing import Tuple

import cv2
import numpy as np

# Path for importing the Mask R-CNN library
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)

from mrcnn.utils import Dataset


class SuperviselyDataset(Dataset):
    """ A dataset that is annotated using Supervisely """

    def add_classes(self, source: str, classes: list) -> None:
        """ Add the classes to train for """
        for i, class_name in enumerate(classes):
            self.add_class(source, i + 1, class_name)

    def add_images(self, source: str, images_dir: str) -> None:
        """ Add the training images """
        for file_name in os.listdir(images_dir):
            self.add_image(
                source=source,
                image_id="abc",
                path=os.path.join(images_dir, file_name),
            )

    def load_image(self, image_id: str) -> np.ndarray:
        """
        Load an image from a file for the given image ID
        :param image_id:
        :return:
        """
        image = cv2.imread(f"train/images/{image_id}")
        return image

    def load_mask(self, image_id: str) -> Tuple[list, list]:
        """
        Load the mask from a file for the given image ID
        :param image_id:
        :return:
            "masks": list(H x W x num_detections)
                    [[[False, False], ... , [False, False]],
                     ...
                     [[False, False], ... , [False, False]]]

            "class_ids": np.ndarray(num_detections)
                    array([c1, c2], dtype=int32)
        """
        with open(f"train/annotations/converted/{image_id}.json") as f:
            converted_annotations = json.load(f)

        masks = converted_annotations["masks"]
        masks = np.asarray(masks)
        masks = masks == 1  # convert from 1/0 to True/False
        masks = masks.tolist()

        class_ids = converted_annotations["class_ids"]

        return masks, class_ids
