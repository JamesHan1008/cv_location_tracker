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

    image_names = []

    def add_classes(self, source: str, classes: list) -> None:
        """ Add the classes to train for """
        for i, class_name in enumerate(classes):
            self.add_class(source, i + 1, class_name)

    def add_images(self, source: str, images_dir: str) -> None:
        """ Add the training images """
        for i, file_name in enumerate(os.listdir(images_dir)):
            self.add_image(
                source=source,
                image_id="placeholder",
                path=os.path.join(images_dir, file_name),
            )
            self.image_names.append(file_name)

    def load_image(self, image_id: str) -> np.ndarray:
        """
        Load an image from a file for the given image ID
        :param image_id:
        :return:
        """
        file_name = self.image_names[image_id]
        image = cv2.imread(f"train/images/{file_name}")
        return image

    def load_mask(self, image_id: str) -> Tuple[list, list]:
        """
        Load the mask from a file for the given image ID
        :param image_id:
        :return:
            "masks": list(H x W x instance_count)
                    [[[False, False], ... , [False, False]],
                     ...
                     [[False, False], ... , [False, False]]]

            "class_ids": list(instance_count)
                    [100, 101]
        """
        file_name = self.image_names[image_id]
        with open(f"train/annotations/converted/{file_name}.json") as f:
            converted_annotations = json.load(f)

        masks = converted_annotations["masks"]
        # TODO: fix this
        masks = np.asarray(masks)
        masks = masks == 1  # convert from 1/0 to True/False
        print(masks)
        masks = masks.tolist()
        print(masks)

        class_ids = converted_annotations["class_ids"]

        return masks, class_ids
