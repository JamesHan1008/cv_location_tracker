import os
import sys

# Path for importing the Mask R-CNN library
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)

from mrcnn.config import Config


class TreesConfig(Config):
    """
    Configuration for training on the data set of trees.
    """
    NAME = "trees"

    # Batch size is (GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape
