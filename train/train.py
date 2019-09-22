import os
import sys

# Path for importing the Mask R-CNN library
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)

import mrcnn.model as modellib

from train.configs.trees_config import TreesConfig
from train.supervisely_dataset import SuperviselyDataset


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
IMAGES_DIR = os.path.join(FILE_DIR, "images")
LOG_DIR = os.path.join(FILE_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

config = TreesConfig()

# Training dataset
dataset_train = SuperviselyDataset()
dataset_train.add_classes("trees", ["tree"])
dataset_train.add_images("trees", IMAGES_DIR)  # TODO: train-valid split
dataset_train.prepare()

# Validation dataset
dataset_val = SuperviselyDataset()
dataset_train.add_classes("trees", ["tree"])
dataset_train.add_images("trees", IMAGES_DIR)  # TODO: train-valid split
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(
    mode="training",
    config=config,
    model_dir=LOG_DIR,
)

# Load weights trained on MS COCO, but skip layers that are different due to the different number of classes
model.load_weights(
    os.path.join(MODEL_DIR, "mask_rcnn_coco.h5"),
    by_name=True,
    exclude=[
        "mrcnn_class_logits",
        "mrcnn_bbox_fc",
        "mrcnn_bbox",
        "mrcnn_mask",
    ],
)

# Train only the head layers
model.train(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=1,
    layers="heads",
)
