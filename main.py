import os
import sys

import cv2

# Import Mask RCNN
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)
sys.path.append(os.path.join(MRCNN_PATH, "samples/coco/"))

import mrcnn.model as modellib
from mrcnn import visualize
import coco

from configs.classes import classes
from util.detections import filter_detections, match_detections
from util.movement import calculate_aggregate_movement

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")

debugging = True


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=TRAIN_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)

# Read from a video
video = cv2.VideoCapture("videos/driving_short.mp4")
frame_count = 0
detections_memory = []

# TODO: take video of walking toward stop sign
# TODO: download map
# TODO: draw path on map

while video.isOpened():
    _, frame = video.read()  # X by Y by 3 (RGB)
    detections = model.detect([frame], verbose=1)[0]
    detections = filter_detections(detections)

    detections_memory.append(detections)

    if len(detections_memory) == 7:
        distance, direction = calculate_aggregate_movement(detections_memory, camera_id="example", image_width=frame.shape[1])
        print(distance)
        print(direction)
        exit(0)

        if debugging:
            visualize.display_instances(
                frame,
                detections["rois"],
                detections["masks"],
                detections["class_ids"],
                list(classes.keys()),
                detections["scores"],
            )

    frame_count += 1
    if frame_count >= 7:
        break

video.release()
cv2.destroyAllWindows()
