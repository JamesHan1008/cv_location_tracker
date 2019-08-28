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
from util.movement import calculate_movement

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
prev_detections = None

while video.isOpened():
    _, frame = video.read()  # X by Y by 3 (RGB)
    curr_detections = model.detect([frame], verbose=1)[0]

    curr_detections = filter_detections(curr_detections)

    if prev_detections is not None:
        matched_detections = match_detections(prev_detections, curr_detections)
        print(matched_detections)

        distance, direction = calculate_movement(matched_detections, camera_id="example", image_width=frame.shape[1])
        print(distance, direction)

    if debugging:
        visualize.display_instances(
            frame,
            curr_detections['rois'],
            curr_detections['masks'],
            curr_detections['class_ids'],
            list(classes.keys()),
            curr_detections['scores'],
        )

    prev_detections = curr_detections
    frame_count += 1
    if frame_count >= 2:
        break

video.release()
cv2.destroyAllWindows()
