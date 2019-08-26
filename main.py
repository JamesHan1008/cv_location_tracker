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

from class_names import class_names


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")


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
while video.isOpened():
    _, frame = video.read()  # X by Y by 3 (RGB)
    results = model.detect([frame], verbose=1)[0]

    print(results)
    visualize.display_instances(
        frame,
        results['rois'],
        results['masks'],
        results['class_ids'],
        class_names,
        results['scores'],
    )
    break

video.release()
cv2.destroyAllWindows()
