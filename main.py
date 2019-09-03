import os
import sys
import cv2
import structlog

# Path for importing Mask RCNN
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)

import mrcnn.model as modellib
from mrcnn import visualize

from configs.classes import classes
from configs.inference_config import InferenceConfig
from util.detections import filter_detections
from util.movement import calculate_aggregate_movement, calculate_new_location
from util.mapping import draw_path

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")

debugging = True
structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])

# Model configurations
config = InferenceConfig()
if debugging is True:
    config.display()

# Create model object in inference mode with weights trained on the MS-COCO data set
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=TRAIN_DIR)
model.load_weights(MODEL_PATH, by_name=True)

# Load map
map = cv2.imread("maps/map_hayward.png")
x_prev = 370  # For SF: 600, for Hayward: 370
y_prev = 198  # For SF: 815, for Hayward: 198
theta_prev = 0

# Read from a video
video = cv2.VideoCapture("videos/intersection_stop_sign.MOV")
frame_count = 0
detections_memory = []

while video.isOpened():
    _, frame = video.read()  # X by Y by 3 (BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = model.detect([frame], verbose=1)[0]
    detections = filter_detections(detections)

    detections_memory.append(detections)

    # calculate movements every 6 frames, starting from the 7th frame
    if frame_count % 6 == 0 and frame_count != 0:
        distance, direction = calculate_aggregate_movement(detections_memory, camera_id="example", image_width=frame.shape[1])
        x_curr, y_curr, theta_curr = calculate_new_location("map_hayward.png", x_prev, y_prev, theta_prev, distance, direction)

        color = (255 - frame_count * 5, 0, frame_count * 5)
        map = draw_path(map, x_prev, y_prev, x_curr, y_curr, color)
        cv2.imwrite("maps/new_map.png", map)

        if debugging:
            visualize.display_instances(
                frame,
                detections["rois"],
                detections["masks"],
                detections["class_ids"],
                list(classes.keys()),
                detections["scores"],
            )

        x_prev, y_prev, theta_prev = x_curr, y_curr, theta_curr
        detections_memory = [detections]  # clear memory and only keep the last frame's detections

    frame_count += 1
    if frame_count >= 25:
        break

video.release()
cv2.destroyAllWindows()
