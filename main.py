import os
import sys
import time

import cv2
import numpy as np
import structlog

# Path for importing Mask RCNN
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)

import mrcnn.model as modellib
from mrcnn import visualize

from configs.classes import classes
from configs.inference_config import InferenceConfig
from util.detections import DetectionSet, filter_detections
from util.movement import calculate_aggregate_movement, calculate_new_location
from util.mapping import draw_path

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")

debugging = False
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
map_height, map_width = map.shape[:2]
x_prev = 370  # For SF: 600, for Hayward: 370
y_prev = 198  # For SF: 815, for Hayward: 198
theta_prev = 0

# Read from a video
video_in = cv2.VideoCapture("videos/intersection_stop_sign.MOV")
fps = video_in.get(cv2.CAP_PROP_FPS)
num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))

# Write map to another video
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_out = cv2.VideoWriter(
    "videos/map.avi",
    fourcc,
    fps,
    (frame_width + map_width, frame_height + map_height),
    isColor=True,
)

logger.info("Processing video", fps=fps, num_frames=num_frames, duration=num_frames/fps)
frame_count = 0
ds_memory = []
t_start = time.time()

while video_in.isOpened():
    _, frame = video_in.read()  # X by Y by 3 (BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # TODO: model detection in batch
    detection_dict = model.detect([frame], verbose=1)[0]

    detection_set = DetectionSet(detection_dict)
    detection_set = filter_detections(detection_set)

    ds_memory.append(detection_set)

    # combine the map with the video, and record it
    combined_frame = np.zeros((frame_height + map_height, frame_width + map_width, 3), dtype="uint8")
    combined_frame[:frame_height, :frame_width] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    combined_frame[-map_height:, -map_width:] = map
    video_out.write(combined_frame)

    # calculate movements every 6 frames, starting from the 7th frame
    if frame_count % 6 == 0 and frame_count != 0:
        logger.info(f"{frame_count + 1} frames processed out of {num_frames}", total_time_elapsed=time.time()-t_start)

        movement = calculate_aggregate_movement(ds_memory, camera_id="example", image_width=frame.shape[1])
        x_curr, y_curr, theta_curr = calculate_new_location("map_hayward.png", x_prev, y_prev, theta_prev, movement)

        color = (255 - frame_count * 3, 0, frame_count * 3)
        map = draw_path(map, x_prev, y_prev, x_curr, y_curr, color)
        cv2.imwrite("maps/new_map.png", map)

        if debugging:
            visualize.display_instances(
                frame,
                detection_set.rois,
                detection_set.masks,
                detection_set.class_ids,
                list(classes.keys()),
                detection_set.scores,
            )

        x_prev, y_prev, theta_prev = x_curr, y_curr, theta_curr
        ds_memory = [detection_set]  # clear memory and only keep the last frame's detections

    frame_count += 1
    if frame_count >= num_frames:
        logger.info("Finished processing video")
        break

    if debugging is True:
        if frame_count >= 7:
            break

video_in.release()
video_out.release()
cv2.destroyAllWindows()
