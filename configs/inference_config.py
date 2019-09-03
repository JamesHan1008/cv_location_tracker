import os
import sys

# Path for importing COCO
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(os.path.join(MRCNN_PATH, "samples/coco/"))

import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
