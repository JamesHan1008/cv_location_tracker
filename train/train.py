import os
import sys

# Path for importing Mask RCNN
MRCNN_PATH = os.getenv("MRCNN_PATH")
sys.path.append(MRCNN_PATH)

import mrcnn.model as modellib

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_DIR = os.path.join(FILE_DIR, "images")
