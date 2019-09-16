import os
import random

import cv2
import structlog

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
VIDEOS_DIR = os.path.join(FILE_DIR, "videos")
IMAGES_DIR = os.path.join(FILE_DIR, "images")

SAMPLE_RATIO = 0.01

structlog.configure(logger_factory=structlog.PrintLoggerFactory())
logger = structlog.get_logger(processors=[structlog.processors.JSONRenderer()])


def main():
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)

    for file_name in os.listdir(VIDEOS_DIR):
        file_path = os.path.join(VIDEOS_DIR, file_name)
        video = cv2.VideoCapture(file_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {file_name}", num_frames=num_frames)

        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()  # X by Y by 3 (BGR)
            if not ret:
                break

            if random.random() < SAMPLE_RATIO:
                image_path = os.path.join(IMAGES_DIR, f"{file_name}_{frame_count}.png")
                cv2.imwrite(image_path, frame)

            frame_count += 1
            if frame_count % 1000 == 0:
                logger.debug(f"{frame_count} frames processed")

        video.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
