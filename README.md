# cv_location_tracker
Track the Location of a Moving Vehicle or Person

pip3 install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

python3 -m main


## Model Training
Here are the steps for training a new model or enhancing an existing model to detect additional classes not included in
the default model used in this project.

1. Gather the training images
    1. Find videos taken from the dashboard of a moving vehicle containing objects of the classes of interest
    2. Store these videos within the directory `train/videos/`
    3. Use `train/generate_images_from_videos.py` to randomly sample frames of videos to generate images to be used for
    training
2. Annotate the training images
    1. Upload the newly generated images in `train/images/` to COCO Annotate


# TO DO
- Train model with additional stationary classes like "tree", "street light", etc.
  - Label the images
  - Write a class that extends utils.Dataset
  - Write a class that extends Config
  - Write a train.py module

- Record a longer video

- Let user select starting position on a map
