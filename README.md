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
    1. Upload the newly generated images in `train/images/` to `supervise.ly`
        1. If another annotation tool is used, convert all annotations to the expected format and skip to step 3
    2. Annotate the objects of the classes of interest using polygons
    3. Use `train/convert_annotations_supervisely.py` to convert the polygons to bitmaps to follow the same structure
    of training data that the Mask RCNN model expects
3. Train the model
    1. 


# TO DO
- Train model with additional stationary classes like "tree", "street light", etc.
  - Label the images
  - Write a class that extends utils.Dataset
  - Write a class that extends Config
  - Write a train.py module

- Record a longer video

- Let user select starting position on a map
