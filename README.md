# Location Tracker Using Computer Vision
Track the location of a moving vehicle or person.


## Set Up
Before trying out this project, the following dependencies need to be set up.

1. This project requires the Mask R-CNN codebase
    1. Clone the Mask R-CNN repository from <https://github.com/matterport/Mask_RCNN>
    2. Set the environment variable `MRCNN_PATH` to the path of this repository
2. This project uses Pipenv
    1. Install Pipenv: `$ brew install pipenv`
    2. Create a new virtual environment: `$ pipenv shell`
3. This project uses Git large file storage
    1. Install Git LFS: `$ brew install git-lfs`
    2. Set up Git LFS and its respective hooks: `$ git lfs install`
    3. Pull down the large files stored using Git LFS: `$ git lfs pull`


## Run the Location Tracker
* `$ python3 -m main`


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
  - Write a train.py module

- Record a longer video

- Let user select starting position on a map
