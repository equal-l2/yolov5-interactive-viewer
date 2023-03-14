# YOLOv5 Interactive Viewer

![screenshot](./assets/screenshot.png)

## Usecase
Run detection on multiple images with a single model (which does single class detection) and view the result

## Prerequisite
- Python 3.10 (Torch requires exactly 3.10)
- Poetry (required for dependency management)

## Run
First, install the dependencies with `poetry install`.  
Then, run any of the scripts below within poetry environment (i.e. `poetry run`, `poetry shell`)  

- `main.py`: the viewer
- `video.py`: run detection on an video input (requires a config file exported from the viewer)

## Model configuration
All values are in `consts.py`.

- `IMG_SIZE`: target image size for the model (all input will be resized to this size by YOLOv5)
    - expect the best result when the same size with the images used for training

## Limitation
- Only a single-class model is supported
    - If the model outputs multiple categories on detection, the behavior is undefined
