from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structs import RgbTuple

# model parameters
IMG_SIZE = 1280  # better to match with the dataset used for training the model

# YOLOv5 parameters, from the default value in yolov5/detect.py
CONFIDENCE_DEFAULT: float = 0.25
IOU_DEFAULT: float = 0.45

# rendering parameters
MASK_THRES_DEFAULT: float = 0.6
UPPER_BOUND_DEFAULT: int = 287
LOWER_BOUND_DEFAULT: int = 850

BOUNDS_COLOR_DEFAULT: str = "#00FF00"  # green
BBOXES_COLOR_DEFAULT: str = "#FF0000"  # red
OUTSIDER_COLOR_DEFAULT: str = "#9900FF"  # purple

BOUNDS_WIDTH_DEFAULT: int = 1
BBOXES_WIDTH_DEFAULT: int = 2
OUTSIDER_WIDTH_DEFAULT: int = 2

LINE_WIDTH_MIN = 1
LINE_WIDTH_MAX = 10

# TODO: make configurable
TEXT_COLOR: RgbTuple = (255, 0, 0)  # red
