from structs import RgbTuple
from yolov5.models.common import AutoShape

# YOLOv5 parameters, from the default value in detect.py
CONFIDENCE_DEFAULT: float = 0.25
IOU_DEFAULT: float = 0.45

# our parameters
OUTSIDE_THRES_DEFAULT: float = 0.6
MASK_THRES_DEFAULT: float = OUTSIDE_THRES_DEFAULT
UPPER_BOUND_DEFAULT: int = 287
LOWER_BOUND_DEFAULT: int = 850

BOUNDS_COLOR_DEFAULT: str = "#00FF00"  # green
BBOXES_COLOR_DEFAULT: str = "#FF0000"  # red
OUTSIDER_COLOR_DEFAULT: str = "#9900FF"  # purple

# TODO: make configurable
TEXT_COLOR: RgbTuple = (255, 0, 0)  # red

# the type of model may change, as it is taken from internal implementation
MODEL_TYPE = AutoShape
