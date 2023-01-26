import typing
from dataclasses import dataclass

from pydantic import BaseModel, Extra

RgbTuple: typing.TypeAlias = tuple[int, int, int]


class LineParam:
    color: RgbTuple
    width: int

    def __init__(self, color: RgbTuple, width: int):
        self.color = color
        self.width = width

    def __iter__(self):
        return iter((self.color, self.width))


class AppConfig(BaseModel, extra=Extra.ignore):
    confidence: float
    iou: float
    augment: bool
    bb_color: RgbTuple
    bb_width: int
    outsider_color: RgbTuple
    outsider_width: int
    bounds_color: RgbTuple
    bounds_width: int
    mask_thres: float
    show_confidence: bool
    hide_outsiders: bool
    # Set the default, for compatibility
    augment = False


@dataclass
class ModelParam:
    confidence: float
    iou: float
    augment: bool
