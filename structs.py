import typing

from pydantic import BaseModel

RgbTuple: typing.TypeAlias = tuple[int, int, int]


class LineParam:
    color: RgbTuple
    width: int

    def __init__(self, color: RgbTuple, width: int):
        self.color = color
        self.width = width

    def __iter__(self):
        return iter((self.color, self.width))


class AppConfig(BaseModel):
    confidence: float
    iou: float
    bb_color: RgbTuple
    bb_width: int
    show_confidence: bool
    outsider_color: RgbTuple
    outsider_width: int
    outsider_thres: float
    hide_outsiders: bool
    bounds_color: RgbTuple
    bounds_width: int
    upper_pixel: int
    lower_pixel: int
    disable_bounds: bool
    mask_thres: float
