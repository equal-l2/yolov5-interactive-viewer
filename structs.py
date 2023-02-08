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
    show_outsiders: bool
    outsider_color: RgbTuple
    outsider_width: int
    mask_thres: float
    show_mask_border: bool
    mask_border_color: RgbTuple
    mask_border_width: int
    show_confidence: bool


OptionalPath = typing.Optional[str]


class ViewerInitConfig(BaseModel, extra=Extra.ignore):
    model_path: OptionalPath
    mask_path: OptionalPath
    config_path: OptionalPath


@dataclass
class ModelParam:
    confidence: float
    iou: float
    augment: bool
