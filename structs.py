from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from pydantic import BaseModel, Extra

if TYPE_CHECKING:
    from collections.abc import Iterator

# required runtime for pydantic
RgbTuple: TypeAlias = tuple[int, int, int]
OptionalPath: TypeAlias = str | None


class LineParam:
    color: RgbTuple
    width: int

    def __init__(self, color: RgbTuple, width: int) -> None:
        self.color = color
        self.width = width

    def __iter__(self) -> Iterator[RgbTuple | int]:
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


class ViewerInitConfig(BaseModel, extra=Extra.ignore):
    model_path: OptionalPath
    mask_path: OptionalPath
    config_path: OptionalPath


@dataclass
class ModelParam:
    confidence: float
    iou: float
    augment: bool
