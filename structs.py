from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, Extra

RgbTuple: TypeAlias = tuple[int, int, int]
OptionalPath: TypeAlias = str | None


class LineConfig(BaseModel):
    color: RgbTuple
    width: int


class DetectConfig(BaseModel):
    confidence: float
    iou: float
    augment: bool


class RenderConfig(BaseModel):
    bb_config: LineConfig
    mask_border_config: LineConfig
    outsider_config: LineConfig
    mask_thres: float
    show_outsiders: bool
    show_mask_border: bool
    show_confidence: bool


class AppConfig(BaseModel, extra=Extra.ignore):
    detect: DetectConfig
    render: RenderConfig


class ViewerInitConfig(BaseModel, extra=Extra.ignore):
    model_path: OptionalPath
    mask_path: OptionalPath
    config_path: OptionalPath
