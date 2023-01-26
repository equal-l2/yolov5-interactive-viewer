import typing
from typing import Optional, TypeAlias
import cv2
import numpy

from structs import RgbTuple, AppConfig
from consts import MODEL_TYPE

Cv2Image: TypeAlias = cv2.Mat
DetectValues: TypeAlias = typing.Any


def run_detect(
    model: MODEL_TYPE, cv2_image: Cv2Image, config: AppConfig
) -> DetectValues:
    """Run detection on `cv2_image` using the model from `model`."""
    model.conf = config.confidence
    model.iou = config.iou
    IMG_SIZE = 1280 # TODO: make configurable?
    detected = model(cv2_image, size=IMG_SIZE, augment=config.augment)

    values = detected.pandas().xyxy[0]

    # coords in dataframe are float, so they need to be cast into int
    # (because cv2 accepts int coords only)
    values.round(0)
    values = values.astype({"xmin": int, "ymin": int, "xmax": int, "ymax": int})

    return values


BBoxXyxy: typing.TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def compute_intersect(box1: BBoxXyxy, box2: BBoxXyxy) -> int:
    """Compute the area of intersect between `box1` and `box2`"""
    max_of_x_min = max(box1[0][0], box2[0][0])
    max_of_y_min = max(box1[0][1], box2[0][1])
    min_of_x_max = min(box1[1][0], box2[1][0])
    min_of_y_max = min(box1[1][1], box2[1][1])
    w = min_of_x_max - max_of_x_min
    h = min_of_y_max - max_of_y_min
    intersect = w * h if w > 0 and h > 0 else 0

    return intersect


def draw_result(
    values: DetectValues,
    cv2_image: Cv2Image,
    filename: Optional[str],
    mask: Optional[Cv2Image],
    text_color: RgbTuple,
    config: AppConfig,
):
    """Read detection results from `values` and draw them to `cv2_image` (destructive)
    filename: if not None, the filename will be rendered on the image
    mask: if not None, mask will be considered for bounding box elimination
    """
    mask_overlay = True  # TODO: add config
    if mask is not None and mask_overlay:
        # draw the edge of the mask
        mask_edge = cv2.Canny(mask, 10, 245)
        cv2_image[mask_edge > 0] = config.bounds_color

    if filename is not None:
        cv2.putText(
            cv2_image,
            text=filename,
            org=(10, cv2_image.shape[0] - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=text_color,
            thickness=2,
        )

    # draw bounding boxes and bounds
    for row in values.itertuples():
        bb_area = (row.xmax - row.xmin) * (row.ymax - row.ymin)

        if bb_area <= 0:
            # not sure if this can happen
            continue

        # handle mask
        is_outsider = False
        if mask is not None:
            mask_cropped = mask[row.ymin : row.ymax, row.xmin : row.xmax]
            whites = numpy.sum(mask_cropped == 255)
            mask_intersect_ratio = whites / bb_area
            if mask_intersect_ratio < config.mask_thres:
                is_outsider = True

        box_color = config.outsider_color if is_outsider else config.bb_color
        box_width = config.outsider_width if is_outsider else config.bb_width

        hide_detect = config.hide_outsiders and is_outsider

        if not hide_detect:
            cv2.rectangle(
                cv2_image,
                (row.xmin, row.ymin),
                (row.xmax, row.ymax),
                box_color,
                box_width,
            )
            if config.show_confidence:
                cv2.putText(
                    cv2_image,
                    text=f"{row.confidence:.2f}",
                    org=(row.xmin, row.ymin - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=text_color,
                    thickness=2,
                )


def rgb2hex(rgb: RgbTuple):
    (r, g, b) = rgb
    return f"#{r:02X}{g:02X}{b:02X}"
