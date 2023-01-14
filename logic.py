import typing
from typing import Optional, TypeAlias
import cv2
from structs import LineParam, RgbTuple
import numpy

Cv2Image: TypeAlias = cv2.Mat
Model: TypeAlias = typing.Any
DetectValues: TypeAlias = typing.Any


def run_detect(
    model: Model, cv2_image: Cv2Image, conf_thres: float, iou_thres: float
) -> DetectValues:
    model.conf = conf_thres
    model.iou = iou_thres
    detected = model(cv2_image, size=1280)

    values = detected.pandas().xyxy[0]

    # coords in dataframe are float, so they need to be cast into int
    # (because cv2 accepts int coords only)
    values.round(0)
    values = values.astype({"xmin": int, "ymin": int, "xmax": int, "ymax": int})

    return values


BBoxXyxy: typing.TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def compute_intersect(box1: BBoxXyxy, box2: BBoxXyxy) -> int:
    max_of_x_min = max(box1[0][0], box2[0][0])
    max_of_y_min = max(box1[0][1], box2[0][1])
    min_of_x_max = min(box1[1][0], box2[1][0])
    min_of_y_max = min(box1[1][1], box2[1][1])
    w = min_of_x_max - max_of_x_min
    h = min_of_y_max - max_of_y_min
    intersect = w * h if w > 0 and h > 0 else 0

    return intersect


# cv2_image will be destructively changed
# filename: if not None, the filename will be rendered on the image
# mask: if not None, mask will be considered for bounding box elimination
def draw_result(
    *,
    values: DetectValues,
    cv2_image: Cv2Image,
    filename: Optional[str],
    bb_params: LineParam,
    show_confidence: bool,
    outsider_params: LineParam,
    outsider_thres: float,
    hide_outsiders: bool,
    bounds_params: LineParam,
    upper_pixel: int,
    lower_pixel: int,
    disable_bounds: bool,
    mask: Optional[Cv2Image],
    mask_thres: float,
    text_color: RgbTuple,
):
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
    (bb_color, bb_width) = bb_params
    (outsider_color, outsider_width) = outsider_params

    bounds = ((0, upper_pixel), (cv2_image.shape[1], lower_pixel))
    for row in values.itertuples():
        bb_area = (row.xmax - row.xmin) * (row.ymax - row.ymin)

        if bb_area <= 0:
            # not sure if this can happen
            continue

        is_outsider = False

        # handle bounds
        if not disable_bounds:
            # compute the area of intersection
            bb = ((row.xmin, row.ymin), (row.xmax, row.ymax))
            intersect = compute_intersect(bounds, bb)
            intersect_ratio = intersect / bb_area

            if intersect_ratio < outsider_thres:
                is_outsider = True

            # draw bound rectangle
            (bounds_color, bounds_width) = bounds_params
            cv2.rectangle(
                cv2_image,
                (0, upper_pixel),
                (cv2_image.shape[1], lower_pixel),
                bounds_color,
                bounds_width,
            )

        # handle mask
        if mask is not None:
            mask_cropped = mask[row.ymin : row.ymax, row.xmin : row.xmax]
            whites = numpy.sum(mask_cropped == 255)
            mask_intersect_ratio = whites / bb_area
            if mask_intersect_ratio < mask_thres:
                is_outsider = True

        box_color = outsider_color if is_outsider else bb_color
        box_width = outsider_width if is_outsider else bb_width

        hide_detect = hide_outsiders and is_outsider

        if not hide_detect:
            cv2.rectangle(
                cv2_image,
                (row.xmin, row.ymin),
                (row.xmax, row.ymax),
                box_color,
                box_width,
            )
            if show_confidence:
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
