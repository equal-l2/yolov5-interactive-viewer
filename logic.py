from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy
from yolov5.helpers import load_model

if TYPE_CHECKING:
    from typing import TypeAlias

    from yolov5.models.common import AutoShape

    from structs import DetectConfig, RenderConfig, RgbTuple

    Cv2Image: TypeAlias = cv2.Mat
    Detection: TypeAlias = Any
    BBoxXyxy: TypeAlias = tuple[tuple[int, int], tuple[int, int]]

    # the model type may change, as it is an internal type taken from the current implementation of yolov5
    MODEL_TYPE: TypeAlias = AutoShape


class Model:
    model: MODEL_TYPE

    def __init__(self, model_path: str) -> None:
        self.model = load_model(model_path)

    def run(
        self,
        cv2_image: Cv2Image,
        config: DetectConfig,
    ) -> Detection:
        from consts import IMG_SIZE

        self.model.conf = config.confidence
        self.model.iou = config.iou
        detected = self.model(cv2_image, size=IMG_SIZE, augment=config.augment)

        values = detected.pandas().xyxy[0]

        # coords in dataframe are float, so they need to be cast into int
        # (because cv2 accepts int coords only)
        values.round(0)
        values = values.astype({"xmin": int, "ymin": int, "xmax": int, "ymax": int})

        return values


class Mask:
    img: Final[Cv2Image]
    contours: Final[Any]

    def __init__(self, img_input: Cv2Image) -> None:
        """
        input: must be a grayscale image
        """

        img = img_input.copy()

        # convert grayscale to black/white
        cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=img)

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        self.img = img
        self.contours = contours


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


def render_result(
    values: Detection,
    cv2_image: Cv2Image,
    filename: str | None,
    mask: Mask | None,
    text_color: RgbTuple,
    config: RenderConfig,
) -> None:
    """Read detection results from `values` and draw them to `cv2_image` (destructive)
    filename: if not None, the filename will be rendered on the image
    mask: if not None, mask will be considered for bounding box elimination
    """
    if mask is not None and config.show_mask_border:
        mask_border = config.mask_border_config
        cv2.drawContours(
            cv2_image,
            mask.contours,
            -1,
            mask_border.color,
            thickness=mask_border.width,
        )

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
            mask_cropped = mask.img[row.ymin : row.ymax, row.xmin : row.xmax]
            whites = numpy.sum(mask_cropped == 255)
            mask_intersect_ratio = whites / bb_area
            if mask_intersect_ratio < config.mask_thres:
                is_outsider = True

        box_params = config.outsider_config if is_outsider else config.bb_config

        show_detect = config.show_outsiders or not is_outsider

        if show_detect:
            cv2.rectangle(
                cv2_image,
                (row.xmin, row.ymin),
                (row.xmax, row.ymax),
                box_params.color,
                box_params.width,
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


def rgb2hex(rgb: RgbTuple) -> str:
    (r, g, b) = rgb
    return f"#{r:02X}{g:02X}{b:02X}"
