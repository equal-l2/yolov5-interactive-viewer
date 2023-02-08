#!/usr/bin/env python3
import argparse
from structs import AppConfig, RgbTuple
from consts import MODEL_TYPE

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--mask", required=False)


def to_bgr(rgb: RgbTuple) -> tuple[int, int, int]:
    (r, g, b) = rgb
    return (b, g, r)


# Convert all colors in config from RGB to BGR, for cv2
def config_to_bgr(config: AppConfig):
    config.bb_color = to_bgr(config.bb_color)
    config.outsider_color = to_bgr(config.outsider_color)
    config.mask_border_color = to_bgr(config.mask_border_color)
    return config


def run(
    src_path: str, dst_path: str, config_path: str, model_path: str, mask_path: str
):
    from os.path import exists
    import json

    import logic
    from consts import TEXT_COLOR

    print(f'[I] Load config from "{config_path}"')
    with open(config_path, "r") as f:
        config_json = json.load(f)
    config = config_to_bgr(AppConfig.parse_obj(config_json))

    # sorry to be here, but for performance...
    print(f"[I] Check mask")
    import cv2

    if mask_path is None:
        print("[W] No mask is provided, disable mask")
        mask = None
    else:
        print(f'[I] Load mask from "{mask_path}"')
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print("[F] Failed to load mask")
            return -1
        mask = logic.Mask(mask_img)

    # For other OSes than macOS, cv2.CAP_FFMPEG can be used
    CV2_API_PREFERENCE = cv2.CAP_AVFOUNDATION

    print(f'[I] Open source "{src_path}"')
    reader = cv2.VideoCapture(src_path, apiPreference=CV2_API_PREFERENCE)

    if not reader.isOpened():
        print(f"[F] Failed to open source")
        return -1

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'[I] Open destination "{dst_path}"')
    if exists(dst_path):
        print(f"[F] Output already exists, cannot save")
        return -1
    writer = cv2.VideoWriter(
        filename=dst_path,
        apiPreference=CV2_API_PREFERENCE,
        fourcc=cv2.VideoWriter_fourcc(*"avc1"),
        fps=fps,
        frameSize=(width, height),
    )
    if not writer.isOpened():
        print(f"[F] Failed to open destination")
        return -1

    # sorry to be here, but for performance...
    print(f'[I] Load model from "{model_path}"')
    from yolov5.helpers import load_model

    model: MODEL_TYPE = load_model(model_path)

    import tqdm

    print("[I] Start processing")
    t = tqdm.tqdm(total=frame_count)
    while True:
        ret, frame = reader.read()
        ret: bool
        frame: cv2.Mat
        if not ret:
            break

        values = logic.run_detect(model, frame, config)
        logic.draw_result(
            values=values,
            cv2_image=frame,
            filename=None,
            mask=mask,
            text_color=TEXT_COLOR,
            config=config,
        )

        writer.write(frame)

        t.update()

    t.close()
    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    status = run(args.src, args.dst, args.config, args.model, args.mask)
    if status != 0:
        import sys

        sys.exit(status)
