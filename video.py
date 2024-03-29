#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from structs import AppConfig

if TYPE_CHECKING:
    from structs import RgbTuple

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--mask", required=False)


# Convert all colors in config from RGB to BGR, for cv2
def config_to_bgr(config: AppConfig) -> AppConfig:
    def to_bgr(rgb: RgbTuple) -> tuple[int, int, int]:
        (r, g, b) = rgb
        return (b, g, r)

    render = config.render
    render.bb_config.color = to_bgr(render.bb_config.color)
    render.outsider_config.color = to_bgr(render.outsider_config.color)
    render.mask_border_config.color = to_bgr(render.mask_border_config.color)
    return config


def run(
    src_path: str,
    dst_path: str,
    config_path: str,
    model_path: str,
    mask_path: str | None,
) -> int:
    import json
    import platform

    import logic
    from consts import TEXT_COLOR

    print(f'[I] Load config from "{config_path}"')
    with Path(config_path).open() as f:
        config_json = json.load(f)
    config = config_to_bgr(AppConfig.parse_obj(config_json))

    print("[I] Check mask")
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

    if platform.system() == "Darwin":
        print("[I] macOS detected, using AVFoundation for video I/O")
        cv2_api_preference = cv2.CAP_AVFOUNDATION
    else:
        print("[I] macOS NOT detected, using auto-detected backend for video I/O")
        # There may be a better cap with hardware acceleration, but I don't know about the other platforms...
        cv2_api_preference = cv2.CAP_ANY

    print(f'[I] Open source "{src_path}"')
    reader = cv2.VideoCapture(src_path, apiPreference=cv2_api_preference)

    if not reader.isOpened():
        print("[F] Failed to open source")
        return -1

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'[I] Open destination "{dst_path}"')
    if Path(dst_path).exists():
        print("[F] Output already exists, cannot save")
        return -1

    writer = cv2.VideoWriter(
        filename=dst_path,
        apiPreference=cv2_api_preference,
        fourcc=cv2.VideoWriter_fourcc(*"avc1"),
        fps=fps,
        frameSize=(width, height),
    )

    if not writer.isOpened():
        print("[F] Failed to open destination")
        return -1

    print(f'[I] Load model from "{model_path}"')

    model = logic.Model(model_path)

    import tqdm

    print("[I] Start processing")
    t = tqdm.tqdm(total=frame_count)
    while True:
        ret, frame = reader.read()

        ret: bool
        frame: cv2.Mat
        if not ret:
            break

        values = model.run(frame, config.detect)
        logic.render_result(
            values=values,
            cv2_image=frame,
            filename=None,
            mask=mask,
            text_color=TEXT_COLOR,
            config=config.render,
        )

        writer.write(frame)

        t.update()

    # set the progress bar to 100%
    t.update(frame_count - t.n)

    t.close()
    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    status = run(args.src, args.dst, args.config, args.model, args.mask)
    if status != 0:
        from sys import exit

        exit(status)
