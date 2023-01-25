"""
This type stub file was generated by pyright.
"""

import sys
from yolov5.utils.torch_utils import smart_inference_mode

"""
Validate a trained YOLOv5 classification model on a classification dataset

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""
FILE = ...
ROOT = ...
if str(ROOT) not in sys.path:
    ...
ROOT = ...
@smart_inference_mode()
def run(data=..., weights=..., batch_size=..., batch=..., imgsz=..., img=..., device=..., workers=..., verbose=..., project=..., name=..., exist_ok=..., half=..., dnn=..., model=..., dataloader=..., criterion=..., pbar=...): # -> tuple[Unknown, Unknown, Unknown | float]:
    ...

def parse_opt(): # -> Namespace:
    ...

def main(): # -> None:
    ...

if __name__ == "__main__":
    ...
