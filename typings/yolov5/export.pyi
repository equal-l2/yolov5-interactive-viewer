"""
This type stub file was generated by pyright.
"""

import platform
import sys
from yolov5.utils.torch_utils import smart_inference_mode

"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ yolov5 export --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ yolov5 detect --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""
FILE = ...
ROOT = ...
if str(ROOT) not in sys.path:
    ...
if platform.system() != 'Windows':
    ROOT = ...
MACOS = ...
def export_formats(): # -> DataFrame:
    ...

def try_export(inner_func): # -> (*args: Unknown, **kwargs: Unknown) -> (tuple[Unknown, Unknown] | tuple[None, None]):
    ...

@try_export
def export_torchscript(model, im, file, optimize, prefix=...): # -> tuple[Unknown, None]:
    ...

@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=...): # -> tuple[Unknown, Unknown]:
    ...

@try_export
def export_openvino(file, metadata, half, prefix=...): # -> tuple[str, None]:
    ...

@try_export
def export_paddle(model, im, file, metadata, prefix=...): # -> tuple[str, None]:
    ...

@try_export
def export_coreml(model, im, file, int8, half, prefix=...): # -> tuple[Unknown, Unknown]:
    ...

@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=..., verbose=..., prefix=...): # -> tuple[Unknown, None]:
    ...

@try_export
def export_saved_model(model, im, file, dynamic, tf_nms=..., agnostic_nms=..., topk_per_class=..., topk_all=..., iou_thres=..., conf_thres=..., keras=..., prefix=...): # -> tuple[str, Unknown]:
    ...

@try_export
def export_pb(keras_model, file, prefix=...): # -> tuple[Unknown, None]:
    ...

@try_export
def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=...): # -> tuple[str, None]:
    ...

@try_export
def export_edgetpu(file, prefix=...): # -> tuple[str, None]:
    ...

@try_export
def export_tfjs(file, prefix=...): # -> tuple[str, None]:
    ...

def add_tflite_metadata(file, metadata, num_outputs): # -> None:
    ...

@smart_inference_mode()
def run(data=..., weights=..., imgsz=..., img=..., batch_size=..., device=..., include=..., half=..., inplace=..., keras=..., optimize=..., int8=..., dynamic=..., simplify=..., opset=..., verbose=..., workspace=..., nms=..., agnostic_nms=..., topk_per_class=..., topk_all=..., iou_thres=..., conf_thres=...):
    ...

def parse_opt(): # -> Namespace:
    ...

def main(): # -> None:
    ...

if __name__ == "__main__":
    ...