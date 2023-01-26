"""
This type stub file was generated by pyright.
"""

import sys
from tensorflow import keras

"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ yolov5 export --weights yolov5s.pt --include saved_model pb tflite tfjs
"""
FILE = ...
ROOT = ...
if str(ROOT) not in sys.path:
    ...
class TFBN(keras.layers.Layer):
    def __init__(self, w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFPad(keras.layers.Layer):
    def __init__(self, pad) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFConv(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., s=..., p=..., g=..., act=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFDWConv(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., s=..., p=..., act=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFDWConvTranspose2d(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., s=..., p1=..., p2=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFFocus(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., s=..., p=..., g=..., act=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFBottleneck(keras.layers.Layer):
    def __init__(self, c1, c2, shortcut=..., g=..., e=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFCrossConv(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., s=..., g=..., e=..., shortcut=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFConv2d(keras.layers.Layer):
    def __init__(self, c1, c2, k, s=..., g=..., bias=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFBottleneckCSP(keras.layers.Layer):
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFC3(keras.layers.Layer):
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFC3x(keras.layers.Layer):
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFSPP(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFSPPF(keras.layers.Layer):
    def __init__(self, c1, c2, k=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFDetect(keras.layers.Layer):
    def __init__(self, nc=..., anchors=..., ch=..., imgsz=..., w=...) -> None:
        ...
    
    def call(self, inputs): # -> tuple[Unknown]:
        ...
    


class TFSegment(TFDetect):
    def __init__(self, nc=..., anchors=..., nm=..., npr=..., ch=..., imgsz=..., w=...) -> None:
        ...
    
    def call(self, x): # -> tuple[Unknown | tuple[Unknown], Unknown] | tuple[Unknown, Unknown]:
        ...
    


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=..., c2=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFUpsample(keras.layers.Layer):
    def __init__(self, size, scale_factor, mode, w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFConcat(keras.layers.Layer):
    def __init__(self, dimension=..., w=...) -> None:
        ...
    
    def call(self, inputs):
        ...
    


def parse_model(d, ch, model, imgsz): # -> tuple[Unknown, list[Unknown]]:
    ...

class TFModel:
    def __init__(self, cfg=..., ch=..., nc=..., model=..., imgsz=...) -> None:
        ...
    
    def predict(self, inputs, tf_nms=..., agnostic_nms=..., topk_per_class=..., topk_all=..., iou_thres=..., conf_thres=...): # -> tuple[Unknown]:
        ...
    


class AgnosticNMS(keras.layers.Layer):
    def call(self, input, topk_all, iou_thres, conf_thres):
        ...
    


def activations(act=...): # -> (x: Unknown) -> Unknown:
    ...

def representative_dataset_gen(dataset, ncalib=...): # -> Generator[list[NDArray[floating[_32Bit]]], None, None]:
    ...

def run(weights=..., imgsz=..., batch_size=..., dynamic=...): # -> None:
    ...

def parse_opt(): # -> Namespace:
    ...

def main(): # -> None:
    ...

if __name__ == "__main__":
    ...