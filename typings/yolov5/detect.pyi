"""
This type stub file was generated by pyright.
"""

import sys
from yolov5.utils.torch_utils import smart_inference_mode

"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolov5 detect --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
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
"""
FILE = ...
ROOT = ...
if str(ROOT) not in sys.path:
    ...
ROOT = ...
@smart_inference_mode()
def run(weights=..., source=..., data=..., imgsz=..., img=..., conf_thres=..., iou_thres=..., max_det=..., device=..., view_img=..., save_txt=..., save_conf=..., save_crop=..., nosave=..., classes=..., agnostic_nms=..., augment=..., visualize=..., update=..., project=..., name=..., exist_ok=..., line_thickness=..., hide_labels=..., hide_conf=..., half=..., dnn=..., vid_stride=...):
    ...

def parse_opt(): # -> Namespace:
    ...

def main(): # -> None:
    ...

if __name__ == "__main__":
    ...
