"""
This type stub file was generated by pyright.
"""

import sys
from yolov5.utils.torch_utils import smart_inference_mode

"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
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
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""
FILE = ...
ROOT = ...
if str(ROOT) not in sys.path:
    ...
ROOT = ...
@smart_inference_mode()
def run(weights=..., source=..., data=..., imgsz=..., img=..., conf_thres=..., iou_thres=..., max_det=..., device=..., view_img=..., save_txt=..., save_conf=..., save_crop=..., nosave=..., classes=..., agnostic_nms=..., augment=..., visualize=..., update=..., project=..., name=..., exist_ok=..., line_thickness=..., hide_labels=..., hide_conf=..., half=..., dnn=..., vid_stride=..., retina_masks=...):
    ...

def parse_opt(): # -> Namespace:
    ...

def main(opt): # -> None:
    ...

if __name__ == "__main__":
    opt = ...