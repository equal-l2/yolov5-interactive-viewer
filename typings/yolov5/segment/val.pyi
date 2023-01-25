"""
This type stub file was generated by pyright.
"""

import sys
from yolov5.utils.torch_utils import smart_inference_mode

"""
Validate a trained YOLOv5 segment model on a segment dataset

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ yolov5 segment val --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ yolov5 segment val --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
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
def save_one_txt(predn, save_conf, shape, file): # -> None:
    ...

def save_one_json(predn, jdict, path, class_map, pred_masks): # -> None:
    ...

def process_batch(detections, labels, iouv, pred_masks=..., gt_masks=..., overlap=..., masks=...): # -> Tensor:
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    ...

@smart_inference_mode()
def run(data, weights=..., batch_size=..., batch=..., imgsz=..., img=..., conf_thres=..., iou_thres=..., max_det=..., task=..., device=..., workers=..., single_cls=..., augment=..., verbose=..., save_txt=..., save_hybrid=..., save_conf=..., save_json=..., project=..., name=..., exist_ok=..., half=..., dnn=..., model=..., dataloader=..., save_dir=..., plots=..., overlap=..., mask_downsample_ratio=..., compute_loss=..., callbacks=...):
    ...

def parse_opt(): # -> Namespace:
    ...

def main(opt): # -> None:
    ...

if __name__ == "__main__":
    ...
