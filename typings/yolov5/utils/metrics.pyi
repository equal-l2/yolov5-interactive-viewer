"""
This type stub file was generated by pyright.
"""

from yolov5.utils import TryExcept, threaded

"""
Model validation metrics
"""
def fitness(x):
    ...

def smooth(y, f=...): # -> NDArray[floating[Any]]:
    ...

def ap_per_class(tp, conf, pred_cls, target_cls, plot=..., save_dir=..., names=..., eps=..., prefix=...): # -> tuple[Unknown, Unknown, ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]], ndarray[Any, dtype[floating[Any]]], NDArray[float64], Unknown]:
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    ...

def compute_ap(recall, precision): # -> tuple[Any | bool_, NDArray[Any], NDArray[Any]]:
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    ...

class ConfusionMatrix:
    def __init__(self, nc, conf=..., iou_thres=...) -> None:
        ...
    
    def process_batch(self, detections, labels): # -> None:
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        ...
    
    def tp_fp(self): # -> tuple[ndarray[Any, dtype[float64]], Any]:
        ...
    
    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=..., save_dir=..., names=...): # -> None:
        ...
    
    def print(self): # -> None:
        ...
    


def bbox_iou(box1, box2, xywh=..., GIoU=..., DIoU=..., CIoU=..., eps=...):
    ...

def box_iou(box1, box2, eps=...):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    ...

def bbox_ioa(box1, box2, eps=...):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """
    ...

def wh_iou(wh1, wh2, eps=...):
    ...

@threaded
def plot_pr_curve(px, py, ap, save_dir=..., names=...): # -> None:
    ...

@threaded
def plot_mc_curve(px, py, save_dir=..., names=..., xlabel=..., ylabel=...): # -> None:
    ...
