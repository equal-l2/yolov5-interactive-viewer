"""
This type stub file was generated by pyright.
"""

import sys

"""
Train a YOLOv5 classifier model on a classification dataset

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""
FILE = ...
ROOT = ...
if str(ROOT) not in sys.path:
    ...
ROOT = ...
LOCAL_RANK = ...
RANK = ...
WORLD_SIZE = ...
def train(opt, device):
    ...

def parse_opt(known=...): # -> Namespace:
    ...

def main(opt): # -> None:
    ...

def run(**kwargs): # -> Namespace:
    ...

def run_cli(**kwargs): # -> None:
    '''
    To be called from yolov5.cli
    '''
    ...

if __name__ == "__main__":
    opt = ...
