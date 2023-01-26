"""
This type stub file was generated by pyright.
"""

def load_model(model_path, device=..., autoshape=..., verbose=..., hf_token: str = ...): # -> DetectMultiBackend | AutoShape | Module | ModuleList | Ensemble:
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent
        hf_token (str): huggingface read token for private models

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    ...

class YOLOv5:
    def __init__(self, model_path, device=..., load_on_init=...) -> None:
        ...
    
    def load_model(self): # -> None:
        """
        Load yolov5 weight.
        """
        ...
    
    def predict(self, image_list, size=..., augment=...): # -> Any:
        """
        Perform yolov5 prediction using loaded model weights.

        Returns results as a yolov5.models.common.Detections object.
        """
        ...
    


def generate_model_usage_markdown(repo_id, ap50, task=..., input_size=..., dataset_id=...): # -> str:
    ...

def push_model_card_to_hfhub(repo_id, exp_folder, ap50, hf_token=..., input_size=..., task=..., private=..., dataset_id=...): # -> None:
    ...

def push_config_to_hfhub(repo_id, exp_folder, best_ap50=..., input_size=..., task=..., hf_token=..., private=...): # -> None:
    """
    Pushes a yolov5 config to huggingface hub

    Arguments:
        repo_id (str): The name of the repository to create on huggingface.co
        exp_folder (str): The path to the experiment folder
        best_ap50 (float): The best ap50 score of the model
        input_size (int): The input size of the model (default: 640)
        task (str): The task of the model (default: object-detection)
        hf_token (str): The huggingface token to use to push the model
        private (bool): Whether the model should be private or not
    """
    ...

def push_model_to_hfhub(repo_id, exp_folder, hf_token=..., private=...): # -> None:
    """
    Pushes a yolov5 model to huggingface hub

    Arguments:
        repo_id (str): huggingface repo id to be uploaded to
        exp_folder (str): yolov5 experiment folder path
        hf_token (str): huggingface write token
        private (bool): whether to make the repo private or not
    """
    ...

def push_to_hfhub(hf_model_id, hf_token, hf_private, save_dir, hf_dataset_id=..., input_size=..., best_ap50=..., task=...): # -> None:
    ...

def convert_coco_dataset_to_yolo(opt, save_dir): # -> None:
    ...

def upload_to_s3(opt, data, save_dir): # -> None:
    ...
