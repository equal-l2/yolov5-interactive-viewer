"""
This type stub file was generated by pyright.
"""

"""Main Logger class for ClearML experiment tracking."""
def construct_dataset(clearml_info_string): # -> dict[Unknown, Unknown]:
    """Load in a clearml dataset and fill the internal data_dict with its contents.
    """
    ...

class ClearmlLogger:
    """Log training runs, datasets, models, and predictions to ClearML.

    This logger sends information to ClearML at app.clear.ml or to your own hosted server. By default,
    this information includes hyperparameters, system configuration and metrics, model metrics, code information and
    basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.
    """
    def __init__(self, opt, hyp) -> None:
        """
        - Initialize ClearML Task, this object will capture the experiment
        - Upload dataset version to ClearML Data if opt.upload_dataset is True

        arguments:
        opt (namespace) -- Commandline arguments for this run
        hyp (dict) -- Hyperparameters for this run

        """
        ...
    
    def log_debug_samples(self, files, title=...): # -> None:
        """
        Log files (images) as debug samples in the ClearML task.

        arguments:
        files (List(PosixPath)) a list of file paths in PosixPath format
        title (str) A title that groups together images with the same values
        """
        ...
    
    def log_image_with_boxes(self, image_path, boxes, class_names, image, conf_threshold=...): # -> None:
        """
        Draw the bounding boxes on a single image and report the result as a ClearML debug sample.

        arguments:
        image_path (PosixPath) the path the original image file
        boxes (list): list of scaled predictions in the format - [xmin, ymin, xmax, ymax, confidence, class]
        class_names (dict): dict containing mapping of class int to class name
        image (Tensor): A torch tensor containing the actual image data
        """
        ...
    


