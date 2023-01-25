"""
This type stub file was generated by pyright.
"""

logger = ...
COMET_PREFIX = ...
COMET_MODEL_NAME = ...
COMET_DEFAULT_CHECKPOINT_FILENAME = ...
def download_model_checkpoint(opt, experiment): # -> None:
    ...

def set_opt_parameters(opt, experiment): # -> None:
    """Update the opts Namespace with parameters
    from Comet's ExistingExperiment when resuming a run

    Args:
        opt (argparse.Namespace): Namespace of command line options
        experiment (comet_ml.APIExperiment): Comet API Experiment object
    """
    ...

def check_comet_weights(opt): # -> Literal[True] | None:
    """Downloads model weights from Comet and updates the
    weights path to point to saved weights location

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if weights are successfully downloaded
            else return None
    """
    ...

def check_comet_resume(opt): # -> Literal[True] | None:
    """Restores run parameters to its original state based on the model checkpoint
    and logged Experiment parameters.

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if the run is restored successfully
            else return None
    """
    ...

