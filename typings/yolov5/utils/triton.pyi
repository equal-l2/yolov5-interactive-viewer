"""
This type stub file was generated by pyright.
"""

import typing
import torch

""" Utils to interact with the Triton Inference Server
"""
class TritonRemoteModel:
    """ A wrapper over a model served by the Triton Inference Server. It can
    be configured to communicate over GRPC or HTTP. It accepts Torch Tensors
    as input and returns them as outputs.
    """
    def __init__(self, url: str) -> None:
        """
        Keyword arguments:
        url: Fully qualified address of the Triton server - for e.g. grpc://localhost:8000
        """
        ...
    
    @property
    def runtime(self):
        """Returns the model runtime"""
        ...
    
    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """ Invokes the model. Parameters can be provided via args or kwargs.
        args, if provided, are assumed to match the order of inputs of the model.
        kwargs are matched with the model input names.
        """
        ...
    


