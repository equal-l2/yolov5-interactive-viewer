"""
This type stub file was generated by pyright.
"""

class NeptuneLogger:
    def __init__(self, opt, job_type=...) -> None:
        ...
    
    def track_dataset(self, opt): # -> None:
        ...
    
    def setup_training(self, data_dict):
        ...
    
    def log(self, log_dict): # -> None:
        ...
    
    def end_epoch(self, best_result=...): # -> None:
        ...
    
    def finish_run(self): # -> None:
        ...
    

