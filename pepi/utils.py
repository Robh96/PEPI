





import builtins
import os

class create_parameters:
    def __init__(self, **kwargs):
        # Set default values
        self.time_skip = 0
        self.time_length = 1e6
        self.time_slice = 1
        self.overlap = 0.5
        self.xlim = [112, 491]
        self.ylim = [47, 556]
        self.pixel_size = 2
        self.moving_average = True
        self.moving_average_window = 5
        self.half_life = 109.8
        self.save_path = 'output'
        

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter '{key}' is not valid for this object.")

def print(*args, **kwargs):
    """ This is a workaround to make the print function work with the parallel backend. """
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

def make_folders(path, params):
    """Creates folders for the path if they do not exist."""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    