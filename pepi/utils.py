





import builtins
import os
import multiprocessing as mp
from functools import wraps

class create_parameters:
    def __init__(self, **kwargs):
        
        self.time_skip = 0
        self.time_length = 1e6
        self.time_slice = 1
        self.overlap = 0.5
        self.xlim = [112, 491]
        self.ylim = [47, 556]
        self.pixel_size = 2
        self.half_life = 109.8
        self.save_path = 'output'
        self.moving_average = False
        self.moving_average_window = 5
        self.volume = False
        self.histogram = True
        self.pde = True
        

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

    dir_names = ['heatmap']
    
    if params.histogram:
        dir_names.extend(['x_histogram', 'y_histogram'])

    if params.pde:
        dir_names.extend(['x_pde', 'y_pde'])
    
    if params.moving_average:
        dir_names.append('average_heatmap')

        if params.histogram:
            dir_names.extend(['average_x_histogram', 'average_y_histogram'])

        if params.pde:
            dir_names.extend(['average_x_pde', 'average_y_pde'])
    
    if params.volume:
        dir_names.append('volume_map')

    directories = [os.path.join(path, dir_name) for dir_name in dir_names]
    directories.insert(0, path)

    for dir in directories:
        os.makedirs(dir, exist_ok=True)




def parallelise(callback, num_processes=4):
    """
    Decorator to run a function in parallel using multiprocessing.
    
    Args:
    - callback: Function to handle the results of the parallelized function.
    - num_processes: Number of worker processes to use.
    
    Returns:
    - Decorated function that runs in parallel and uses the callback.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a pool of worker processes
            with mp.Pool(processes=num_processes) as pool:
                # Prepare the inputs for the function if needed, here args[0] is expected to be iterable
                results = [pool.apply_async(func, args=(arg,), callback=callback) for arg in args[0]]
                # Close the pool and wait for the tasks to complete
                pool.close()
                pool.join()
        return wrapper
    return decorator