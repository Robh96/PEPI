

from . import utils
import pept
from scipy.ndimage import zoom
import numpy as np


def get_zlim(data):
    """Returns the z limits of the data."""
    zmin = np.min(np.min(data[:, 3], np.min(data[:, 6])))
    zmax = np.max(np.max(data[:, 3], np.max(data[:, 6])))
    return [zmin, zmax]


def get_resolution(xlim, ylim, zlim, px_size=2):
    """Returns the xyz resolution of the data
      based on provided limits and pixel size."""
    
    x_res = int((xlim[1] - xlim[0]) / px_size)
    y_res = int((ylim[1] - ylim[0]) / px_size)
    z_res = int((zlim[1] - zlim[0]) / px_size)
    return (x_res, y_res, z_res)


def line_index(data, time_skip, time_length):
    """returns the lines that lie between the time_skip and time_length."""
    index_start = np.where(data[:, 0] >= time_skip * 1000)[0][0]
    if data[-1, 0] > (time_skip + time_length) * 1000:
        index_end = np.where(data[:, 0] >= (time_skip + time_length) * 1000)[0][0]
    else:
        index_end = len(data)
    return data[index_start:index_end, :]


def upscale_grid(original_grid, new_shape, order=1):
    """Upscales the original grid to the new shape."""
    scaling_factors = [n / o for n, o in zip(new_shape, original_grid.shape)]
    return zoom(original_grid, zoom=scaling_factors, order=order)



def egeom(xlim, ylim, zlim, resolution):
    """Returns the memory efficient geometric efficiency of the detector."""
    geom = pept.scanners.ADACGeometricEfficiency(abs(zlim[1] - zlim[0]))
    low_res_x = int(resolution[0] / 4)
    low_res_y = int(resolution[1] / 4)
    low_res_z = int(resolution[2] / 4)

    x = np.linspace(xlim[0], xlim[1], low_res_x)
    y = np.linspace(ylim[0], ylim[1], low_res_y)
    z = np.linspace(zlim[0], zlim[1], low_res_z)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    eg = geom(xx, yy, zz)
    return upscale_grid(eg, resolution, order=1)


def load_data(path):
    """Loads the data from the path."""
    data = np.loadtxt(path)
    return data

def samples(data, params=None):
    if params is None:
        params = utils.create_parameters() 
    
    raw_line_data = pept.scanners.adac_forte(data).lines

    crop_lines = line_index(data, 0, 10)

