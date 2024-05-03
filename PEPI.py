
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : PEPI.py
# License: GNU v3.0
# Author : Roberto Hart-Villamil <rxh659@student.bham.ac.uk>
# Date   : 01.05.2024


"""
This file contains a PEPI algorithm that accomplishes the following:
1. Reads PEPI Lines of Response from a file using the PEPT library
2. Computes the cutpoints using PEPT library
3. Clusters the cutpoints using HDBSCAN
4. Pixellises the cutpoints to create a 2D image using Konigcell library
6. Applies bilateral filter to the 2D heatmap
7. Plot the 2D heatmap image using matplotlib
8. Plot the x and y histograms using matplotlib
9. A moving average between frames is applied to the data
"""

# Imports
import os
import io
import sys
import traceback
import time as _time
import multiprocessing as mp
import builtins
from joblib import parallel_backend
import pept
from pept.tracking import *
import numpy as np
from cv2 import bilateralFilter, HoughCircles, GaussianBlur, HOUGH_GRADIENT
import csv


from scipy.stats import gaussian_kde
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# methods
def print(*args, **kwargs):
    """ This is a workaround to make the print function work with the parallel backend. """
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)


def PDE(data, xlim, ylim):
    """Returns the probability density estimation for the data."""

    xmin, xmax = xlim
    ymin, ymax = ylim

    # Normalize the data
    max_grid_sum = np.sum(data)

    # Resolution is length of each dimension in data
    resolution = data.shape

    # Create x and y values for each point in the grid
    x_vals = np.repeat(np.linspace(xmin, xmax, resolution[0]), resolution[1])
    y_vals = np.tile(np.linspace(ymin, ymax, resolution[1]), resolution[0])
    data_flattened = data.flatten() / max_grid_sum

    # Create KDE for x and y dimensions
    x_kde = gaussian_kde(x_vals, weights=data_flattened)
    y_kde = gaussian_kde(y_vals, weights=data_flattened)

    # Evaluate KDE on a grid
    x = np.linspace(xmin, xmax, resolution[0])
    y = np.linspace(ymin, ymax, resolution[1])
    
    x_density = x_kde(x)
    x_density /= x_density.sum()
    y_density = y_kde(y)
    y_density /= y_density.sum()

    return x, x_density, y, y_density



def histogram(data, xlim, ylim, n_bins=10):
    """
    a 2D array is given as data, the data is discretised into 1D bins.
    Then this funciton returns the histogram of the data.
    """

    xmin, xmax = xlim
    ymin, ymax = ylim

    # Calculate the sum of the pixels to normalise the histograms
    max_grid_sum = np.sum(data)

    # Resolution is length of each dimension in data
    resolution = data.shape
    # Create a histogram between the ranges of the X and Y axis
    x_rng = np.linspace(xmin, xmax, resolution[0])
    y_rng = np.linspace(ymin, ymax, resolution[1])
    
    x_vals = data.T.sum(axis=0)/max_grid_sum   
    y_vals = data.T.sum(axis=1)/max_grid_sum

    # Create new bin edges
    x_edges = np.linspace(min(x_rng), max(x_rng), n_bins + 1)
    y_edges = np.linspace(min(y_rng), max(y_rng), n_bins + 1)

    # Assign each original x_position to a new bin
    x_bin_pos = np.digitize(x_rng, x_edges, right=True)
    y_bin_pos = np.digitize(y_rng, y_edges, right=True)

    # Initialize new frequencies
    x_frequencies = np.zeros(n_bins)
    y_frequencies = np.zeros(n_bins)

    # Sum frequencies for each new bin
    for i in range(n_bins):
        x_frequencies[i] = np.sum(x_vals[x_bin_pos == i + 1])
        y_frequencies[i] = np.sum(y_vals[y_bin_pos == i + 1])

    # Calculate new x_positions (using midpoints of new bin edges)
    x_positions = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_positions = 0.5 * (y_edges[:-1] + y_edges[1:])

    x_bar_width = np.diff(x_edges)
    y_bar_width = np.diff(y_edges)

    return x_positions, x_frequencies, x_bar_width, y_positions, y_frequencies, y_bar_width


def make_3dscatter(id, voxels, time, xlim, ylim, zlim, min_val, cbar_range, dirpath, filename, logscale=True):
    """Renders the 3D volume of the voxel data using Matplotlib scatter plot."""

    # use upscale_grid to increase resolution of the grid
    #voxels = upscale_grid(voxels, (voxels.shape[0] * 2, voxels.shape[1] * 2, voxels.shape[2] * 2), order=2)

    # clip voxels between min val and 1
    voxels = np.clip(voxels, min_val, 1)

    # Calculate aspect ratios based on voxel dimensions
    aspect_x = voxels.shape[0] / max(voxels.shape)
    aspect_y = voxels.shape[1] / max(voxels.shape)
    aspect_z = voxels.shape[2] / max(voxels.shape)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')


    # Get the coordinates and values of the voxels above the minimum value
    x, y, z = np.where(voxels > min_val)
    values = voxels[voxels > min_val]


    # Calculate the axis values based on voxel dimensions and axis limits
    x_true = np.linspace(xlim[0], xlim[1], voxels.shape[0])[x]
    y_true = np.linspace(ylim[0], ylim[1], voxels.shape[1])[y]
    z_true = np.linspace(zlim[0], zlim[1], voxels.shape[2])[z]

    # Define colors based on voxel_data values
    if logscale:
        norm = LogNorm(vmin=cbar_range[0], vmax=cbar_range[1])
        alpha = np.interp(np.log(values), [np.log(cbar_range[0]), np.log(cbar_range[1])], [0, 1])
    else:
        norm = Normalize(vmin=cbar_range[0], vmax=cbar_range[1])
        alpha = values

    # Create a scatter plot with colors based on voxel values
    sc = ax.scatter(x_true, z_true, y_true, c=values, cmap='turbo', norm=norm, alpha=alpha, s=1, lw=0)# edgecolors='none'

    # Set labels and title
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(zlim[0], zlim[1])
    ax.set_zlim(ylim[0], ylim[1])
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Z Position (mm)', fontsize=12)
    ax.set_zlabel('Y Position (mm)', fontsize=12)
    ax.set_box_aspect([aspect_x, aspect_z, aspect_y])
    ax.set_title('Time (s): ' + str(round(time, 2)), fontsize=16)

    # Add colorbar
    cbar = fig.colorbar(sc, shrink=0.8)
    cbar.set_label('Normalised concentration (-)', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{dirpath}{os.sep}{filename}{id:04d}.png', dpi=300)
    plt.close()


def make_heatmap(id, norm_pixels, time, xlim, ylim, min_val, cbar_range, dirpath, filename, logscale = True):
    """Plots the heatmap of the pixelised data."""
    filter_pixels = norm_pixels.T
    x_res, y_res = filter_pixels.shape
    aspect_ratio = y_res / x_res

    # Apply bilateral smoothing filter
    filter_pixels = bilateralFilter(
        filter_pixels.astype(np.float32),
        d=3,
        sigmaColor=7,
        sigmaSpace=7
        )



    filter_pixels[filter_pixels < min_val] = min_val
    if logscale:
        norm = LogNorm(vmin=cbar_range[0], vmax=cbar_range[1])
        
    else:
        norm = Normalize(vmin=cbar_range[0], vmax=cbar_range[1])

    # Plot the x-y heatmap
    fig1 = plt.figure(1, clear=True, figsize=(aspect_ratio * 10, 10))
    ax = plt.gca()
    im = plt.imshow(
        filter_pixels,
        origin='lower',
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap='turbo',
        norm=norm
        )
        
    plt.xlabel('X Position (mm)', fontsize=18)
    plt.ylabel('Y Position (mm)', fontsize=18)
    plt.title(f'Time (s): {round(time, 2)}', fontsize=20)
    cax = fig1.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Normalised concentration (-)', fontsize=14)
    plt.savefig(f'{dirpath}{os.sep}{filename}_{id:04d}.png', dpi=300, bbox_inches='tight')
    plt.close()




def make_xPDE(id, time, x, x_density, dirpath, filename):
    """Plots the KDE (Probability Density Estimation) for the x-dimension of the data."""
    plt.figure(2, clear=True)
    plt.plot(x, x_density, color='blue', lw=2)
    plt.xlim(x[0], x[-1])
    plt.ylim(0, 10 / len(x))  # Adding a small margin above the max value for aesthetics
    plt.xlabel('X Position (mm)', fontsize=18)
    plt.ylabel('Probability Density (-)', fontsize=18)
    plt.title(f'Time (s): {round(time, 2)}', fontsize=20)
    plt.savefig(os.path.join(dirpath, f"{filename}_{id:04d}.png"), dpi=300, bbox_inches='tight')
    plt.close()



def make_yPDE(id, time, y, y_density, dirpath, filename):
    """Plots the KDE (Probability Density Estimation) for the y-dimension of the data."""
    plt.figure(3, clear=True)
    plt.plot(y, y_density, color='blue', lw=2)
    plt.xlim(y[0], y[-1])
    plt.ylim(0, 10 / len(y))  # Adding a small margin above the max value for aesthetics
    plt.xlabel('Y Position (mm)', fontsize=18)
    plt.ylabel('Probability Density (-)', fontsize=18)
    plt.title(f'Time (s): {round(time, 2)}', fontsize=20)
    plt.savefig(os.path.join(dirpath, f"{filename}_{id:04d}.png"), dpi=300, bbox_inches='tight')
    plt.close()



def make_xhistogram(id, time, x_positions, x_frequencies, x_bar_width, dirpath, filename):
    """Plots the x histogram of the pixelised data."""
    plt.figure(4, clear=True)
    plt.bar(
        x_positions,
        x_frequencies,
        width = x_bar_width,
        edgecolor='black',
        color='lightblue')
    plt.xlim(x_positions[0] - x_bar_width[0] / 2, x_positions[-1] + x_bar_width[-1] / 2)
    plt.ylim(0, 1.0)
    plt.xlabel('X Position (mm)', fontsize=18)
    plt.ylabel('Normalised frequency (-)', fontsize=18)
    plt.title(f'Time (s): {round(time, 2)}', fontsize=20)
    plt.savefig(f'{dirpath}{os.sep}{filename}_{id:04d}.png', dpi=300, bbox_inches='tight')
    plt.close()



def make_yhistogram(id, time, y_positions, y_frequencies, y_bar_width, dirpath, filename):
    """Plots the y histogram of the pixelised data."""
    plt.figure(5, clear=True)
    plt.bar(
        y_positions,
        y_frequencies,
        width = y_bar_width,
        edgecolor='black',
        color='lightblue')
    plt.xlim(y_positions[0] - y_bar_width[0] / 2, y_positions[-1] + y_bar_width[-1] / 2)
    plt.ylim(0, 1.0)
    plt.xlabel('Y Position (mm)', fontsize=18)
    plt.ylabel('Normalised frequency (-)', fontsize=18)
    plt.title(f'Time (s): {round(time, 2)}', fontsize=20)
    plt.savefig(f'{dirpath}{os.sep}{filename}_{id:04d}.png', dpi=300, bbox_inches='tight')
    plt.close()



def make_figs(args):
    """This function makes the figures for the PEPI data."""
    (
        id, voxels, pixels, time, xlim, ylim, zlim,
        min_val, cbar_range, volmap_dir, heatmap_dir, 
        xhist_dir, yhist_dir, xpde_dir, ypde_dir, filename
    ) = args

    print(f"{id}...Figging")

    # return PDE data
    (
        x_grid,
        x_density,
        y_grid,
        y_density
    ) = PDE(
        pixels,
        xlim=xlim, # PEPT coordinates
        ylim=ylim
        )

    # return histogram data
    (
        x_positions, 
        x_frequencies, 
        x_bar_width, 
        y_positions, 
        y_frequencies, 
        y_bar_width
    ) = histogram(
        pixels,
        xlim=xlim, # PEPT coordinates
        ylim=ylim,
        n_bins=10
        )
    
    # Make the plots
    if voxels is not None:
        
        make_3dscatter(id, voxels, time, xlim, ylim, zlim, min_val, cbar_range, volmap_dir, filename, logscale=True)
    make_heatmap(id, pixels, time, xlim, ylim, min_val, cbar_range, heatmap_dir, filename, logscale=True)
    make_xPDE(id, time, x_grid, x_density, xpde_dir, filename)
    make_yPDE(id, time, y_grid, y_density, ypde_dir, filename)
    make_xhistogram(id, time, x_positions, x_frequencies, x_bar_width, xhist_dir, filename)
    make_yhistogram(id, time, y_positions, y_frequencies, y_bar_width, yhist_dir, filename)

    return id



def moving_average(data, n_max = 15, n_min = 5, min_val=1e-4, verbose = False):
    """
    Computes the moving median average over a window of 2-D array data.
    The moving window dynamically adjusts its size based on the standard deviation of the data.
    Returns a list of moving median average arrays.
    """

    # Sort the results by id
    data.sort(key=lambda x: x[0])
    buffer, avg, ids, times = [], [], [], [] # stores up to n_max recent frames

    for frame in data:
        if frame is None:
            print("Empty frame, skipping.")
            continue
        
        # add frame to buffer, remove oldest frame if full
        pixels = frame[1]
        buffer.append(pixels)
        if len(buffer) > n_max:
            buffer.pop(0)

        # Compute the mean and the standard deviation of the length of the > min pixels for the window
        len_data = [len(p[p > min_val]) for p in buffer]
        mean = np.mean(len_data)
        std = np.std(len_data)
        n = round(n_max - (n_max - n_min) / (1 + (std / mean)))
        
        # append the last n frames to the pixel list
        pixel_list = buffer[-n:]

        # Compute the moving average
        avg.append(np.median(pixel_list, axis=0))
        ids.append(frame[0])
        times.append(frame[2])

        if verbose:
            print(f"Computing average for {frame[0]}.", end=" ")
            print(f"Mean: {mean:.2f} | Std: {std:.2f} | n: {n:d}")

    return ids, avg, times



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



def voxels_from_points(points, xlim, ylim, zlim, resolution):
    """Bin PEPT point data positions onto 3-D grid."""
    bins = [np.linspace(xlim[0], xlim[1], resolution[0] + 1),
            np.linspace(ylim[0], ylim[1], resolution[1] + 1),
            np.linspace(zlim[0], zlim[1], resolution[2] + 1)]
    voxels, _ = np.histogramdd(points, bins=bins)
    return voxels



def threshold_voxels(data):
    """
    Thresholds the voxels based on the Gini coefficient of the data.
    Return a modified copy of the data.
    """
    #t=_time.time()

    flattened_sorted = np.sort(data.flatten())
    flattened_sorted = flattened_sorted[flattened_sorted > 0]

    # Compute the cumulative sum of the sorted array
    cumulative_sum = np.cumsum(flattened_sorted)
    
    # Calculate the total sum
    total_sum = cumulative_sum[-1]

    # Normalise the cumulative sum
    cumulative_sum /= total_sum
    n = len(flattened_sorted)
    
    # The Gini coefficient is calculated as the ratio of the area between the line of equality and the Lorenz curve
    # to the total area under the line of equality.
    gini_coefficient = 1 - 2 * np.trapz(cumulative_sum, dx=1/n)

    # For an unequal distribution,threshold the voxels tighter (squared gini coefficient)
    # Gini of 0 is perfect equality, Gini of 1 is perfect inequality
    # sqrt(gini) penalises noisier distributions
    index = np.where(cumulative_sum >= np.sqrt(gini_coefficient))[0][0] + 1
    threshold = flattened_sorted[index]
    data[data < threshold] = 0

    #print(f"Gini coefficient: {gini_coefficient:.2f} | computed in {_time.time() - t:.2f} s.")
    return data



def process_line_samples(id, eg, line_sample, linelen, t_i, run_time, time_slice, xlim, ylim, zlim, max_val_vox, max_val_pix, resolution, isotope_half_life):
    """
    This function processes the line samples and returns the pixelised data.
    The cutpoints are computed, then clustered, then pixelised.
    In PEPT coordinates: X is length, Y is height, Z is depth
    """

    # time-stamp of the current frame
    time = round((line_sample.lines[-1, 0] - t_i) / 1000)
    print(f"Current id: {id}/{linelen - 1}, Time: {time}/{int(run_time - run_time % time_slice):d} s.")

    # Discretise LoRs onto 3D grid
    sys.stdout = io.StringIO() # avoid printing to console
    lor_voxels = pept.Voxels.from_lines(
        line_sample,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        number_of_voxels = resolution,
        ).voxels
    sys.stdout = sys.__stdout__

    # Compute cutpoints from lines
    ctpts = pept.tracking.Cutpoints(
        max_distance=0.05, # mm
        cutoffs=[xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]]
        ).fit(line_sample, verbose=False)

    # bin cutpoints
    cutpoints_voxels = voxels_from_points(
        pept.PointData(ctpts).points[:, [1, 2, 3]],
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        resolution=resolution
        )
    
    # recompute cutpoints in each voxel based on LoRs and adjust for efficiency
    cutpoints_voxels = np.where(cutpoints_voxels > 0, lor_voxels / (2 * eg), 0)

    # threshold the cutpoint voxels
    voxels = threshold_voxels(cutpoints_voxels)

    # Determine how many half-lives have passed, compensate for decay
    voxels /= 2 ** (time / (isotope_half_life * 60))
    
    # Collapse the points onto a 2D space
    pixels = np.mean(voxels, axis=2)

    # Normalise by the maximum from the first frame
    if id == 0:
        max_val_vox = np.percentile(voxels[voxels>0], 99)
        max_val_pix = np.percentile(pixels[pixels>0], 99)

    voxels /= max_val_vox
    pixels /= max_val_pix

    # flip the x direction
    #voxels = np.flip(voxels, 0)
    #pixels = np.flip(pixels, 0)
    print(f"{id}...Max: {np.max(pixels[pixels>0]):.3f} | Mean: {np.mean(pixels[pixels>0]):.3f} | Std: {np.std(pixels[pixels>0]):.3f}")
    return voxels, pixels, time, max_val_vox, max_val_pix



def pepi_run(args):
    """This function enables parallel processing of the data"""
    
    try:
            # Unpack all arguments from the tuple
        (
            id, eg, line_sample, linelen, t_i, run_time, time_slice, 
            xlim, ylim, zlim, max_val_vox, max_val_pix, min_val, 
            cbar_range, resolution, isotope_half_life, volmap_dir, heatmap_dir, 
            xhist_dir, yhist_dir, xpde_dir, ypde_dir, filename
        ) = args


        with parallel_backend('threading', n_jobs=1):
            # return pixelised data
            t = _time.time()
            data = process_line_samples(id, eg, line_sample, linelen, t_i, run_time, time_slice, xlim, ylim, zlim, max_val_vox, max_val_pix, resolution, isotope_half_life)
            if data is None:
                print(f"{id}...Missing data, exiting function.")
                return

            voxels, pixels, time, max_val_vox, max_val_pix = data # pixels is in PEPT coordinates, i.e. x, z, y
            voxels = None # free up memory
            figargs = (
                id, voxels, pixels, time, xlim, ylim, zlim,
                min_val, cbar_range, volmap_dir, heatmap_dir, 
                xhist_dir, yhist_dir, xpde_dir, ypde_dir, filename
                )
            make_figs(figargs)
            print(f"Frame {id} completed in {_time.time() - t:.2f} s.")

    except Exception:
        # print entire traceback
        print(traceback.format_exc())
        # tb = traceback.extract_tb(e.__traceback__)[-1]
        # line_number = tb.lineno
        # file_name = tb.filename
        # print(f"{id}...Error: {e} at line {line_number} in {file_name}")
        return

    return id, voxels, pixels, time, max_val_vox, max_val_pix


if len(sys.argv) != 2:
    print("Usage ", sys.argv[0]," <p>")
    sys.exit()
else:
    p = int(sys.argv[1])


# Global parameters
DIRPATH = r"/rds/projects/i/ingrama-unilever-soap-cfd/PEPTpipeline/PEPI/evolved_geom/01"
PEPI_FILE_PATH = r"/rds/projects/i/ingrama-unilever-soap-cfd/PEPTpipeline/PEPI/data/evolved_geom/EvolvedPEPI.da01"

TIME_SKIP = 280 # Seconds to skip if there is initial noise
TIME_LENGTH = 600 # Seconds to read

TIME_SLICE = 1  # Seconds per slice
OVERLAP = 0.5 # The fraction overlap between the time slices
XMIN, XMAX, YMIN, YMAX = [200, 440, 200, 380] #[180, 410, 200, 380] #[112, 491, 47, 556] # X is length, Y is height, Z is depth
PIXEL_SIZE = (2, 2, 2) # x, y, z size of each pixel in mm




MEDIAN_WINDOW = 5 # Moving window length (frames) for median filter



ISOTOPE_HALF_LIFE = 109.8 # In minutes

MIN_VAL = 1e-4 # Minimum normalised value allowable
COLORBAR_RANGE = [1e-4, 1] # Range of the colorbar


TIMEOUT_DURATION = 900 # seconds per worker





if __name__ == "__main__":

    t = _time.time()
    mp.set_start_method("spawn", force=True)

    # Directories
    filename = os.path.splitext(os.path.basename(PEPI_FILE_PATH))[0]
    print("\nReading data from", filename)
    volmap_dir = DIRPATH + os.sep + 'volume_map'
    heatmap_dir = DIRPATH + os.sep + 'heatmap'
    avg_heatmap_dir = DIRPATH + os.sep + 'average_heatmap'
    xhist_dir = DIRPATH + os.sep + 'x_histogram'
    yhist_dir = DIRPATH + os.sep + 'y_histogram'
    avg_xhist_dir = DIRPATH + os.sep + 'average_x_histogram'
    avg_yhist_dir = DIRPATH + os.sep + 'average_y_histogram'
    xpde_dir = DIRPATH + os.sep + 'x_pde'
    ypde_dir = DIRPATH + os.sep + 'y_pde'
    avg_xpde_dir = DIRPATH + os.sep + 'average_x_pde'
    avg_ypde_dir = DIRPATH + os.sep + 'average_y_pde'
    

    directories = [DIRPATH, avg_heatmap_dir, volmap_dir, heatmap_dir, xhist_dir, yhist_dir, avg_xhist_dir, avg_yhist_dir, xpde_dir, ypde_dir, avg_xpde_dir, avg_ypde_dir]

    for dir in directories:
        os.makedirs(dir, exist_ok=True)
    
    sys.stdout = io.StringIO() # avoid printing to console
    lines = pept.scanners.adac_forte(PEPI_FILE_PATH)
    sys.stdout = sys.__stdout__
    raw_line_data = lines.lines

    # Determine the limits of the data
    zmin = min(min(raw_line_data[:, 3]), min(raw_line_data[:, 6]))
    zmax = max(max(raw_line_data[:, 3]), max(raw_line_data[:, 6]))


    # Determine resolution of the data
    x_resolution = (XMAX - XMIN) / PIXEL_SIZE[0]
    y_resolution = (YMAX - YMIN) / PIXEL_SIZE[1]
    z_resolution = (zmax - zmin) / PIXEL_SIZE[2]
    resolution = (int(x_resolution), int(y_resolution), int(z_resolution))


    xlim = [XMIN, XMAX]
    ylim = [YMIN, YMAX]
    zlim = [zmin, zmax]

    # Compute geometric efficiency of 3-D space
    # eg = 1
    eg = egeom(xlim=xlim,
               ylim=ylim,
               zlim=zlim,
               resolution=resolution)
    

    # Get line data within the time range
    cropped_lines = line_index(raw_line_data, TIME_SKIP, TIME_LENGTH)

    # Create a new pept object with the collimated data
    lines = pept.LineData(cropped_lines)

    # The number of lines that should be returned when iterating over lines
    lines.sample_size = pept.TimeWindow(TIME_SLICE * 1000)

    lines.overlap = pept.TimeWindow(OVERLAP * TIME_SLICE * 1000)

    t_i = lines.lines[0, 0]
    run_time = round((lines.lines[-1, 0] - t_i) / 1000) # gets the last time-step in seconds
    
    # Linelength is the total slices of the data
    linelen = len(range(int(
        (run_time - run_time % (TIME_SLICE - (OVERLAP * TIME_SLICE))) / (TIME_SLICE - (OVERLAP * TIME_SLICE))
        )))

    print(f"Start-up time: {_time.time() - t:.2f} s.")

    # Initialising variables
    max_val_vox, max_val_pix, futures, results, start_id = [], [], [], [], []

    # Pixellise and plot the first sample of data
    for id, line_sample in enumerate(lines[::]):
        args = (
            id,
            eg,
            line_sample,
            linelen,
            t_i,
            run_time,
            TIME_SLICE,
            xlim,
            ylim,
            zlim,
            max_val_vox,
            max_val_pix,
            MIN_VAL,
            COLORBAR_RANGE,
            resolution,
            ISOTOPE_HALF_LIFE,
            volmap_dir,
            heatmap_dir,
            xhist_dir,
            yhist_dir,
            xpde_dir,
            ypde_dir,
            filename
        )
        print(f"{id} has {len(line_sample.lines)} lines.")
        data = pepi_run(args)
        
        if data is None:
            continue
        
        _, _, _, _, max_val_vox, max_val_pix = data
        start_id = id # first valid id
        print("Completed first frame.")
        print("CPU count: ", p)
        break

    # Pixellise and plot the rest of the data
    args_list = []

    for id, line_sample in enumerate(lines[::]):
        if id < start_id:
            continue
        
        args = (
            id,
            eg,
            line_sample,
            linelen,
            t_i,
            run_time,
            TIME_SLICE,
            xlim,
            ylim,
            zlim,
            max_val_vox,
            max_val_pix,
            MIN_VAL,
            COLORBAR_RANGE,
            resolution,
            ISOTOPE_HALF_LIFE,
            volmap_dir,
            heatmap_dir,
            xhist_dir,
            yhist_dir,
            xpde_dir,
            ypde_dir,
            filename
        )
        args_list.append(args)

    results = []

    
    with mp.Pool(processes=p-1) as pool:
        result_objects = [pool.apply_async(pepi_run, (args,)) for args in args_list]
        
        for result_obj in result_objects:
            try:
                result = result_obj.get(timeout=TIMEOUT_DURATION)
                if result is None:
                    print("...Missing results, skipping.")
                    continue

                else:
                    id, _, array, time, _, _ = result
                    result_data = [id, array, time]
                    print(f"{id}...Saving results.")
                    results.append(result_data)

            except mp.context.TimeoutError:
                print("...Worker timed out, skipping.")
                continue


    # Data averaging over a moving window
    average_frames = moving_average(results, n_max=20)
    args_list = []

    for arg_item in zip(*average_frames):
        if arg_item is None:
            continue

        id, frame, time = arg_item
        args = (
            id,
            None,
            frame,
            time,
            xlim,
            ylim,
            None,
            MIN_VAL,
            COLORBAR_RANGE,
            None,
            avg_heatmap_dir,
            avg_xhist_dir,
            avg_yhist_dir,
            avg_xpde_dir,
            avg_ypde_dir,
            filename,
        )
        args_list.append(args)

    print("Items in args_list:", len(args_list))

    # Plot the x-y heatmap for each frame
    with mp.Pool(processes=p-1) as pool:
        result_objects = []
        for args in args_list:
            try:
                result = pool.apply_async(make_figs, (args,))
                if result is None:
                    print("...Missing results, skipping.")
                    continue
                result_objects.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")
        
        for result_obj in result_objects:
            try:
                result = result_obj.get(timeout=TIMEOUT_DURATION)

            except mp.context.TimeoutError:
                print("...Worker timed out, skipping.")
                continue
