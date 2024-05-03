

import numpy as np

def threshold_voxels(data):
    """
    Thresholds the voxels based on the Gini coefficient of the data.
    Return a modified copy of the data.

    For an unequal distribution,threshold the voxels tighter (squared gini coefficient)
    Gini of 0 is perfect equality, Gini of 1 is perfect inequality
    sqrt(gini) penalises noisier distributions
    """

    flattened_sorted = np.sort(data.flatten())
    flattened_sorted = flattened_sorted[flattened_sorted > 0]

    # Compute and normalise the cumulative sum of the sorted array
    cumulative_sum = np.cumsum(flattened_sorted)
    cumulative_sum /= cumulative_sum[-1]
    n = len(flattened_sorted)
    
    gini_coefficient = 1 - 2 * np.trapz(cumulative_sum, dx=1/n)
    index = np.where(cumulative_sum >= np.sqrt(gini_coefficient))[0][0] + 1
    threshold = flattened_sorted[index]
    data[data < threshold] = 0
    return data
