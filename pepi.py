import pept
import numpy as np
from scipy.ndimage import zoom
# from scipy.stats import gaussian_kde
from cv2 import bilateralFilter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
import dill
import os
from typing import Callable, List, Tuple, Optional, Union, Any
        
class PepiData:
    '''
    A class to hold the voxel and pixel data, along with associated metadata.

    Attributes:
        voxels (Optional[np.ndarray]): A 3D numpy array representing the voxel data.
        pixels (Optional[List[np.ndarray]]): A list of 2D numpy arrays representing pixel projections (xy, xz, yz).
        xlim (Optional[Tuple[float, float]]): The limits of the x-axis.
        ylim (Optional[Tuple[float, float]]): The limits of the y-axis.
        zlim (Optional[Tuple[float, float]]): The limits of the z-axis.
        id (Optional[int]): An identifier for the data.
        time (Optional[float]): The time at which the data was acquired.
    '''
    
    def __init__(self, voxels=None, pixels=None, xlim=None, ylim=None, zlim=None, id=None, time=None):
        self._voxels = voxels
        self._pixels = pixels
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim
        self._id = id
        self._time = time

    @property
    def voxels(self) -> Optional[np.ndarray]:
        return self._voxels

    @voxels.setter
    def voxels(self, value: Optional[np.ndarray]):
        self._voxels = value
    
    @property
    def pixels(self) -> Optional[List[np.ndarray]]:
        return self._pixels
    
    @pixels.setter
    def pixels(self, value: List[np.ndarray]):
        self._pixels = value

    @property
    def xlim(self) -> Optional[Tuple[float, float]]:
        return self._xlim

    @xlim.setter
    def xlim(self, value: Optional[Tuple[float, float]]):
        self._xlim = value

    @property
    def ylim(self) -> Optional[Tuple[float, float]]:
        return self._ylim

    @ylim.setter
    def ylim(self, value: Optional[Tuple[float, float]]):
        self._ylim = value

    @property
    def zlim(self) -> Optional[Tuple[float, float]]:
        return self._zlim

    @zlim.setter
    def zlim(self, value: Optional[Tuple[float, float]]):
        self._zlim = value

    @property
    def id(self) -> Optional[int]:
        return self._id

    @id.setter
    def id(self, value: Optional[int]):
        self._id = value

    @property
    def time(self) -> Optional[float]:
        return self._time

    @time.setter
    def time(self, value: Optional[float]):
        self._time = value


class PepiSave(pept.base.Filter):
    '''
    Saves the PepiData object to a file using dill serialization.

    The saved file is named "pepi_voxels_{id:04d}.pkl" and stored in the
    specified save path.

    Attributes:
        save_path (str): The path to the directory where the PepiData objects will be saved.
    '''

    def __init__(self, save_path: str):
        self.save_path = save_path

    def fit_sample(self, sample: PepiData) -> PepiData:
        data = sample
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        id = data.id if data.id is not None else 0
        filename = os.path.join(self.save_path, f"pepi_voxels_{id:04d}.pkl")
        with open(filename, 'wb') as f:
            dill.dump(data, f)
        return data
    
class PepiLoad(pept.base.Filter):
    '''
    Loads a PepiData object from a file using dill serialization.

    Attributes:
        load_path (str): The path to the directory where the PepiData objects are stored.
    '''

    def __init__(self, load_path: str):
        self.load_path = load_path

    def fit_sample(self, sample: PepiData) -> PepiData:
        return sample

    def load_sample(self, id: int) -> PepiData:
        filename = os.path.join(self.load_path, f"pepi_voxels_{id:04d}.pkl")
        with open(filename, 'rb') as f:
            pepi_obj = dill.load(f)
        return pepi_obj

    def load_all(self) -> List[PepiData]:
        files = os.listdir(self.load_path)
        pepi_objs = []
        for file in files:
            with open(os.path.join(self.load_path, file), 'rb') as f:
                pepi_obj = dill.load(f)
            pepi_objs.append(pepi_obj)
        return pepi_objs

class Norm(pept.base.Filter):
    '''
    Normalizes the voxel and pixel data in a PepiData object.

    The normalization is performed based on the 99th percentile of the voxel
    and pixel values from the first sample (id=0). This ensures that the
    normalization is consistent across all samples.

    Attributes:
        vox_max (Optional[float]): The 99th percentile of the voxel values from the first sample.
        pix_max (Optional[np.ndarray]): The 99th percentile of the pixel values from the first sample, for each projection.
    '''

    def __init__(self):
        self.vox_max = None
        self.pix_max = None

    def fit_sample(self, sample: PepiData) -> PepiData:
        '''
        Normalizes the voxel and pixel data in a PepiData object.

        Args:
            sample (PepiData): The PepiData object to normalize.
        
        Returns:
            PepiData: The normalized PepiData object.
        '''

        data = sample
        id = data.id
        voxels = data.voxels
        pixels = data.pixels # xy, xz, yz projections
        vox_max = np.percentile(voxels[voxels > 0], 99) if voxels is not None else None
        pix_max = np.array([np.percentile(p[p > 0], 99) for p in pixels]) if pixels is not None else None
        self.vox_max = max(self.vox_max, vox_max) if self.vox_max is not None else vox_max
        self.pix_max = np.maximum(self.pix_max, pix_max) if self.pix_max is not None else pix_max
        voxels /= self.vox_max if self.vox_max is not None else voxels
        pixels = [p / self.pix_max[i] for i, p in enumerate(pixels)] if self.pix_max is not None else pixels
        data.voxels = voxels
        data.pixels = pixels
        return data
        
    def load_and_normalise(self, path: str) -> PepiData:
        '''
        Loads and normalises a specific PepiData object from a file.
        '''

        with open(path, "rb") as f:
            pepi_obj = dill.load(f)
        norm_sample = self.fit_sample(pepi_obj)
        return norm_sample

class Pepi(pept.base.LineDataFilter):
    '''
    The core transformations of the LineData in the PEPI algorithm.
    Implemented as a pept.base.LineDataFilter to be used in a pept pipeline.
    This class discretises the LoRs into voxels, calculates cutpoints, and
    superimposes the voxels. It also applies a half-life decay and geometric efficiency
    correction to the voxels. The resulting voxel data is then used to create a
    PepiData object, which contains the voxel and pixel data, along with metadata.

    Args:
        line_data (pept.LineData): The input LineData object.
        cell_size (float): The size of the voxel cells in mm.
        xlim (Tuple[float, float], optional): The limits of the x-axis. Defaults to None, which infers from line_data.
        ylim (Tuple[float, float], optional): The limits of the y-axis. Defaults to None, which infers from line_data.
        zlim (Tuple[float, float], optional): The limits of the z-axis. Defaults to None, which infers from line_data.
        egeom (np.ndarray, optional): The geometric efficiency of the detector. Defaults to None, which calculates it.
        half_life (Union[float, int], optional): The half-life of the tracer in minutes. Defaults to None, which sets it to infinity.
        threshold (bool, optional): Whether to apply a threshold to the voxel data. Defaults to True.
        dims (Union[int, List[int]], optional): The dimensions to include in the output PepiData object. Defaults to [2, 3].

    Returns:
        None

    Example:
        >>> line_data = pept.LineData(...)
        >>> pepi_filter = Pepi(line_data, cell_size=4)
    '''

    def __init__(
        self,
        line_data: pept.LineData,
        cell_size: float,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        zlim: Tuple[float, float] = None,
        egeom = None,
        half_life: Union[float, int] = None,
        threshold: bool = True,
        dims: Union[int, List[int]] = [2, 3]
    ):
        
        lims = Helpers.find_limits(line_data)
        self.xlim = lims[0] if xlim is None else xlim
        self.ylim = lims[1] if ylim is None else ylim
        self.zlim = lims[2] if zlim is None else zlim
        print(f"xlim: {self.xlim}, ylim: {self.ylim}, zlim: {self.zlim}")
        self.cell_size = cell_size
        self.dims = dims
        self.shape = Helpers.calc_shape(self.xlim, self.ylim, self.zlim, self.cell_size)
        self.egeom = Helpers.egeom(self.xlim, self.ylim, self.zlim, self.shape) if egeom is None else egeom
        self.half_life = np.inf if half_life is None else half_life
        self.threshold = threshold
        self.id_times = np.array([[sample.lines[0][0]] for sample in line_data])
    
    def fit_sample(
        self,
        sample: pept.LineData,
    ) -> PepiData:
        '''
        Executes the PEPI algorithm on the LineData object.

        Args:
            sample (pept.LineData): A LineData object containing the data.

        Returns:
            PepiData: A PepiData object containing the processed voxel and pixel data.

        Example:
            >>> line_data = pept.LineData(...)
            >>> pepi_filter = Pepi(line_data, cell_size=4)
            >>> pepi_data = pepi_filter.fit_sample(line_data)
        '''

        data = sample
        time = data.lines[0, 0] # ms
        id = Helpers.get_id(self.id_times, time)
        lor_voxels = Pepi.voxels_from_lines(data, self.xlim, self.ylim, self.zlim, self.shape)
        cutpoints = Pepi.cutpoints_from_lines(data, self.xlim, self.ylim, self.zlim)
        point_voxels = Pepi.voxels_from_cutpoints(cutpoints, self.xlim, self.ylim, self.zlim, self.shape)
        voxels = Pepi.superimpose_voxels(lor_voxels, point_voxels)
        voxels = Pepi.half_life_decay(voxels, time, self.half_life)
        voxels /= self.egeom
        if self.threshold:
            voxels = Pepi.threshold_voxels(voxels)
        pixels = Helpers.vox_to_pix(voxels)
        return self._create_pepi_data(voxels, pixels, id, time)


    def _create_pepi_data(self, voxels, pixels, id, time):
        '''
        Helper function to create a PepiData object based on the value of self.dims.

        Args:
            voxels (np.ndarray): The 3D voxel data.
            pixels (List[np.ndarray]): The list of 2D pixel projections.
            id (int): The identifier for the data.
            time (float): The time at which the data was acquired.

        Returns:
            PepiData: A PepiData object containing the specified data.

        Raises:
            ValueError: If `dims` is not 2, 3, or [2,3].

        Example:
            >>> pepi_data = self._create_pepi_data(voxels, pixels, id, time)
        '''

        if isinstance(self.dims, list) and 2 in self.dims and 3 in self.dims:
            return PepiData(
                voxels=voxels,
                pixels=pixels,
                xlim=self.xlim,
                ylim=self.ylim,
                zlim=self.zlim,
                id=id,
                time=time
            )
        elif self.dims == 3 or (isinstance(self.dims, list) and 3 in self.dims):
            return PepiData(
                voxels=voxels,
                xlim=self.xlim,
                ylim=self.ylim,
                zlim=self.zlim,
                id=id,
                time=time
            )
        elif self.dims == 2 or (isinstance(self.dims, list) and 2 in self.dims):
            return PepiData(
                pixels=pixels,
                xlim=self.xlim,
                ylim=self.ylim,
                zlim=self.zlim,
                id=id,
                time=time
            )
        else:
            raise ValueError('dims must be 2 or 3 or [2,3]')

    
    @staticmethod
    def voxels_from_lines(
        line_data: pept.LineData,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        '''
        Given LineData.lines object, limits, and shape,
        discretises LoRs into voxels and returns a 3-D array of voxels.

        Args:
            line_data (pept.LineData): The input LineData object.
            xlim (Tuple[float, float]): The limits of the x-axis.
            ylim (Tuple[float, float]): The limits of the y-axis.
            zlim (Tuple[float, float]): The limits of the z-axis.
            shape (Tuple[int, int, int]): The shape of the voxel grid.

        Returns:
            np.ndarray: A 3D NumPy array representing the voxel data.

        Example:
            >>> voxels = Pepi.voxels_from_lines(line_data, xlim, ylim, zlim, shape)
        '''
        
        lor_voxels = pept.Voxels.from_lines(
            line_data,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            number_of_voxels = shape,
            ).voxels
        return lor_voxels
    
    @staticmethod
    def cutpoints_from_lines(
        line_data: pept.LineData,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        max_distance: float = 0.05,
        verbose: bool = False
    ) -> pept.PointData:
        '''
        Given LineData.lines object, limits, and max_distance,
        calculates cutpoints and returns a PointData object.

        Args:
            line_data (pept.LineData): The input LineData object.
            xlim (Tuple[float, float]): The limits of the x-axis.
            ylim (Tuple[float, float]): The limits of the y-axis.
            zlim (Tuple[float, float]): The limits of the z-axis.
            max_distance (float, optional): The maximum distance between cutpoints. Defaults to 0.05.
            verbose (bool, optional): Whether to print progress. Defaults to False.

        Returns:
            pept.PointData: A PointData object containing the cutpoints.

        Example:
            >>> cutpoints = Pepi.cutpoints_from_lines(line_data, xlim, ylim, zlim)
        '''

        print(f"Length of cutpoint line_data: {len(line_data.lines)}")
        ctpts = pept.tracking.Cutpoints(
            max_distance=max_distance, # mm
            cutoffs=[
                xlim[0],
                xlim[1],
                ylim[0],
                ylim[1],
                zlim[0],
                zlim[1]
                ]).fit(line_data, verbose=verbose)
        return pept.PointData(ctpts)

    @staticmethod
    def voxels_from_cutpoints(
        cutpoints: pept.PointData,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        '''
        Bin cutpoint point data positions onto 3-D grid.
        Input must be a PointData object.
        Returns a np.ndarray[ndim=3, dtype=float64]
        containing frequency of points per voxel.

        Args:
            cutpoints (pept.PointData): The input PointData object.
            xlim (Tuple[float, float]): The limits of the x-axis.
            ylim (Tuple[float, float]): The limits of the y-axis.
            zlim (Tuple[float, float]): The limits of the z-axis.
            shape (Tuple[int, int, int]): The shape of the voxel grid.

        Returns:
            np.ndarray: A 3D NumPy array representing the voxel data.

        Example:
            >>> voxels = Pepi.voxels_from_cutpoints(cutpoints, xlim, ylim, zlim, shape)
        '''
        
        points = cutpoints.points[:, [1, 2, 3]]
        bins = [np.linspace(xlim[0], xlim[1], shape[0] + 1),
                np.linspace(ylim[0], ylim[1], shape[1] + 1),
                np.linspace(zlim[0], zlim[1], shape[2] + 1)]
        voxels, _ = np.histogramdd(points, bins=bins)
        return voxels
    
    @staticmethod
    def superimpose_voxels(
        lor_voxels: np.ndarray,
        point_voxels: np.ndarray
    ) -> np.ndarray:
        '''
        Given discretised LoRs and cutpoints, superimpose LoR and point
        voxels and return a 3-D array.
        This adjusts for the fact that LoRs can have multiple cutpoints.

        Args:
            lor_voxels (np.ndarray): The voxel data from the lines of response.
            point_voxels (np.ndarray): The voxel data from the cutpoints.

        Returns:
            np.ndarray: A 3D NumPy array representing the superimposed voxel data.

        Example:
            >>> voxels = Pepi.superimpose_voxels(lor_voxels, point_voxels)
        '''

        return np.where(point_voxels > 0, lor_voxels / 2, 0)


    @staticmethod
    def half_life_decay(
        voxels: np.ndarray,
        time: float,
        half_life: float
    ) -> np.ndarray:
        '''
        Applies a half-life decay to the voxels. Returns a modified copy of the data.

        Args:
            voxels (np.ndarray): The input voxel data.
            time (float): The time at which the data was acquired.
            half_life (float): The half-life of the tracer in minutes.

        Returns:
            np.ndarray: A 3D NumPy array representing the voxel data with half-life decay applied.

        Example:
            >>> voxels = Pepi.half_life_decay(voxels, time, half_life)
        '''

        voxels /= 2 ** (time / (half_life * 60))
        return voxels

    @staticmethod
    def threshold_voxels(
        voxels: np.ndarray
    ) -> np.ndarray:
        '''
        Thresholds the voxels based on the Gini coefficient of the data.
        Return a modified copy of the data. Used as a dynamic denoiser.
        
        The array is flattened and normalised, then the Gini coefficient
        is calculated. The Gini coefficient is calculated as the ratio of
        the area between the line of equality and the Lorenz curve to the
        total area under the line of equality. The threshold is set to
        remove the bottom sqrt(gini) of the data.

        Args:
            voxels (np.ndarray): The input voxel data.

        Returns:
            np.ndarray: A 3D NumPy array representing the thresholded voxel data.

        Example:
            >>> voxels = Pepi.threshold_voxels(voxels)
        '''

        flattened_sorted = np.sort(voxels.flatten())
        flattened_sorted = flattened_sorted[flattened_sorted > 0]
        cumulative_sum = np.cumsum(flattened_sorted)
        cumulative_sum /= cumulative_sum[-1]
        gini_coefficient = 1 - 2 * np.trapz(cumulative_sum, dx = 1 / (len(flattened_sorted)))
        index = np.where(cumulative_sum >= np.sqrt(gini_coefficient))[0][0] + 1
        threshold = flattened_sorted[index]
        voxels[voxels < threshold] = 0
        return voxels


class Helpers:
    '''
    Contains helper functions for PEPI processing.

    This class provides static methods for various tasks such as calculating voxel grid shapes,
    upscaling grids, computing geometric efficiency, finding data limits, converting voxels to pixels,
    reading line data from files, and cropping line data based on time.
    '''

    @staticmethod
    def get_id(sample_time_list, time):
        '''
        Given a list of sample times, returns the index of the
        sample that is closest to the given time.

        Args:
            sample_time_list (List[float]): A list of sample times.
            time (float): The time to find the closest sample for.

        Returns:
            int: The index of the sample time closest to the given time.

        Example:
            >>> sample_times = [1.0, 2.0, 3.0]
            >>> time_to_find = 2.2
            >>> index = Helpers.get_id(sample_times, time_to_find)
        '''

        return np.argmin(np.abs(sample_time_list - time))

    @staticmethod
    def calc_shape(
        xlim:Tuple[float, float],
        ylim:Tuple[float, float],
        zlim:Tuple[float, float],
        cell_size: float
    ) -> Tuple[int, int, int]:
        '''
        Given the xyz limits as tuple (xmin, xmax), (ymin, ymax), (zmin, zmax),
        and pixel size in mm. Calculates the shape of the data,
        returned as a list [x_res, y_res, z_res]

        Args:
            xlim (Tuple[float, float]): The limits of the x-axis (xmin, xmax).
            ylim (Tuple[float, float]): The limits of the y-axis (ymin, ymax).
            zlim (Tuple[float, float]): The limits of the z-axis (zmin, zmax).
            cell_size (float): The size of the voxel cells in mm.

        Returns:
            Tuple[int, int, int]: The shape of the data as (x_res, y_res, z_res).

        Example:
            >>> x_limits = (0, 100)
            >>> y_limits = (0, 80)
            >>> z_limits = (0, 60)
            >>> voxel_size = 4
            >>> shape = Helpers.calc_shape(x_limits, y_limits, z_limits, voxel_size)
        '''

        limits = [xlim, ylim, zlim]
        res = []
        for lim in limits:
            res.append(int((lim[1] - lim[0]) / cell_size))
        return res

    @staticmethod
    def upscale_grid(
        original_grid: np.ndarray,
        new_shape: Tuple[int, int, int],
        order: int = 1
    ) -> np.ndarray:
        '''
        Upscales the original grid to the new shape.

        Args:
            original_grid (np.ndarray): The original grid to upscale.
            new_shape (Tuple[int, int, int]): The desired shape of the upscaled grid.
            order (int, optional): The order of interpolation. Defaults to 1.

        Returns:
            np.ndarray: The upscaled grid.

        Example:
            >>> original_grid = np.zeros((10, 10, 10))
            >>> new_shape = (20, 20, 20)
            >>> upscaled_grid = Helpers.upscale_grid(original_grid, new_shape)
        '''

        scaling_factors = [n / o for n, o in zip(new_shape, original_grid.shape)]
        return zoom(original_grid, zoom=scaling_factors, order=order)

    @staticmethod
    def egeom(
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        shape: Tuple[int, int, int],
        downscale_factor: int = 4
    ) -> np.ndarray:
        '''
        Returns the memory efficient geometric efficiency of the detector.

        Args:
            xlim (Tuple[float, float]): The limits of the x-axis (xmin, xmax).
            ylim (Tuple[float, float]): The limits of the y-axis (ymin, ymax).
            zlim (Tuple[float, float]): The limits of the z-axis (zmin, zmax).
            shape (Tuple[int, int, int]): The shape of the voxel grid.
            downscale_factor (int, optional): The factor by which to downscale the grid for efficiency calculation. Defaults to 4.

        Returns:
            np.ndarray: The geometric efficiency of the detector as a NumPy array.

        Example:
            >>> x_limits = (0, 100)
            >>> y_limits = (0, 80)
            >>> z_limits = (0, 60)
            >>> voxel_shape = (25, 20, 15)
            >>> efficiency = Helpers.egeom(x_limits, y_limits, z_limits, voxel_shape)
        '''

        low_res_shape = tuple(int(res / downscale_factor) for res in shape)

        x = np.linspace(xlim[0], xlim[1], low_res_shape[0])
        y = np.linspace(ylim[0], ylim[1], low_res_shape[1])
        z = np.linspace(zlim[0], zlim[1], low_res_shape[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        geom = pept.scanners.ADACGeometricEfficiency(abs(zlim[1] - zlim[0]))
        eff = geom(xx, yy, zz)
        return Helpers.upscale_grid(eff, shape, order=1)
    
    @staticmethod
    def find_limits(
        line_data: pept.LineData
    ) -> List[Tuple[float, float]]:
        '''
        Finds the limits of the xyz data. Input is a LineData object.
        Output are three tuples (xlim), (ylim), (zlim) containing
        limits of the data.

        Args:
            line_data (pept.LineData): The input LineData object.

        Returns:
            List[Tuple[float, float]]: A list containing three tuples representing the x, y, and z limits.

        Example:
            >>> line_data = pept.LineData(...)
            >>> limits = Helpers.find_limits(line_data)
        '''

        lines = line_data.lines
        xmin = np.min(lines[:, [1, 4]])
        xmax = np.max(lines[:, [1, 4]])
        ymin = np.min(lines[:, [2, 5]])
        ymax = np.max(lines[:, [2, 5]])
        zmin = np.min(lines[:, [3, 6]])
        zmax = np.max(lines[:, [3, 6]])
        return [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    
    @staticmethod
    def vox_to_pix(voxels: np.ndarray) -> list[np.ndarray]:
        '''
        Given a 3-D array of voxels,
        returns a list of 2-D arrays representing the projections.

        Args:
            voxels (np.ndarray): A 3D NumPy array representing the voxel data.

        Returns:
            list[np.ndarray]: A list of 2D NumPy arrays representing the xy, xz, and yz projections.

        Example:
            >>> voxels = np.zeros((10, 10, 10))
            >>> projections = Helpers.vox_to_pix(voxels)
        '''

        projections = [
            np.mean(voxels, axis=2),  # xy projection
            np.mean(voxels, axis=1),  # xz projection
            np.mean(voxels, axis=0)  # yz projection
        ]
        return projections
    
    @staticmethod
    def read_lines(
        file_path: str,
        time_skip: float = 0,
        time_length: float = np.inf
    ) -> pept.LineData:
        '''
        Checks if file extension is .da gamma rays or .csv LoRs.
        If gamma rays, then Reads the lines from the file

        Args:
            file_path (str): The path to the data file.
            time_skip (float, optional): The amount of time to skip at the beginning of the file. Defaults to 0.
            time_length (float, optional): The length of time to read from the file. Defaults to np.inf.

        Returns:
            pept.LineData: A LineData object containing the lines from the file.

        Example:
            >>> file_path = "data.da01_01"
            >>> line_data = Helpers.read_lines(file_path)
        '''

        _, ext = os.path.splitext(file_path)
        if ext.startswith('.da'): # file is .da gamma rays            
            lines = pept.scanners.adac_forte(file_path)
        elif ext == '.csv':
            lines = pept.read_csv(file_path)
        else:
            raise ValueError(f"{ext} extension is unsupported.")
        return Helpers.crop_lines(lines, time_skip, time_length)

    @staticmethod
    def crop_lines(
        line_data: pept.LineData,
        time_skip: float = None,
        time_length: float = None
    ) -> pept.LineData:
        '''
        input is a pept.LineData object, output is a pept.LineData
        object containing a cropped list of the input list with
        that lie between the time_skip and time_length.

        Args:
            line_data (pept.LineData): The input LineData object.
            time_skip (float, optional): The amount of time to skip at the beginning of the data. Defaults to None.
            time_length (float, optional): The length of time to keep in the data. Defaults to None.

        Returns:
            pept.LineData: A LineData object containing the cropped lines.

        Example:
            >>> line_data = pept.LineData(...)
            >>> cropped_data = Helpers.crop_lines(line_data, time_skip=10, time_length=20)
        '''

        time_skip = time_skip if not None else 0
        time_length = time_length if not None else np.inf
        idx_start = 0
        idx_end = len(line_data.lines)

        if time_skip > 0:
            idx_start = np.where(line_data.lines[:, 0] >= time_skip * 1000)[0][0]
        if line_data.lines[-1, 0] > (time_skip + time_length) * 1000:
            idx_end = np.where(line_data.lines[:, 0] >= (time_skip + time_length) * 1000)[0][0]
        
        lines = pept.LineData(line_data.lines[idx_start:idx_end, :])
        lines.sample_size = line_data.sample_size
        lines.overlap = line_data.overlap
        return lines


class PostProcess(pept.base.Filter):
    '''
    Performs post-processing steps on PepiData objects, including plotting and saving.

    This filter can generate heatmaps, histograms, and probability density estimations (PDEs)
    from the voxel and pixel data within a PepiData object. It also handles saving these
    visualizations to files.
    Implemented as a pept.base.Filter to be used in a pept pipeline.

    Attributes:
        save_path (str, optional): The directory where the generated plots will be saved.
            Defaults to None, which means plots will be displayed but not saved.
        heatmap (bool, optional): If True, generates and saves/displays heatmaps of the pixel data.
            Defaults to True.
        histogram (bool, optional): If True, generates and saves/displays histograms of the voxel data.
            Defaults to False.
        hm_filter (bool, optional): If True, applies a bilateral filter to the heatmap projections
            before plotting. Defaults to False.
        h_bins (int, optional): The number of bins to use for the histograms. Defaults to 10.
    '''

    def __init__(
            self,
            save_path: str = None,
            heatmap: bool = True,
            histogram: bool = False,
            # pde: bool = False,
            hm_filter: bool = False,
            h_bins: int = 10
            ):
        
        self.save_path = save_path
        self.heatmap = heatmap
        self.histogram = histogram
        #self.pde = pde
        self.data = None
        self.hm_filter = hm_filter
        self.h_bins = h_bins
    
    def fit_sample(self, sample: PepiData):
        self.data = sample
        if self.heatmap:
            self.plot_heatmap(save_path=self.save_path, filter=self.hm_filter)
        if self.histogram:
            self.plot_histogram(save_path=self.save_path, n_bins = self.h_bins)
        
        # currently not supporting pde functionality use histogram instead
        # if self.pde:
        #     self.plot_pde(save_path=self.save_path)

    @staticmethod
    def compute_pde(
        voxels: np.ndarray,
        limits: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns the probability density estimation for the voxels in 3-D.
        Input is a 3-D array of voxels and the limits of the data.
        Output is a tuple of x, y, z values and the corresponding density.
        '''

        xlim, ylim, zlim = limits

        # Downsample the voxels array
        downscale_factor = 10  # Adjust as needed
        voxels_downsampled = zoom(voxels, 1 / downscale_factor)

        # Recompute the meshgrid and positions
        x_vals = np.linspace(xlim[0], xlim[1], voxels_downsampled.shape[0])
        y_vals = np.linspace(ylim[0], ylim[1], voxels_downsampled.shape[1])
        z_vals = np.linspace(zlim[0], zlim[1], voxels_downsampled.shape[2])
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        positions = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        voxels_flattened = voxels_downsampled.flatten() / np.sum(voxels_downsampled)
        voxels_flattened = np.clip(voxels_flattened, 0, None)  # Clip values to be non-negative

        # Perform KDE
        kde = gaussian_kde(positions, weights=voxels_flattened)
        density = kde(positions).reshape(voxels_downsampled.shape)

        # x_vals = np.linspace(xlim[0], xlim[1], voxels.shape[0])
        # y_vals = np.linspace(ylim[0], ylim[1], voxels.shape[1])
        # z_vals = np.linspace(zlim[0], zlim[1], voxels.shape[2])
        # x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        # # Flatten the data and the meshgrid
        # positions = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        # voxels_flattened = voxels.flatten() / np.sum(voxels)

        # kde = gaussian_kde(positions, weights=voxels_flattened)
        # density = kde(positions).reshape(voxels.shape)
        return x_vals, y_vals, z_vals, density
    
    @staticmethod
    def compute_histogram(
        voxels: np.ndarray,
        limits: List[Tuple[float, float]],
        n_bins: int = 10
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        '''
        Calculates 1D histograms for each dimension of the input voxels.

        Args:
            voxels: A multi-dimensional NumPy array.
            limits: A list of tuples, where each tuple contains the (min, max) limits for a dimension.
            n_bins: The number of bins to use for each histogram.

        Returns:
            A tuple containing three lists:
            - positions: A list of NumPy arrays, where each array contains the bin positions for a dimension.
            - frequencies: A list of NumPy arrays, where each array contains the bin frequencies for a dimension.
            - bar_widths: A list of NumPy arrays, where each array contains the bin widths for a dimension.
        '''

        max_grid_sum = np.sum(voxels)
        shape = voxels.shape
        num_dims = len(shape)

        positions = []
        frequencies = []
        bar_widths = []

        for dim in range(num_dims):
            min_val, max_val = limits[dim]
            rng = np.linspace(min_val, max_val, shape[dim])

            # Calculate marginal distribution for the current dimension
            axes_to_sum = tuple(i for i in range(num_dims) if i != dim)
            vals = voxels.sum(axis=axes_to_sum) / max_grid_sum

            # Create bin edges
            edges = np.linspace(min_val, max_val, n_bins + 1)

            # Calculate frequencies
            freq, _ = np.histogram(rng, bins=edges, weights=vals)

            # Calculate positions and bar widths
            pos = (edges[:-1] + edges[1:]) / 2
            bar_width = np.diff(edges)

            positions.append(pos)
            frequencies.append(freq)
            bar_widths.append(bar_width)
        histograms = (positions, frequencies, bar_widths)
        return histograms

    @staticmethod
    def make_pde(
        pde: Tuple,
        axis_labels: Optional[List[str]] = None,
        time=None
    ) -> Figure:
        '''
        Given a pde and axis values (1-D, 2-D), plots the pdes.
        Axis labels should be list of strings ["x (mm)", "y (mm)", "z (mm)"].
        If time is given, it is used as the title of the plot.
        Figure is returned.
        '''

        # check if 3-D or 2-D
        x_label, y_label, z_label = ["x (mm)", "y (mm)", "z (mm)"]

        if len(pde) == 4:
            if axis_labels is not None:
                x_label, y_label, z_label = axis_labels
            x_vals, y_vals, z_vals, density = pde
            
            array_size = [
                x_vals[-1] - x_vals[0],
                y_vals[-1] - y_vals[0],
                z_vals[-1] - z_vals[0]
                ]

            aspect_ratios = [size / np.max(array_size) for size in array_size]

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Create a meshgrid for x, y, z
            x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

            # Flatten the arrays
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()
            density = density.ravel()

            alpha = density / np.max(density)
            sc = ax.scatter(x, y, z, c=density, cmap='viridis', alpha=alpha)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label('Probability Density (-)')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_box_aspect([aspect_ratios[0], aspect_ratios[1], aspect_ratios[2]])
            if time is not None:
                plt.title(f'Time (s): {time/1000:.2f}', fontsize=20)
            plt.close(fig)

        elif len(pde) == 3:
            if axis_labels is not None:
                x_label, y_label = axis_labels
            x, y, density = pde
            fig = plt.figure(2, clear=True)
            plt.contourf(x, y, density, cmap='viridis', vmin=0, vmax=10 / len(x))
            plt.xlim(x[0], x[-1])
            plt.ylim(y[0], y[-1])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            if time is not None:
                plt.title(f'Time (s): {time/1000:.2f}', fontsize=20)
            plt.close(fig)

        elif len(pde) == 2:
            if axis_labels is not None:
                x_label = axis_labels
            x, density = pde
            fig = plt.figure(2, clear=True)
            plt.plot(x, density, color='black', lw=2)
            plt.xlim(x[0], x[-1])
            plt.ylim(0, 10 / len(x))  # Adding a small margin above the max value for aesthetics
            plt.xlabel(x_label, fontsize=18)
            plt.ylabel('Probability Density (-)', fontsize=18)
            if time is not None:
                plt.title(f'Time (s): {time/1000:.2f}', fontsize=20)
            plt.close(fig)
        return fig

    @staticmethod
    def make_histogram(
        histogram: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
        axis_labels: Optional[List[str]] = None,
        time=None
    ) -> Union[Figure, List[Figure]]:
        '''
        Plots the histogram of the pixelised data.
        Input is a tuple containing the list of positions (histogram[0]),
        frequencies (histogram[1]) and bar width (histogram[2]) for each dimension.
        A histogram is plotted for each dimension and returned.
        Axis labels should be list of strings ["x (mm)", "y (mm)", "z (mm)"].
        time is used as the title of the plot.
        
        Args:
            histogram: A tuple containing the list of positions, frequencies,
            and bar widths for each dimension.
            axis_labels: A list of strings containing the axis labels.
            must match the number of dimensions ["x (mm)", "y (mm)", "z (mm)"].
            time: A float representing the time in ms.
        
        Returns:
            Union[Figure, List[Figure]]: A Matplotlib Figure object or a list of Figure objects,
            depending on the number of dimensions. If there is only one dimension, a single Figure
            is returned. If there are multiple dimensions, a list of Figure objects is returned,
            where each Figure represents the histogram for one dimension.

        '''

        n_dims = len(histogram[0])
        # check that length of axis_labels matches number of dimensions
        if axis_labels is not None:
            if len(axis_labels) != n_dims:
                raise ValueError('Length of axis_labels must match the number of dimensions.')
        fig_list = []

        for dim in range(n_dims):
            if axis_labels is not None:
                x_label = axis_labels[dim]
            else:
                if dim == 0:
                    x_label = 'x (mm)'
                elif dim == 1:
                    x_label = 'y (mm)'
                else:
                    x_label = 'z (mm)'
            
            positions = histogram[0][dim]
            frequency = histogram[1][dim]
            bar_width = histogram[2][dim]

            fig = plt.figure(2, clear=True)
            plt.bar(
                positions,
                frequency,
                width = bar_width,
                edgecolor='black',
                color='lightblue'
                )
            plt.xlim(positions[0] - bar_width[0] / 2, positions[-1] + bar_width[-1] / 2)
            plt.ylim(0, 1)
            plt.xlabel(x_label, fontsize=18)
            plt.ylabel('Normalised frequency (-)', fontsize=18)
            if time is not None:
                plt.title(f'Time (s): {time/1000:.2f}', fontsize=20)
            plt.close(fig)
            
            fig_list.append(fig)
        return fig_list if n_dims > 1 else fig_list[0]

    @staticmethod
    def make_heatmap(
        projections: List[np.ndarray],
        limits: List[Tuple[float, float]],
        axis_labels: Optional[List[str]] = None,
        time: float = None,
        cbar_range: Optional[List[float]] = None,
        cmap: str = 'turbo',
        logscale: bool = False,
        filter: bool = False,
    ) -> Union[Figure, List[Figure]]:
        '''
        Plots a single or multiple heatmaps of the 2-D projection.
        
        Args:
            projections List[np.ndarray]: containing a 2-D array for each projection and the limits of the data.
                projections = [
                    np.mean(voxels, axis=2),  # xy projection
                    np.mean(voxels, axis=1),  # xz projection
                    np.mean(voxels, axis=0),  # yz projection
                ])
            limits (List[Tuple[float, float]]): A list of tuples, where each tuple contains the (min, max)
                limits for a dimension. The order should correspond to the order of dimensions
                in the projections (e.g., [(xmin, xmax), (ymin, ymax), (zmin, zmax)]).
            axis_labels (List[str], optional): A list of strings specifying the labels for the axes.
                The length of the list should match the number of dimensions.
                Defaults to None, which uses default axis labels.
                e.g. ["x (mm)", "y (mm)", "z (mm)"] if three projections,
                ["x (mm)", "y (mm)"] or ["x (mm)", "z (mm)"] or ["y (mm), "z (mm)"] if two projections.
            time (float, optional): The time associated with the data, used for the plot title.
                Defaults to None, which omits the time from the title.
            cbar_range (List[float], optional): A list of two floats specifying the colorbar range (min, max).
                Defaults to None, which uses a default colorbar range.
            cmap (str, optional): The name of the Matplotlib colormap to use. Defaults to 'turbo'.
            logscale (bool, optional): If True, uses a logarithmic color scale. Defaults to False.
            filter (bool, optional): If True, applies a bilateral filter to the projections before plotting.
                Defaults to False.
            
        Returns:
            Union[Figure, List[Figure]]: A Matplotlib Figure object or a list of Figure objects,
            depending on the number of projections. If there is only one projection, a single Figure
            is returned. If there are multiple projections, a list of Figure objects is returned,
            where each Figure represents the heatmap for one projection.
        '''

        if cbar_range is None:
            cbar_range = [1e-4, 1]  # Default value

        fig_list = []

        for dim, projection in enumerate(projections):
            
            aspect_ratio = projection.shape[0] / projection.shape[1]

            if filter:
                projection = bilateralFilter(
                    projection.astype(np.float32),
                    d=3,
                    sigmaColor=7,
                    sigmaSpace=7
                    )

            if dim == 0:
                x_lim = limits[0]
                y_lim = limits[1]
                x_label = 'x (mm)'
                y_label = 'y (mm)'
            elif dim == 1:
                x_lim = limits[0]
                y_lim = limits[2]
                x_label = 'x (mm)'
                y_label = 'z (mm)'
            elif dim == 2:
                x_lim = limits[1]
                y_lim = limits[2] 
                x_label = 'y (mm)'
                y_label = 'z (mm)'
            else:
                raise ValueError(f"Unexpected projection dimension: {dim}")

            if axis_labels is not None:
                if dim == 0:
                    x_label = axis_labels[0]
                    y_label = axis_labels[1]
                elif dim == 1:
                    x_label = axis_labels[0]
                    y_label = axis_labels[2]
                elif dim == 2:
                    x_label = axis_labels[1]
                    y_label = axis_labels[2]

            if logscale:
                norm = LogNorm(vmin=cbar_range[0], vmax=cbar_range[1])
            else:
                norm = Normalize(vmin=cbar_range[0], vmax=cbar_range[1])

            fig = plt.figure(2, clear=True, figsize=(aspect_ratio * 10, 10))
            ax = plt.gca()
            im = plt.imshow(
                projection.T,
                origin='lower',
                extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                cmap=cmap,
                norm=norm
            )

            plt.xlabel(x_label, fontsize=18)
            plt.ylabel(y_label, fontsize=18)
            if time is not None:
                plt.title(f'Time (s): {time/1000:.2f}', fontsize=20)
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            cbar = plt.colorbar(im, cax = cax)
            cbar.set_label('Normalised concentration (-)', fontsize=14)
            plt.close(fig)
            fig_list.append(fig)
        return fig_list if len(projections) > 1 else fig_list[0]

    @staticmethod
    def save_fig(
        fig: (Union[Figure, List[Figure]]),
        id: int,
        savepath: str,
        filename: str
    ) -> None:
        '''
        Saves a Matplotlib Figure object or a list of Figure objects to files.

        Args:
            fig (Union[Figure, List[Figure]]): The Matplotlib Figure object or a list of Figure objects to save.
            id (int): An identifier for the data, used in the filename.
            savepath (str): The directory where the files will be saved.
            filename (str): The base filename to use for the saved files.
        '''

        if isinstance(fig, list):
            for i, f in enumerate(fig):
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                f.savefig(os.path.join(savepath, f"{filename}_{i}_{id:04d}.png"), dpi=300, bbox_inches='tight')
                plt.close(f)
        else:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            fig.savefig(os.path.join(savepath, f"{filename}_{id:04d}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_histogram(
            self,
            save_path: str = None,
            n_bins: int = 10
        ):
        '''
        Generates and saves/displays histograms of the voxel data.

        Args:
            save_path (str, optional): The directory where the generated plots will be saved.
                Defaults to None, which means plots will be displayed but not saved.
            n_bins (int, optional): The number of bins to use for the histograms.
                Defaults to the value specified during class initialization (self.h_bins).
        '''
        
        histograms = PostProcess.compute_histogram(
            self.data.voxels,
            [self.data.xlim, self.data.ylim, self.data.zlim],
            n_bins=n_bins
            )
        figs = PostProcess.make_histogram(
            histograms,
            axis_labels=["x (mm)", "y (mm)", "z (mm)"],
            time=self.data.time
            )
        if save_path is not None:
            PostProcess.save_fig(
                figs,
                self.data.id,
                self.save_path,
                "histogram")
        else:
            for f in figs:
                f.show()

    def plot_pde(
            self,
            save_path: str = None
        ):
        '''
        Generates and saves/displays a probability density estimation (PDE) plot of the voxel data.

        Args:
            save_path (str, optional): The directory where the generated plot will be saved.
                Defaults to None, which means the plot will be displayed but not saved.
        '''

        pde = PostProcess.compute_pde(
            self.data.voxels,
            [self.data.xlim, self.data.ylim, self.data.zlim]
            )
        
        fig = PostProcess.make_pde(
            pde,
            axis_labels=["x (mm)", "y (mm)", "z (mm)"],
            time=self.data.time
            )
        
        if save_path is not None:
            PostProcess.save_fig(
                fig,
                self.data.id,
                self.save_path,
                "pde")
        else:
            fig.show()

    def plot_heatmap(
            self,
            save_path: str = None,
            filter: bool = False
            ):
        '''
        Generates and saves/displays heatmaps of the pixel data.

        Args:
            save_path (str, optional): The directory where the generated plots will be saved.
                Defaults to None, which means plots will be displayed but not saved.
            filter (bool, optional): If True, applies a bilateral filter to the heatmap projections
                before plotting. Defaults to the value specified during class initialization (self.hm_filter).
        '''

        figs = PostProcess.make_heatmap(
            self.data.pixels,
            [self.data.xlim, self.data.ylim, self.data.zlim],
            time=self.data.time,
            filter=filter
            )
        
        if save_path is not None:
            PostProcess.save_fig(
                figs,
                self.data.id,
                self.save_path,
                "heatmap"
                )
        else:
            for f in figs:
                f.show()

class TemporalFilter(pept.base.Reducer):
    '''
    Temporal filter to remove noise from the data.
    Implemented as a pept.base.Reducer due to requiring multiple frames
    to compute the filter.
    '''

    def __init__(
        self,
        window_range: Tuple[int, int] = (5, 15),
        min_val: float = 1e-4,    
    ):
        self.window_range = window_range
        self.min_val = min_val
        self.buffer = [[], [], []] # stores up to n_max recent frames

    def fit(self, samples: List[PepiData]) -> List[PepiData]:
        '''
        Computes the moving median average over a window of 2-D array data.
        The moving window dynamically adjusts its size based on the standard deviation of the data.
        
        Args: 
            samples (List[PepiData]): A list of PepiData objects containing the pixel data.
        Returns:
            List[PepiData]: A list of PepiData objects with the pixel data replaced by the moving median average.
        '''

        for sample in samples:
            frame = sample.pixels
            if frame is None:
                    print("Empty frame, skipping.")
                    return samples
            
            # Process each projection independently
            median_projections = []
            for i, projection in enumerate(frame):
                # add frame to buffer, remove oldest frame if full
                self.buffer[i].append(projection)
                if len(self.buffer[i]) > self.window_range[1]:
                    self.buffer[i].pop(0)

                # Compute the mean and the standard deviation of the length of the > min pixels for the window
                len_data = [len(p[p > self.min_val]) for p in self.buffer[i]]
                mean = np.mean(len_data)
                std = np.std(len_data)
                n = round(self.window_range[1] - (self.window_range[1] - self.window_range[0]) / (1 + (std / mean)))
                n = np.clip(n, self.window_range[0], self.window_range[1])

                # append the last n frames to the pixel list
                pixel_list = self.buffer[i][-n:]

                # Compute the moving median average for this projection
                median_projections.append(np.median(pixel_list, axis=0))

            sample.pixels = median_projections
        return samples