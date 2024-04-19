import warnings
from typing import Tuple, Iterator

import numpy as np
import scipy.interpolate

from core.analyzer import Analyzer
from core.diffusion_array import DiffusionArray


class RadialTimeProfile:
    """
    Represents a radial time profile of the *homogenized* diffusion data. That is a 2D array of the intensities indexed
    by time/frame and distance form the center point.

    Properties:
        ndarray (np.ndarray): The underlying numpy array.
        shape (Tuple[int]): Shape of the ndarray.
        number_of_frames (int): Number of time frames.
        width (int): Width of the radial profile.
        ndim (int): Number of array dimensions.


    Methods:
        __init__(self, diffusion_array: DiffusionArray, center: Tuple[int | float, int | float] = None):
            Initialize a RadialTimeProfile instance with a DiffusionArray and an optional center point.
            If center is not provided, it will be automatically detected.

        frame(self, frame: int | str) -> np.ndarray:
            Get a specific frame from the radial time profile. You can provide the frame index as an integer or a string
             in the format "start:stop:step".

        resized(self, new_radius, degree=1) -> 'RadialTimeProfile':
            Resizes the radial time profile objet to the new radius.

        to_diffusion_array(self, degree=3) -> DiffusionArray:
            Creates a diffusion array from the radial time profile.

    Class Methods:
        _create_data_parallel(diffusion_array, center):
            Create a radial time profile data parallel to the center.
            This is a static method used during initialization.

    Special Methods:
        __iter__(self) -> Iterator:
            Allows iterating over the ndarray using the class instance.

        __array__(self):
            Enables the use of the instance as a numpy array.

    Raises:
        ValueError: If the diffusion_array is not 3-dimensional (time, x, y).
    """

    @staticmethod
    def _create_data_parallel(diffusion_array: DiffusionArray, center: Tuple[float | int, float | int]) -> np.ndarray:
        """
        Create a radial time profile along line segment, that starts and the center is parallel to the edges of the
        diffusion array and is the longest of them.
        This method is intended for internal use and should not be called directly.

        Args:
            diffusion_array (DiffusionArray): A 3-dimensional array containing diffusion data (time, x, y).
            center (Tuple[int | float, int | float]): One end point of the line segment.

        Returns:
            np.ndarray: The radial time profile along the segment.

        Note:
            The radius is parallel with the edges of the diffusion array, which is assumed to be a rectangle.
        """
        width = diffusion_array.width
        height = diffusion_array.height
        center_x, center_y = (round(cord) for cord in center)
        edge_distances = [center_x, center_y, width - center_x, height - center_y]
        max_distance = max(edge_distances)

        if center_x == max_distance:
            squeezed = np.squeeze(diffusion_array.ndarray[:, :, center_y:center_y + 1, center_x:0:-1])

        elif center_y == max_distance:
            squeezed = np.squeeze(diffusion_array.ndarray[:, :, center_y:0:-1, center_x:center_x + 1])

        elif width - center_x == max_distance:
            squeezed = np.squeeze(diffusion_array.ndarray[:, :, center_y:center_y + 1, center_x:width])

        else:
            squeezed = np.squeeze(diffusion_array.ndarray[:, :, center_y:height, center_x:center_x + 1])

        return squeezed

    def __init__(
            self,
            diffusion_array: DiffusionArray,
            center: Tuple[int | float, int | float] = None,
            profile_array: np.ndarray = None
    ):
        """
        Initialize a RadialTimeProfile instance with the provided DiffusionArray and an optional center point.

        Args:
            diffusion_array (DiffusionArray): A 3-dimensional array containing diffusion data (time, x, y).
            center (Tuple[int | float, int | float], optional): The center point of the radial profile. If not provided,
                it will be automatically detected using the Analyzer class.

        Raises:
            ValueError: If the diffusion_array is not 3-dimensional (time, x, y).
        """
        if diffusion_array.ndim != 3:
            raise ValueError('diffusion_array must be 3 dimensional (time, x, y)')

        if center is None:
            center = Analyzer(diffusion_array).detect_diffusion_start_place()

        self._original_width = diffusion_array.width
        self._original_height = diffusion_array.height
        self._original_center = center
        self._original_name = diffusion_array.meta.name

        if profile_array is None:
            profile_array = RadialTimeProfile._create_data_parallel(diffusion_array, center)

        if profile_array.ndim != 2:
            raise ValueError('profile_array must be 2D')

        if profile_array.shape[0] != diffusion_array.shape[0]:
            raise ValueError('profile_array must have the same temporal shape as the diffusion_array')

        self._ndarray = profile_array

    @property
    def ndarray(self) -> np.ndarray:
        return self._ndarray

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.ndarray.shape

    @property
    def number_of_frames(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def ndim(self) -> int:
        return self.ndarray.ndim

    @property
    def original_width(self) -> int:
        return self._original_width

    @property
    def original_height(self) -> int:
        return self._original_height

    @property
    def original_name(self) -> str:
        return self._original_name

    @property
    def original_center(self) -> Tuple[int | float, int | float]:
        return self._original_center

    def __iter__(self) -> Iterator:
        return iter(self.ndarray)

    def __array__(self):
        return self.ndarray

    def frame(self, frame: int | str) -> np.ndarray:
        """
        Get a specific frame from the radial time profile data.

        Args:
            frame (int | str): The frame to retrieve. It can be an integer representing the frame index,
                or a string in the format "start:stop:step" to specify a range.

        Returns:
            np.ndarray: The requested frame data.

        Example:
            frame_1 = radial_profile.frame(1)
            frame_slice = radial_profile.frame("5:10:2")
        """
        if isinstance(frame, str):
            frame = slice(*([int(x) for x in frame.split(':')]))
        return self.ndarray[frame, :]

    def resized(self, new_radius: int, degree: int = 1) -> 'RadialTimeProfile':
        """
        Resizes the radial time profile objet to the new radius. Uses polynomial spline interpolation of degree to
        estimate the values at the new grid points. Returns a new object, doesn't modify the original.

        Args:
            new_radius (int): the new size of the underlying np.ndarray.
            degree (int): the degree of the polynomial used in interpolation.

        Returns:
            a new 'RadialTimeProfile' object constructed by resampling the old one.
        """
        if new_radius > self.ndarray.shape[1]:
            warnings.warn('You are over-increasing the size of the radius', category=Warning)

        new_profile_array = np.zeros((self.ndarray.shape[0], new_radius))
        y_old = np.hstack((self.ndarray, self.ndarray[:, -1][:, np.newaxis]))
        x_range_old = np.arange(y_old.shape[1])
        x_range_new = np.linspace(0, self.ndarray.shape[1], new_radius)

        for i in range(self.ndarray.shape[0]):
            spline = scipy.interpolate.make_interp_spline(
                x=x_range_old,
                y=y_old[i, :],
                k=degree,
                bc_type='natural' if degree == 3 else None
            )
            new_profile_array[i, :] = spline(x_range_new)

        return RadialTimeProfile(
            DiffusionArray.empty(self.ndarray.shape[0], self.original_width, self.original_height, self.original_name),
            self.original_center,
            profile_array=new_profile_array
        )

    def to_diffusion_array(self, degree: int = 3) -> DiffusionArray:
        """
        Creates a diffusion array from the radial time profile. Uses polynomial spline interpolation.

        Args:
            degree (int):  the degree of the polynomial used in interpolation.

        Returns:
            a new DiffusionArray object created from the radial time profile.
        """
        values_by_distance = np.hstack((self.ndarray, self.ndarray[:, -1, np.newaxis]))
        x_range = np.arange(values_by_distance.shape[1])

        width = self.original_width
        height = self.original_height
        center_x, center_y = tuple(map(round, self.original_center))

        darr_data = np.zeros((self.ndarray.shape[0], width, height))

        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        xs -= center_x
        ys -= center_y
        # distances = np.vectorize(np.floor)(np.hypot(xs, ys))
        distances = np.hypot(xs, ys)

        for i in range(self.ndarray.shape[0]):
            spline = scipy.interpolate.make_interp_spline(
                x=np.hstack([x_range, x_range[-1] * 20]),
                y=np.hstack([values_by_distance[i, :], values_by_distance[i, -1] * 20]),
                k=degree,
                bc_type='natural' if degree == 3 else None
            )
            darr_data[i, :] = spline(distances.flatten()).reshape(darr_data.shape[1:])

        return DiffusionArray(path=None, ndarray=darr_data)
