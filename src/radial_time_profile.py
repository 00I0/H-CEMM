import numpy as np
from typing import Tuple, Iterator

from src.analyzer import Analyzer
from src.diffusion_array import DiffusionArray


class RadialTimeProfile:
    """
    Represents a radial time profile of the *homogenized* diffusion data. That is a 2D array of the intensities indexed
    by time/frame and distance form the center point.

    Attributes:
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

    def __init__(self, diffusion_array: DiffusionArray, center: Tuple[int | float, int | float] = None):
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

        self._ndarray = RadialTimeProfile._create_data_parallel(diffusion_array, center)

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
