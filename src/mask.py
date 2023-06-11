import numpy as np


class Mask:
    """A class representing a mask for a diffusion_array.

    Args:
        ndarray (np.ndarray): The underlying ndarray, could be static (the same for each frame, 2D) or dynamic
        (different for each frame, 3D)

    Methods:
        for_frame(frame: int) -> np.ndarray:
            Returns the mask for a specific frame of the diffusion.

        flip() -> Mask:
            Returns a new Mask object with the logical complement of the current mask.

    Static Methods:
        circle(shape: tuple, center: tuple, radius: int | float) -> Mask:
            Creates a circular mask with the specified shape, center, and radius.

        ring(shape: tuple, center: tuple, inner_radius: int | float, outer_radius: int | float) -> Mask:
            Creates a ring-shaped mask with the specified shape, center, inner radius, and outer radius.

        bottom_right_quarter(shape: tuple, point: tuple) -> Mask:
            Creates a mask representing the bottom-right quarter of the specified shape starting from the given point.
    """

    def __init__(self, ndarray: np.ndarray):
        if ndarray.ndim not in (2, 3):
            raise ValueError(f'ndarray must be 2 or 3 dimensional, but it was: {ndarray.ndim}')
        self._ndarray = ndarray

    @property
    def ndarray(self) -> np.ndarray:
        return self._ndarray

    def for_frame(self, frame: int) -> np.ndarray:
        """Returns the mask for a specific frame of a diffusion_array.

        Args:
            frame (int): The index of the frame.

        Returns:
            np.ndarray: The mask for the specified index if the input ndarray is 3D,
                        or the mask itself if the input ndarray is 2D.
        """

        if self.ndarray.ndim == 2:
            return self.ndarray

        return self.ndarray[frame, ...]

    def flip(self):
        """Returns a new Mask object with the logical complement of the current mask.

        Returns:
            Mask: A new Mask object with the logical complement of the current mask.
        """
        return Mask(np.logical_not(self.ndarray))

    @staticmethod
    def circle(shape: tuple, center: tuple, radius: int | float) -> 'Mask':
        """Creates a circular mask with the specified shape, center, and radius.

        Args:
            shape (tuple): The shape of the mask in (rows, columns) format.
            center (tuple): The center coordinates of the circle in (y, x) format.
            radius (int | float): The radius of the circle, inclusive.

        Returns:
            Mask: A new Mask object representing the circular mask.
        """

        shape = shape[-2:]
        rows, cols = shape
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

        distances_sq = (x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2

        ndarray = np.zeros(shape=shape, dtype=bool)
        ndarray[distances_sq <= radius ** 2] = True

        return Mask(ndarray)

    @staticmethod
    def ring(shape: tuple, center: tuple, inner_radius: int | float, outer_radius: int | float) -> 'Mask':
        """Creates a ring-shaped mask with the specified shape, center, inner radius, and outer radius.

        Args:
            shape (tuple): The shape of the mask in (rows, columns) format.
            center (tuple): The center coordinates of the ring in (y, x) format.
            inner_radius (int | float): The inner radius of the ring, inclusive.
            outer_radius (int | float): The outer radius of the ring, exclusive.

        Returns:
            Mask: A new Mask object representing the ring-shaped mask.
        """

        shape = shape[-2:]
        rows, cols = shape
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

        distances_sq = (x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2
        ring = np.logical_and(distances_sq >= inner_radius ** 2, distances_sq < outer_radius ** 2)

        ndarray = np.zeros(shape=shape, dtype=bool)
        ndarray[np.where(ring)] = True

        return Mask(ndarray)

    @staticmethod
    def bottom_right_quarter(shape: tuple, point: tuple) -> 'Mask':
        """Creates a mask for the bottom-right quarter of the specified shape starting from the given point.

        Args:
            shape (tuple): The shape of the mask in (rows, columns) format.
            point (tuple): The starting coordinates of the bottom-right quarter in (y, x) format.

        Returns:
            Mask: A new Mask object representing the bottom-right quarter of the specified shape.
        """

        shape = shape[-2:]
        x, y = point
        ndarray = np.zeros(shape=shape, dtype=bool)
        ndarray[x:, y:] = True

        return Mask(ndarray)
