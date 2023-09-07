import numpy as np
from typing import Tuple


class Mask:
    """
    A class representing a mask for a diffusion_array.

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
        """
        Returns the mask for a specific frame of a diffusion_array.

        Args:
            frame (int): The index of the frame.

        Returns:
            np.ndarray: The mask for the specified index if the input np.ndarray is 3D, or the mask itself if the input
            np.ndarray is 2D.
        """

        if self.ndarray.ndim == 2:
            return self.ndarray

        return self.ndarray[frame, ...]

    def flip(self):
        """
        Returns a new Mask object with the logical complement of the current mask.

        Returns:
            Mask: A new Mask object with the logical complement of the current mask.
        """
        return Mask(np.logical_not(self.ndarray))

    def __and__(self, other) -> 'Mask':
        """ Returns a new Mask object created by combining 2 Masks by the logical and operator. If the other mask is
        none returns self.
        Args:
            other (Optional[np.ndarray | 'Mask']): The other mask

        Returns:
            Mask: The mask created by combining self with the other Mask provided by applying conjugation
        """

        if other is None:
            return self

        if isinstance(other, np.ndarray):
            return Mask(np.logical_and(self.ndarray, other))

        return Mask(np.logical_and(self.ndarray, other.ndarray))

    def __or__(self, other) -> 'Mask':
        """
        Returns a new Mask object created by combining 2 Masks by the logical or operator.  If the other mask is
        none returns self.

        Args:
            other (Optional[np.ndarray | 'Mask']): The other mask

        Returns:
            Mask: The mask created by combining self with the other Mask provided by applying disjunction
        """

        if other is None:
            return self

        if isinstance(other, np.ndarray):
            return Mask(np.logical_or(self.ndarray, other))
        return Mask(np.logical_or(self.ndarray, other.ndarray))

    @staticmethod
    def circle(
            shape: Tuple[int, ...] | np.ndarray,
            center: Tuple[float, ...],
            radius: int | float | Tuple[int | float, ...]
    ) -> 'Mask':
        """
        Creates a circular mask with the specified shape, center, and radius.

        Shape should be a tuple of ints specifying the shape of the mask created. If it's a np.ndarray or a
        DiffusionArray its shape will be used.

        Radius could be a scalar or a Tuple of ints or floats. If a tuple is provided a dynamic (3D) mask will be
        created each value acting as an inclusive radius for a frame.

        Args:
            shape (tuple | ndarray): The shape of the mask in (rows, columns) format.
            center (Tuple[float, ...]): The center coordinates of the circle in (y, x) format.
            radius (int | float | Tuple[int | float, ...]): The radius of the circle, inclusive.

        Returns:
            Mask: A new Mask object representing the circular mask.
        """

        if isinstance(shape, np.ndarray):
            shape = shape.shape

        shape = shape[-2:]
        rows, cols = shape
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

        distances_sq = (x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2

        if isinstance(radius, (int, float)):
            ndarray = np.zeros(shape=shape, dtype=bool)
            ndarray[distances_sq <= radius ** 2] = True

            return Mask(ndarray)

        circle_masks = []
        for r in radius:
            circle_masks.append(distances_sq <= r ** 2)

        return Mask(np.ndarray(circle_masks))

    @staticmethod
    def ring(
            shape: Tuple[int, ...] | np.ndarray,
            center: Tuple[float, ...],
            inner_radius: int | float | Tuple[int | float, ...],
            outer_radius: int | float | Tuple[int | float, ...],
    ) -> 'Mask':
        """
        Creates a ring-shaped mask with the specified shape, center, inner radius, and outer radius.

        Shape should be a tuple of ints specifying the shape of the mask created. If it's a np.ndarray or a
        DiffusionArray its shape will be used.

        Radius could be a scalar or a Tuple of ints or floats. If a tuple is provided a dynamic (3D) mask will be
        created each value acting as an inner radius or an exclusive outer radius for a frame.

        Inner and outer radius must have the same length if they are both tuples.

        Args:
            shape (Tuple[int, ...] | np.ndarray): The shape of the mask in (rows, columns) format.
            center (Tuple[float, ...]): The center coordinates of the ring in (x, y) format.
            inner_radius (int | float | Tuple[int | float, ...]): The inner radius of the ring, inclusive.
            outer_radius (int | float | Tuple[int | float, ...]): The outer radius of the ring, exclusive.

        Returns:
            Mask: A new Mask object representing the ring-shaped mask.
        """

        if not isinstance(shape, tuple):
            shape = shape.shape

        if (isinstance(inner_radius, tuple) and isinstance(outer_radius, tuple)
                and len(inner_radius) != len(outer_radius)):
            raise ValueError(f'Both inner and outer radii are a tuple but they have different lengths ('
                             f'{len(inner_radius)} and {len(outer_radius)} respectively).')

        shape = shape[-2:]
        rows, cols = shape
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

        distances_sq = (x_indices - center[0]) ** 2 + (y_indices - center[1]) ** 2

        if isinstance(inner_radius, (int, float)) and isinstance(outer_radius, (int, float)):
            ring = np.logical_and(distances_sq >= inner_radius ** 2, distances_sq < outer_radius ** 2)

            ndarray = np.zeros(shape=shape, dtype=bool)
            ndarray[np.where(ring)] = True

            return Mask(ndarray)

        if isinstance(inner_radius, (int, float)):
            inner_radius = tuple(inner_radius for _ in outer_radius)

        if isinstance(outer_radius, (int, float)):
            outer_radius = tuple(outer_radius for _ in inner_radius)

        ring_masks = []
        for inner_r, outer_r in zip(inner_radius, outer_radius):
            ring = np.logical_and(distances_sq >= inner_r ** 2, distances_sq < outer_r ** 2)
            ring_masks.append(ring)
        return Mask(np.ndarray(ring_masks))

    @staticmethod
    def bottom_right_quarter(shape: Tuple[int, ...] | np.ndarray, point: tuple) -> 'Mask':
        """
        Creates a mask for the bottom-right quarter of the specified shape starting from the given point.

        Shape should be a tuple of ints specifying the shape of the mask created. If it's a np.ndarray or a
        DiffusionArray its shape will be used

        Args:
            shape (Tuple[int, ...] | np.ndarray): The shape of the mask in (rows, columns) format.
            point (tuple): The starting coordinates of the bottom-right quarter in (y, x) format.

        Returns:
            Mask: A new Mask object representing the bottom-right quarter of the specified shape.
        """

        if isinstance(shape, np.ndarray):
            shape = shape.shape

        shape = shape[-2:]
        x, y = point
        ndarray = np.zeros(shape=shape, dtype=bool)
        ndarray[x:, y:] = True

        return Mask(ndarray)

    @staticmethod
    def cutoff(diffusion_array: np.ndarray, cutoff_by_frame: np.ndarray | float | int) -> 'Mask':
        """
        Generates a mask by thresholding the diffusion array based on cutoff values for each frame. The mask is useful
        for findig cells or isolating very bright 'star-like' points. Intensities greater or equal to the cutoff value
        will be set to 1 (or true) otherwise they will be set to 0 (false).

        The diffusion array must be a 2D or 3D NumPy array or a DiffusionArray with the appropriate index strategy.

        cutoff_by_frame should be a 1D NumPy array whose length must be equal to diffusion_array.shape[0]. It should
        contain a cutoff value for each frame. If only a scalar value is provided a new NumPy array will be crated with
        the appropriate shape and be filled with the value.

        Args:
            diffusion_array (np.ndarray | DiffusionArray): the DiffusionArray as a np.ndarray
            cutoff_by_frame (np.ndarray | float | int): The cutoff values for each frame.
        Returns:
            np.ndarray: The cell mask as a boolean array.
        """

        if isinstance(cutoff_by_frame, (int, float)):
            cutoff_by_frame = np.full(diffusion_array.shape[0], cutoff_by_frame)

        if diffusion_array.ndim not in (2, 3):
            raise ValueError(f'diffusion_array must be 2D or 3D, but was: {diffusion_array.ndim}')

        if cutoff_by_frame.ndim != 1:
            raise ValueError(f'cutoff_by_frame must be 1D, but was: {cutoff_by_frame.ndim}')

        if diffusion_array.shape[0] != len(cutoff_by_frame):
            raise ValueError(f'diffusion_array and cutoff_by_frame have the same number of frames')

        # if isinstance(diffusion_array, DiffusionArray):
        #     diffusion_array = diffusion_array[:]

        mask = np.zeros(shape=diffusion_array.shape)
        if diffusion_array.ndim == 3:
            mask[diffusion_array >= cutoff_by_frame[:, np.newaxis, np.newaxis]] = 1
        elif diffusion_array.ndim == 2:
            mask[diffusion_array >= cutoff_by_frame[0]] = 1

        return Mask(mask.astype(bool))
