from typing import Tuple

import numpy as np


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
    def bottom_right_quarter(
            shape: Tuple[int, ...] | np.ndarray,
            point: tuple
    ) -> 'Mask':
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
    def threshold_percentile_low(
            diffusion_array: np.ndarray,
            percentile: float,
            by_frame: bool = True
    ) -> 'Mask':
        """
        Creates a Mask by thresholding the diffusion array with a cutoff value calculated as the given percentile of the
        entire array or just one frame of it depending on the `by_frame` flag. Values smaller than the cutoff value will
        become true.

        Args:
            diffusion_array (np.ndarray): A 3D array representing the diffusion data.
            percentile (float): The value used as q in the percentile calculations. Must be: 0 <= percentile <= 100.
            by_frame (bool): If true, calculates the threshold independently for each frame, uses the same threshold for
                each frame otherwise.

        Returns:
            Mask: The mask obtained form: diffusion_array <= cutoff_value
        """
        if diffusion_array.ndim != 3:
            raise ValueError(f'`diffusion_array` should be 3D, but it was: {diffusion_array.ndim}')
        if not 0 <= percentile <= 100:
            raise ValueError(f'`percentile` should be between 0 and 100, but it was: {percentile:.2f}')

        if by_frame:
            cutoff_by_frame = np.percentile(diffusion_array, q=percentile, axis=(1, 2))
            masks = [subarray <= cutoff for subarray, cutoff in zip(diffusion_array, cutoff_by_frame)]
            mask = np.array(masks, dtype=bool)
            return Mask(mask)

        cutoff = np.percentile(diffusion_array, q=percentile)
        mask = np.array(diffusion_array) > cutoff
        return Mask(mask)

    @staticmethod
    def threshold_percentile_high(
            diffusion_array: np.ndarray,
            percentile: float,
            by_frame: bool = True
    ) -> 'Mask':
        """
        Creates a Mask by thresholding the diffusion array with a cutoff value calculated as the given percentile of the
        entire array or just one frame of it depending on the `by_frame flag`. Values greater than the cutoff value will
        become true.

        Args:
            diffusion_array (np.ndarray): A 3D array representing the diffusion data.
            percentile (float): The value used as q in the percentile calculations. Must be: 0 <= percentile <= 100.
            by_frame (bool): If true, calculates the threshold independently for each frame, uses the same threshold for
                each frame otherwise.

        Returns:
            Mask: The mask obtained form: diffusion_array > cutoff_value
        """
        return Mask.threshold_percentile_low(diffusion_array, percentile, by_frame).flip()

    @staticmethod
    def range_threshold_percentile(
            diffusion_array: np.ndarray,
            percentile_low: float,
            percentile_high: float,
            by_frame: bool = True
    ) -> 'Mask':
        """
        Creates a Mask by thresholding the diffusion array with a cutoff values calculated as the given percentiles of
        the entire array or just one frame of it depending on the `by_frame flag`. Values between the cutoff_low and
        cutoff_high will be true.

        Args:
            diffusion_array (np.ndarray): A 3D array representing the diffusion data.
            percentile_low (float): The value used as q in the percentile calculations for cutoff_low.
                Must be: 0 <= percentile <= 100.
            percentile_high (float): The value used as q in the percentile calculations for cutoff_high.
                Must be: 0 <= percentile <= 100.
            by_frame (bool): If true, calculates the threshold independently for each frame, uses the same threshold for
                each frame otherwise.

        Returns:
            Mask: The mask obtained form: cutoff_low < diffusion_array <= cutoff_high.
        """
        return (
                Mask.threshold_percentile_high(diffusion_array, percentile_low, by_frame) &
                Mask.threshold_percentile_low(diffusion_array, percentile_high, by_frame)
        )

    @staticmethod
    def ones(
            shape: Tuple[int, ...],
    ) -> 'Mask':
        """
        Creates a musk in a given shape with only True values.

        Args:
            shape (Tuple[int, ...]): The shape of the new mask.

        Returns:
            Mask: A mask with only True values as entries.
        """
        if len(shape) not in (2, 3):
            raise ValueError(f'`shape` must be 2 or 3D, but it was: {len(shape)}')

        mask = np.ones(shape, dtype=bool)
        return Mask(mask)

    @staticmethod
    def zeros(
            shape: Tuple[int, ...],
    ) -> 'Mask':
        """
        Creates a musk in a given shape with only False values.

        Args:
            shape (Tuple[int, ...]): The shape of the new mask.

        Returns:
            Mask: A mask with only False values as entries.
        """
        if len(shape) not in (2, 3):
            raise ValueError(f'`shape` must be 2 or 3D, but it was: {len(shape)}')

        mask = np.zeros(shape, dtype=bool)
        return Mask(mask)
