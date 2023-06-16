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

    @staticmethod
    def cell(diffusion_array: np.ndarray, center: tuple, radius: int | float,
             cutoff_by_frame: np.ndarray) -> 'Mask':
        """
        Generates a mask based on a circular mask and cutoff values for each frame. The mask tries to find the cells in
        the given circular mask, in order to do so it uses a different cutoff value for each frame
        Args:
            diffusion_array (np.ndarray): the DiffusionArray as a np.ndarray
            radius (int | float): The radius, giving bounds to the mask
            center (tuple): The center around which cells should be detected
            cutoff_by_frame (np.ndarray): The cutoff values for each frame.
        Returns:
            np.ndarray: The cell mask as a boolean array.
        """
        if diffusion_array.ndim != 3:
            raise ValueError(f'diffusion_array must be 3D, but was: {diffusion_array.ndim}')

        if cutoff_by_frame.ndim != 1:
            raise ValueError(f'cutoff_by_frame must be 1D, but was: {cutoff_by_frame.ndim}')

        if diffusion_array.shape[0] != len(cutoff_by_frame):
            raise ValueError(f'diffusion_array and cutoff_by_frame have the same number of frames')

        mask_cir = np.zeros(shape=diffusion_array.shape)
        circle_mask = Mask.circle(diffusion_array.shape, center, radius).ndarray
        print(mask_cir.shape, circle_mask.shape)
        mask_cir[:, circle_mask] = 1

        mask_cut = np.zeros(shape=diffusion_array.shape)
        mask_cut[diffusion_array >= cutoff_by_frame[:, np.newaxis, np.newaxis]] = 1

        mask = mask_cut * mask_cir
        return Mask(mask.astype(bool))

    @staticmethod
    def star(diffusion_array: np.ndarray, cutoff_extractor: callable = lambda x: np.percentile(x, 95)) -> 'Mask':
        """
        Create a mask by thresholding a diffusion array based on a cutoff value. It's useful for identifying the bright
        values at the start of the diffusion process.

        The diffusion_array must be a 2D NumPy array. The method calculates a cutoff value using the provided
        cutoff_extractor function (default: np.percentile(diffusion_array, 95)), and creates a binary mask by setting
        elements of diffusion_array above the cutoff value to 1 and the rest to 0.

        Parameters:
            diffusion_array (np.ndarray): The 2D array representing the diffusion data.
            cutoff_extractor (callable): A function to extract the cutoff value from the diffusion array
                (default: np.percentile(diffusion_array, 95)).

        Returns:
            Mask: A Mask object representing the binary mask.
        """
        if diffusion_array.ndim != 2:
            raise ValueError(f'diffusion_array must be 2D, but was: {diffusion_array.ndim}')

        mask = np.zeros(shape=diffusion_array.shape)
        mask[diffusion_array > cutoff_extractor(diffusion_array)] = 1
        return Mask(mask.astype(bool))
