import numpy as np
from scipy.ndimage import gaussian_filter

from diffusion_array import DiffusionArray


class Analyzer:
    def __init__(self, diffusion_array: DiffusionArray):
        if diffusion_array.ndim != 3:
            raise ValueError(f'diffusion_array must be 3-dimensional, but it was {diffusion_array.ndim}')
        self._diffusion_array = diffusion_array

    @property
    def diffusion_array(self) -> DiffusionArray:
        return self._diffusion_array

    @diffusion_array.setter
    def diffusion_array(self, diffusion_array: DiffusionArray):
        self._diffusion_array = diffusion_array

    def detect_diffusion_start_frame(self) -> int:
        """
        Detects the start of the diffusion process based on the maximum differences between 2 consecutive frames.
        Assumes that the process never starts in the first 2 frames

        Returns:
            int: The frame index where the diffusion process starts.
        """
        arr = np.array(self.diffusion_array)
        arr = arr - gaussian_filter(np.mean(self.diffusion_array.frame('0:3'), axis=0), sigma=2)

        arr = np.diff(np.max(arr, axis=(1, 2)))

        return int(np.argmax(arr))

    def detect_diffusion_start_place(self) -> tuple:
        """
        Detects the place where the diffusion process starts based on the maximum differences between 2 consecutive frames.
        Assumes that the process never starts in the first 2 frames

        Returns:
            tuple: The coordinates (row, column) of the place where the diffusion process starts.
        """
        start_frame_number = self.detect_diffusion_start_frame()
        frame = self.diffusion_array.frame(start_frame_number + 1)
        frame = frame - np.mean(self.diffusion_array.frame(slice(start_frame_number - 1, start_frame_number + 1)),
                                axis=0)

        place = np.unravel_index(np.argmax(frame), frame.shape)
        frame[frame < 0] = 0
        radius = int(((frame.shape[0] * frame.shape[1]) ** (1 / 2)) / 3)
        frame[self.circle_mask(place, radius, outside=True)] = 0

        # centroid
        total_intensity = np.sum(frame)
        rows, cols = np.indices(frame.shape)
        weighted_rows = np.sum(frame * rows) / total_intensity
        weighted_cols = np.sum(frame * cols) / total_intensity
        place = (round(weighted_rows), round(weighted_cols))

        return place

    def circle_mask(self, point: tuple, radius: int | float, outside: bool = False) -> np.ndarray:
        """
        Generates a circular mask around a given point with a specified radius.

        Args:
            point (tuple): The coordinates (row, column) of the center of the circle.
            radius (int | float): The radius of the circle.
            outside (bool, optional): If True, generates a mask for the area outside the circle. Default is False.

        Returns:
            np.ndarray: The circular mask as a boolean array.
        """
        rows, cols = self.diffusion_array.frame(0).shape
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

        distances_sq = (x_indices - point[1]) ** 2 + (y_indices - point[0]) ** 2

        if outside:
            return np.where(distances_sq > radius ** 2)

        return np.where(distances_sq <= radius ** 2)

    def cell_mask(self, circle_mask: np.ndarray, cutoff_by_frame: np.ndarray) -> np.ndarray:
        """
        Generates a mask based on a circular mask and cutoff values for each frame. The mask tries to find the cells in
        the given circular mask, in order to do so it uses a different cutoff value for each frame

        Args:
            circle_mask (np.ndarray): The circular mask.
            cutoff_by_frame (np.ndarray): The cutoff values for each frame.

        Returns:
            np.ndarray: The cell mask as a boolean array.
        """
        mask_cir = np.zeros(shape=self.diffusion_array.shape)
        mask_cir[:, circle_mask[0], circle_mask[1]] = 1

        mask_cut = np.zeros(shape=self.diffusion_array.shape)
        mask_cut[self.diffusion_array >= cutoff_by_frame[:, np.newaxis, np.newaxis]] = 1

        mask = mask_cut * mask_cir
        return mask.astype(bool)

    def apply_for_each_frame(self, function: callable,
                             remove_background: bool = False,
                             normalize: bool = False,
                             mask: np.ndarray | None = None) -> np.ndarray:
        """
        Applies a function to each frame of the diffusion array. The function must take a frame (2D ndarray) as it's
        parameter and return a single number. After applying the function it collects the outputs in an array.

        Args:
            function (callable): The function to apply to each frame.
            remove_background (bool, optional): If True, removes the background by subtracting the average of the first three frames. Default is False.
            normalize (bool, optional): If True, normalizes the result to the range [0, 1]. Default is False.
            mask (np.ndarray | None, optional): A mask to apply to the diffusion array. Default is None.

        Returns:
            np.ndarray: The result of applying the function to each frame.

        """
        arr = np.array(self.diffusion_array)

        if remove_background:
            arr = arr - gaussian_filter(np.mean(self.diffusion_array.frame('0:3'), axis=0), sigma=2)

        if mask is not None:
            if isinstance(mask, tuple):
                arr = arr[:, mask[0], mask[1]]
                applied = function(arr, axis=1)
            else:
                # TODO
                mask = np.where(mask)
                arr = arr[:, mask[1], mask[2]]
                applied = function(arr, axis=1)
        else:
            applied = function(arr, axis=(1, 2))

        if not normalize:
            return applied

        applied = applied.astype(np.float64)
        max_val = np.max(applied)
        min_val = np.min(applied)
        normalized_array = (applied - min_val) / (max_val - min_val)
        return normalized_array
