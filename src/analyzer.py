import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from diffusion_array import DiffusionArray
from src.mask import Mask


class Analyzer:
    def __init__(self, diffusion_array: DiffusionArray):
        if diffusion_array.ndim != 3:
            raise ValueError(f'diffusion_array must be 3-dimensional, but it was {diffusion_array.ndim}')
        self._diffusion_array = diffusion_array

    @property
    def diffusion_array(self) -> DiffusionArray:
        return self._diffusion_array

    def detect_diffusion_start_frame(self) -> int:
        """
        Detects the start of the diffusion process based on the maximum differences between 2 consecutive frames.
        Assumes that the process never starts in the first 2 frames

        Returns:
            int: The frame index where the diffusion process starts.
        """

        if self.diffusion_array.get_cached('diffusion_start_frame') is not None:
            return self.diffusion_array.get_cached('diffusion_start_frame')

        arr = np.array(self.diffusion_array)
        arr = arr - gaussian_filter(np.mean(self.diffusion_array.frame('0:3'), axis=0), sigma=2)

        arr = np.diff(np.max(arr, axis=(1, 2)))

        ans = int(np.argmax(arr))
        self.diffusion_array.cache(diffusion_start_frame=ans)
        return ans

    def detect_diffusion_start_place(self, strategy: str = 'connected-components', **kwargs) -> tuple:
        """
        Detects the place where the diffusion process starts based on the maximum differences between 2 consecutive
        frames. Assumes that the process never starts in the first 2 frames.

        Args:
            strategy (str): The strategy used for detecting the diffusion start place. Options are 'connected-components',
                            'weighted-centroid', and 'biggest-difference'. Default is 'connected-components'.
            **kwargs: Additional keyword arguments.
                use_inner (bool): Specifies whether to use the inner region for the 'connected-components' strategy.
                                  Default is False.

        Returns:
            tuple: The coordinates (row, column) of the place where the diffusion process starts.

        Raises:
            ValueError: If the strategy is not one of 'connected-components', 'weighted-centroid', or 'biggest-difference'.
        """

        use_inner = kwargs.get('use_inner', False)

        if self.diffusion_array.get_cached(f'diffusion_start_place {strategy}') is not None and not use_inner:
            return self.diffusion_array.get_cached(f'diffusion_start_place {strategy}')

        if self.diffusion_array.get_cached(f'diffusion_start_place {strategy}_use_inner') is not None and use_inner:
            return self.diffusion_array.get_cached(f'diffusion_start_place {strategy}_use_inner')

        def save_and_return(place, strategy=strategy):
            kwargs_dict = {f'diffusion_start_place {strategy}': place}
            self.diffusion_array.cache(**kwargs_dict)
            return place

        start_frame_number = self.detect_diffusion_start_frame()
        frame = self.diffusion_array.frame(start_frame_number + 1)
        frame = frame - np.mean(self.diffusion_array.frame(slice(start_frame_number - 1, start_frame_number + 1)),
                                axis=0)

        place = np.unravel_index(np.argmax(np.abs(frame)), frame.shape)

        if strategy == 'biggest-difference':
            return save_and_return((place[1], place[0]))

        if strategy == 'weighted-centroid':
            frame[frame < 0] = 0
            radius = int(((frame.shape[0] * frame.shape[1]) ** (1 / 2)) / 3)
            frame[Mask.circle(frame.shape, place, radius).flip().ndarray] = 0

            # centroid
            total_intensity = np.sum(frame, dtype=np.int64)
            rows, cols = np.indices(frame.shape)
            weighted_rows = np.sum(frame * rows, dtype=np.int64) / total_intensity
            weighted_cols = np.sum(frame * cols, dtype=np.int64) / total_intensity

            place = (round(weighted_rows), round(weighted_cols))
            return save_and_return(place)

        if strategy == 'connected-components':
            med = np.percentile(frame.flatten(), 67)
            frame_copy = frame.copy()

            frame[frame < med] = med - 1
            frame[frame >= med] = 255
            frame[frame == med - 1] = 0

            image_uint8 = frame.astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            image_uint8 = cv2.morphologyEx(image_uint8, cv2.MORPH_OPEN, kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_uint8, 4)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            center_x, center_y = centroids[largest_label]

            if not use_inner:
                return save_and_return((center_x, center_y))

            med = np.percentile(frame_copy.flatten(), 45)
            frame_copy[frame_copy < med] = med - 1
            frame_copy[frame_copy >= med] = 255
            frame_copy[frame_copy == med - 1] = 0
            image_uint8 = frame_copy.astype(np.uint8)
            image_uint8 = cv2.morphologyEx(image_uint8, cv2.MORPH_OPEN, kernel)
            image_uint8 = 255 - image_uint8
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_uint8, 4)
            distances = np.sqrt((centroids[:, 0] - center_x) ** 2 + (centroids[:, 1] - center_y) ** 2)
            closest_label = np.argmin(distances)
            closest_x, closest_y = centroids[closest_label]

            strategy = strategy + '_use_inner'
            return save_and_return((closest_x, closest_y), strategy=strategy)

        raise ValueError(f"Unknown strategy ({strategy}). It must be either 'connected-components', 'weighted-centroid'"
                         f" or 'biggest-difference'.")

    def apply_for_each_frame(self, function: callable,
                             remove_background: bool = False,
                             normalize: bool = False,
                             use_gaussian_blur: bool = False,
                             mask: Mask | None = None) -> np.ndarray:
        """
        Applies a function to each frame of the diffusion array. The function must take a frame (2D ndarray) as it's
        parameter and return a single number. After applying the function it collects the outputs in an array.

        Args:
            function (callable): The function to apply to each frame.
            remove_background (bool, optional): If True, removes the background by subtracting the average of the first three frames. Default is False.
            normalize (bool, optional): If True, normalizes the result to the range [0, 1]. Default is False.
            use_gaussian_blur (bool, optional): If True, will use gaussian blur for the background removal.
            If remove background is False gaussian blur will not be used. Default is False.
            mask (np.ndarray | None, optional): A mask to apply to the diffusion array. Default is None.

        Returns:
            np.ndarray: The result of applying the function to each frame.

        """
        arr = self.diffusion_array
        if remove_background:
            mean = np.mean(self.diffusion_array.frame('0:3'), axis=0)
            if use_gaussian_blur:
                difference = arr - gaussian_filter(mean, sigma=2)
            else:
                difference = arr - mean
            arr = arr.update_ndarray(difference)

        if mask is not None:
            applied = function(arr[mask], axis=1)
        else:
            applied = function(arr, axis=(1, 2))

        if not normalize:
            return applied

        applied = applied.astype(np.float64)
        max_val = np.max(applied)
        min_val = np.min(applied)
        normalized_array = (applied - min_val) / (max_val - min_val)
        return normalized_array
