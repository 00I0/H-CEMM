import numpy as np
import skimage
from scipy import ndimage
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

    def detect_diffusion_start_frame(self, strategy: str = 'adaptive') -> int:
        """
        Detects the start of the diffusion process based on the maximum differences between 2 consecutive frames.
        Assumes that the process never starts in the first 2 frames

        Args:
            strategy (str): The strategy used for detecting the diffusion start frame.
                Options are: 'adaptive' (the default) and 'max-derivative'.

        Returns:
            int: The frame index where the diffusion process starts.
        """

        ans = -1

        if strategy not in ('adaptive', 'max-derivative'):
            raise ValueError(f'Unknown strategy: {strategy}')

        if self.diffusion_array.get_cached(f'diffusion_start_frame {strategy}') is not None:
            return self.diffusion_array.get_cached(f'diffusion_start_frame {strategy}')

        if strategy == 'adaptive':
            arr = np.array(self.diffusion_array.normalized())
            prev_diff = np.sum((arr[3, ...] - np.mean(arr[0:3, ...], axis=0)) ** 2)

            i = 3
            for i in range(3, arr.shape[0]):
                diff = np.sum((arr[i, ...] - np.mean(arr[i - 3:i, ...], axis=0)) ** 2)
                if diff > 1.15 * prev_diff:
                    break
                prev_diff = diff

            ans = i

        if strategy == 'max-derivative':
            arr = np.array(self.diffusion_array)
            arr = arr - gaussian_filter(np.mean(self.diffusion_array.frame('0:3'), axis=0), sigma=2)
            arr = np.diff(np.max(arr, axis=(1, 2)))
            ans = round(np.argmax(arr))

        self.diffusion_array.cache(**{f'diffusion_start_frame {strategy}': ans})
        return ans

    def detect_diffusion_start_place(self, strategy: str = 'connected-components', **kwargs) -> tuple:
        r"""
        Detects the place where the diffusion process starts. You can specify different strategies for the detection.
        The returned indices are in the form of (x, y).

        Args:
            strategy (str): The strategy used for detecting the diffusion start place. Options are:

                *  'connected-components' (default): After thresholding finds the largest connected component end and
                returns its center. An additional kwarg (use_inner) could specify whether to use the centroid of the
                largest connected component or the centroid of the largest connected component within the largest
                component.

                *  'weighted-centroid': Calculates the average of the (x, y) indices weighted by their intensity.

                *  'biggest-difference': Returns the indices of the biggest difference between the start-frame and
                the previous frame.

            **kwargs: Additional keyword arguments.
                use_inner (bool): Specifies whether to use the inner region for the 'connected-components' strategy.
                                  Default is False.

        Returns:
            tuple: The indices (x, y) of the place where the diffusion process starts.

        Raises: ValueError: If the strategy is not one of 'connected-components', 'weighted-centroid',
            or 'biggest-difference'.
        """

        use_inner = kwargs.get('use_inner', False)

        if self.diffusion_array.get_cached(f'diffusion_start_place {strategy}') is not None and not use_inner:
            return self.diffusion_array.get_cached(f'diffusion_start_place {strategy}')

        if self.diffusion_array.get_cached(f'diffusion_start_place {strategy}_use_inner') is not None and use_inner:
            return self.diffusion_array.get_cached(f'diffusion_start_place {strategy}_use_inner')

        def save_and_return(_place, _strategy=strategy):
            kwargs_dict = {f'diffusion_start_place {_strategy}': _place}
            self.diffusion_array.cache(**kwargs_dict)
            return _place

        start_frame_number = self.detect_diffusion_start_frame()
        image = self.diffusion_array.background_removed(start_frame_number)
        image = image.frame(start_frame_number + 1)[:]
        biggest_difference_place = np.unravel_index(np.argmax(np.abs(image)), image.shape)

        if strategy == 'biggest-difference':
            return save_and_return(biggest_difference_place[::-1])

        if strategy == 'weighted-centroid':
            image[image < 0] = 0
            radius = int(((image.shape[0] * image.shape[1]) ** (1 / 2)) / 3)
            image[Mask.circle(image.shape, biggest_difference_place, radius).flip().ndarray] = 0

            # centroid
            total_intensity = np.sum(image, dtype=np.float64)
            rows, cols = np.indices(image.shape)
            weighted_rows = np.sum(image * rows, dtype=np.float64) / total_intensity
            weighted_cols = np.sum(image * cols, dtype=np.float64) / total_intensity

            weighted_centroid_place = (weighted_cols, weighted_rows)
            return save_and_return(weighted_centroid_place)

        if strategy == 'connected-components':
            smooth = skimage.filters.gaussian(image, sigma=1.5)
            thresh_value = np.percentile(smooth, 80)
            thresh = smooth > thresh_value
            morphed = skimage.morphology.closing(
                skimage.morphology.opening(thresh),
                footprint=np.ones((3, 3))
            )
            labels = skimage.measure.label(morphed, connectivity=2)
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

            if not use_inner:
                centroid = skimage.measure.regionprops(largest_cc.astype(np.uint8))[0].centroid[::-1]
                return save_and_return(centroid)

            inverted = 1 - morphed
            inverted = inverted * ndimage.binary_fill_holes(largest_cc)
            inner_labels = skimage.measure.label(inverted, connectivity=2)
            largest_inner_cc = inner_labels == np.argmax(np.bincount(inner_labels.flat)[1:]) + 1
            centroid = skimage.measure.regionprops(largest_inner_cc.astype(np.uint8))[0].centroid[::-1]
            return save_and_return(centroid, strategy + '_use_inner')

        raise ValueError(f"Unknown strategy ({strategy}). It must be either 'connected-components', 'weighted-centroid'"
                         f" or 'biggest-difference'.")

    def apply_for_each_frame(
            self,
            function: callable,
            remove_background: bool = False,
            normalize: bool = False,
            mask: Mask | None = None
    ) -> np.ndarray:
        """
        Applies a function to each frame of the diffusion array. The function must take a frame (2D ndarray) as it's
        parameter and return a single number. After applying the function it collects the outputs in an array.

        Args:
            function (callable): The function to apply to each frame.
            remove_background (bool, optional): If True, removes the background by subtracting the average of the first
                three frames. Default is False.
            normalize (bool, optional): If True, normalizes the result to the range [0, 1]. Default is False.
            If remove background is False gaussian blur will not be used. Default is False.
            mask (np.ndarray | None, optional): A mask to apply to the diffusion array. Default is None.

        Returns:
            np.ndarray: The result of applying the function to each frame.

        """
        arr = self.diffusion_array
        if remove_background:
            arr = arr.background_removed()

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
