import functools
from typing import Callable, Tuple

import numpy as np

from src.diffusion_array import DiffusionArray
from src.mask import Mask
from src.radial_time_profile import RadialTimeProfile


class Homogenizer:
    """
    A class for homogenizing a DiffusionArray.

    This class is used to homogenize a given DiffusionArray using various parameters specified during its construction.
    The homogenization process is conducted on concentric rings around a center point in the input array.

    To create an instance of the `Homogenizer` class, it is recommended to use the nested class `Homogenizer.Builder` to
    ensure that all required fields are provided.

    Methods:
        homogenize(diffusion_array: DiffusionArray) -> DiffusionArray:
            Conducts the homogenization process on the given DiffusionArray based on the provided parameters and
            returns the homogenized array.

    \n\n
    Example usage:
    \n

    homogenizer = Homogenizer.Builder().center_point(start_place).start_frame(start_frame).build() \n
    homogenized = homogenizer.homogenize(diffusion_array)

    """

    def __init__(
            self,
            center_point: Tuple[int | float, int | float],
            start_frame: int,
            mask_extractor: Callable[[np.ndarray | DiffusionArray], Mask],
            delta_r: int,
            is_silent: bool,
            is_normalizing: bool,
            aggregating_function: Callable[[DiffusionArray | np.ndarray], float],
    ):
        """
        The constructor **should not be invoked directly**; use the Builder pattern instead.
        """
        self._center_point = center_point
        self._start_frame = start_frame
        self._mask_extractor = mask_extractor
        self._aggregating_function = aggregating_function
        self._delta_r = delta_r
        self._is_silent = is_silent
        self._is_normalizing = is_normalizing

    def homogenize(self, diffusion_array: DiffusionArray) -> RadialTimeProfile:
        """
        Creates a homogenized version of the given DiffusionArray based on the parameters provided at construction.
        The DiffusionArray will not be changed, instead a new DiffusionArray will be created.

        Args:
            diffusion_array (DiffusionArray): The input array for the homogenization.

        Returns:
            DiffusionArray: The homogenized version of the input array as a new DiffusionArray object.
        """

        mask = self._mask_extractor(diffusion_array).ndarray
        if mask.ndim == 2:
            mask = np.tile(mask[np.newaxis, ...], (diffusion_array.number_of_frames, 1, 1))

        if mask.shape != diffusion_array.shape:
            raise ValueError(f'The mask and the diffusion array must have the same shape.')

        center_x, center_y = tuple(map(round, self._center_point))
        width = diffusion_array.width
        height = diffusion_array.height

        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        xs -= center_x
        ys -= center_y
        distances = np.hypot(xs, ys)

        max_distance = int(np.ceil(np.max(distances)) * 0.95)
        masked_data_array = diffusion_array[:].copy() * mask
        intensities_by_distance = np.zeros((diffusion_array.number_of_frames - 1, max_distance))

        for i in range(0, max_distance, self._delta_r):
            distances_i_delta_r_ = (i <= distances) & (distances < i + self._delta_r)
            masked_data = masked_data_array[:, distances_i_delta_r_]
            intensities_by_distance[:, i] = self._aggregating_function(masked_data)

            if not self._is_silent and i % 100 == 0:
                print(f'Homogenizing... step: {i:4d} of {max_distance:4d}')

        intensities_by_distance = np.nan_to_num(intensities_by_distance, nan=0)

        min_value = np.min(intensities_by_distance)
        max_value = np.max(intensities_by_distance)

        if self._is_normalizing:
            normalized_arr = (intensities_by_distance - min_value) / (max_value - min_value)
        else:
            normalized_arr = intensities_by_distance

        return RadialTimeProfile(
            diffusion_array=diffusion_array,
            center=self._center_point,
            profile_array=normalized_arr
        )

    class Builder:
        """
        A nested class to build instances of the `Homogenizer` class with specified parameters.

        To create an instance of the `Homogenizer` class, it is recommended to use the nested class
        `Homogenizer.Builder`. The `Builder` allows you to configure the homogenization process.

        Attributes:
            All attributes of the `Homogenizer` class are available for customization through the builder.
            The following attributes have default values, making them optional:

            _mask_extractor (Optional[Callable[[np.ndarray | DiffusionArray], Mask]]): Default is None.
            _aggregating_function (Callable[[DiffusionArray | np.ndarray], float]): Default is np.mean.
            _delta_r (int): Default is 1.
            _is_silent (bool): Default is True.
            _is_normalizing (bool): Default is False.

            The following attributes don't have default values, thus it is necessary to provide them:

            _center (Tuple[float | int, ...]): The point around which the homogenization is conducted, required.
            _start_frame (int): The starting frame of the diffusion process, required.

        Methods:
            All methods for setting the attributes of the `Homogenizer` class are available through the `Builder`.
            Once the desired attributes are configured using the `Builder` methods, you can use the `build` method to
            create an instance of the `Homogenizer` class.
        """

        def __init__(self):
            self._center = None  # this must be provided
            self._start_frame = None  # this must be provided
            self._mask_extractor: Callable[[np.ndarray | DiffusionArray], Mask] = lambda darr: Mask.ones(darr.shape)
            self._delta_r: int = 1
            self._is_silent: bool = True
            self._is_normalizing: bool = False
            self._aggregating_function: Callable[[DiffusionArray | np.ndarray], float] = functools.partial(
                np.median,
                axis=1,
            )

        def center_point(self, center: Tuple[int, int]) -> 'Homogenizer.Builder':
            """
            Specifies the point around which the array should be homogenized. This point must be provided.

            Args:
                center (tuple): The center point of the homogenization.

            Returns:
                self
            """
            self._center = center
            return self

        def start_frame(self, start_frame: int) -> 'Homogenizer.Builder':
            """
            Specifies the starting frame for the diffusion process. Useful for creating certain types of Masks. The
            starting frame must be provided.

            Args:
                start_frame (int): The first frame of the diffusion process.

            Return:
                self
            """
            self._start_frame = start_frame
            return self

        def with_delta_r(self, delta_r: int) -> 'Homogenizer.Builder':
            """
            Changes the difference between the inner and outer radii of the annuli.

            Args:
                delta_r (int). The new difference Must be greater than 0.

            Returns:
                self
            """
            if delta_r <= 0:
                raise ValueError('`delta_r` must be positive.')

            self._delta_r = delta_r
            return self

        def use_mask(self, mask_extractor: Callable[[np.ndarray | DiffusionArray], Mask]) -> 'Homogenizer.Builder':
            """
            During homogenization only values where the mask is True will be used. If another mask was previously
            provided or only_homogenize_on_high_intensities were called their effect is disregarded as only the
            mask provided in this method call will be used.

            Args:
                mask_extractor (Mask) : The mask to be used.

            Returns:
                self
            """
            self._mask_extractor = mask_extractor
            return self

        def only_homogenize_on_high_intensities(
                self,
                cutoff_percentile: float = 90,
                compute_for_each_frame: bool = False
        ) -> 'Homogenizer.Builder':
            """
            This method creates a mask that filters indexes in the input array based on intensity values during the
            homogenization. An intensity value is considered too low if it does not exceed the `cutoff_percentile`-th
            quantile of the intensity distribution.

            If another mask was previously provided or only_homogenize_on_high_intensities were called their effect is
            disregarded as only the mask created in this method call will be used.

            Args:
                cutoff_percentile (float): The percentile value (0 to 100) used as the threshold for low intensities.
                compute_for_each_frame (bool): Whether to compute the intensity distribution for each frame
                    individually.

            Returns:
                self
            """
            self._mask_extractor = functools.partial(
                Mask.threshold_percentile_high,
                cutoff_percentile=cutoff_percentile,
                compute_for_each_frame=compute_for_each_frame
            )
            return self

        def report_progress(self, is_reporting: bool = True) -> 'Homogenizer.Builder':
            """
            Specifies if the progress report should be printed during homogenization.

            Args:
                is_reporting (bool): whether to print the homogenization progress.

            Returns:
                self
            """
            self._is_silent = not is_reporting
            return self

        def normalize(self, is_normalizing: bool = True) -> 'Homogenizer.Builder':
            """
            Normalization remaps the values during homogenization in the DiffusionArray to [0, 1) interval.

            Args:
                is_normalizing (bool): indicates if the diffusion array should be remapped to [0, 1).

            Returns:
                self
            """
            self._is_normalizing = is_normalizing
            return self

        def aggregating_function(self, function: Callable) -> 'Homogenizer.Builder':
            """
            Specifies the aggregating function for the diffusion process. For example whether to use the mean or the
            median of the values selected by the aggregating mask.

            Args:
                function (Callable): The aggregating function should return an int or a double.

            Returns:
                self
            """
            self._aggregating_function = function
            return self

        def build(self) -> 'Homogenizer':
            """
            Build and return an instance of the `Homogenizer` class with the specified parameters. Should only be called
            if center_point and start_frame are already specified.

            Returns:
                Homogenizer: The created `Homogenizer` object.
            """
            if self._center is None:
                raise ValueError('A center point for the homogenization must be provided.')

            if self._start_frame is None:
                raise ValueError('The start frame of the diffusion process must be provided.')

            return Homogenizer(
                center_point=self._center,
                start_frame=self._start_frame,
                mask_extractor=self._mask_extractor,
                aggregating_function=self._aggregating_function,
                delta_r=self._delta_r,
                is_silent=self._is_silent,
                is_normalizing=self._is_normalizing,
            )
