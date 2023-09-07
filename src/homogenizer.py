import warnings

import numpy as np
from typing import Callable, NewType, Tuple, Optional

from src.analyzer import Analyzer
from src.diffusion_array import DiffusionArray
from src.mask import Mask


class Homogenizer:
    """
    A class for homogenizing a DiffusionArray.

    This class is used to homogenize a given DiffusionArray using various parameters specified during its construction.
    The homogenization process is conducted on concentric rings around a center point in the input array.

    To create an instance of the `Homogenizer` class, it is recommended to use the nested class `Homogenizer.Builder` to
    ensure that all required fields are provided.

    Attributes:
        _center_point (Tuple[float | int, ...]): The point around which homogenization is conducted.

        _start_frame (int) The first frame of the diffusion process, useful in creating certain types of Masks

        _aggregating_mask_extractor (Optional[_MaskExtractor]): A mask used in the computation for
        the homogenized value. This mask allows the exclusion of specific indexes (for example those whose value deemed
        too high) during computation. If not provided, no values will be excluded.

        _homogenizing_mask_extractor (_MaskExtractor): A mask used to select indexes of a concentric ring. The selected
        indexes values will change during an iteration the homogenization process.

        _background_removing_function (Callable[[DiffusionArray], DiffusionArray]): A function used for background
        removal. Background removal is applied if `is_removing_background` is set to True.

        _delta_r (int): The difference between the inner and outer radii of a ring.

        _is_silent (bool): Whether to print the progress of the homogenization. Progress is indicated by printing the
        current iteration counter every 100 iterations.


        _is_removing_background (bool): Whether to remove background during homogenization.

        _is_normalizing (bool): Whether to normalize the array to [0, 1) after homogenization. Normalization is applied
        after background removal (if both should be applied).

    Methods:
        homogenize(diffusion_array: DiffusionArray) -> DiffusionArray:
            Conducts the homogenization process on the given DiffusionArray based on the provided parameters and
            returns the homogenized array.

    \n\n
    Example usage:
    \n

    homogenizer = Homogenizer.Builder().center_point(start_place)start_frame(start_frame).build() \n
    homogenized = homogenizer.homogenize(diffusion_array)

    """
    _MaskExtractor = NewType('MaskExtractor', Callable[
        [
            DiffusionArray,  # DiffusionArray
            Tuple[float | int, ...],  # center
            int,  # first frame
            int | float,  # inner radius
            int | float,  # outer radius
        ],
        Mask
    ])

    def __init__(
            self,
            center_point: Tuple[float | int, ...],
            start_frame: int,
            aggregating_mask_extractor: Optional[_MaskExtractor],
            homogenizing_mask_extractor: _MaskExtractor,
            aggregating_function: Callable[[DiffusionArray | np.ndarray], float],
            background_removing_function: Callable[[DiffusionArray], DiffusionArray],
            delta_r: int,
            is_silent: bool,
            is_removing_background: bool,
            is_normalizing: bool,
    ):
        """
        Constructor **should not be invoked directly**; use the Builder pattern instead.
        """
        self._center_point = center_point
        self._start_frame = start_frame
        self._aggregating_mask_extractor = aggregating_mask_extractor
        self._homogenizing_mask_extractor = homogenizing_mask_extractor
        self._aggregating_function = aggregating_function
        self._background_removing_function = background_removing_function
        self._delta_r = delta_r
        self._is_silent = is_silent
        self._is_removing_background = is_removing_background
        self._is_normalizing = is_normalizing

    def homogenize(self, diffusion_array: DiffusionArray) -> DiffusionArray:
        """
        Creates a homogenized version of the given DiffusionArray based on the parameters provided at construction.
        The DiffusionArray will not be changed, instead a new DiffusionArray will be created.

        Args:
            diffusion_array (DiffusionArray): The input array for the homogenization.

        Returns:
            DiffusionArray: The homogenized version of the input array as a new DiffusionArray object.
        """
        # TODO  add support to only homogenize one frame
        if self._is_removing_background:
            diffusion_array = self._background_removing_function(diffusion_array)

        if self._is_normalizing:
            diffusion_array = diffusion_array.normalized()

        center = self._center_point
        center_x, center_y = center
        aggregator = Analyzer(diffusion_array)
        homogenized = diffusion_array.copy()

        width = max(center_x, diffusion_array.width - center_x)
        height = max(center_y, diffusion_array.height - center_y)
        diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

        if self._aggregating_mask_extractor is not None:
            aggregating_mask = self._aggregating_mask_extractor(diffusion_array, center, self._start_frame, 0, 0)
        else:
            aggregating_mask = None

        for i in range(0, diagonal, self._delta_r):
            if i % 100 == 0 and not self._is_silent:
                print(f'Homogenizing {diffusion_array.meta.name}, {i:4d} - {diagonal:4d}')

            homogenizing_mask = self._homogenizing_mask_extractor(diffusion_array, center, self._start_frame, i,
                                                                  i + self._delta_r)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                aggregate = aggregator.apply_for_each_frame(
                    self._aggregating_function,
                    mask=homogenizing_mask & aggregating_mask
                )
                aggregate = np.nan_to_num(aggregate, nan=0)
                homogenized[homogenizing_mask] = aggregate

        return homogenized

    class Builder:
        """
        A nested class to build instances of the `Homogenizer` class with specified parameters.

        To create an instance of the `Homogenizer` class, it is recommended to use the nested class
        `Homogenizer.Builder`. The `Builder` allows you to configure the homogenization process.

        Attributes:
            All attributes of the `Homogenizer` class are available for customization through the builder.
            The following attributes have default values, making them optional:

            _aggregating_mask_extractor (Homogenizer._MaskExtractor): Default is None.
            _homogenizing_mask_extractor (Homogenizer._MaskExtractor): Default is Mask.ring.
            _delta_r (int): Default is 2.
            _silent (bool): Default is False.
            _is_removing_background (bool): Default is False
            _is_normalizing (bool): Default is False.
            _aggregating_function (Callable[[DiffusionArray | np.ndarray], float]): Default is np.mean.
            _background_removing_function (Callable[[DiffusionArray], DiffusionArray]): Default is
            DiffusionArray.background_removed.


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
            self._aggregating_mask_extractor: Homogenizer._MaskExtractor = None
            self._homogenizing_mask_extractor: Homogenizer._MaskExtractor = None
            self._delta_r: int = 2
            self._silent: bool = False
            self._is_removing_background: bool = False
            self._is_normalizing: bool = False
            self._aggregating_function: Callable[[DiffusionArray | np.ndarray], float] = np.mean
            self._background_removing_function: Callable[[DiffusionArray], DiffusionArray] = lambda diffusion_array: (
                diffusion_array.background_removed()
            )

            self.homogenize_on_rings()

        def homogenize_on_rings(self) -> 'Homogenizer.Builder':
            """
            Changes the homogenizing_mask to be concentric rings.

            Returns:
                self
            """

            def ring_mask_extractor(diffusion_array, center, _, inner_radius, outer_radius):
                return Mask.ring(diffusion_array, center, inner_radius, outer_radius)

            self._homogenizing_mask_extractor = ring_mask_extractor
            return self

        def report_progress(self, is_reporting: bool = True) -> 'Homogenizer.Builder':
            """
            Specifies if the progress report should be printed during homogenization.

            Args:
                is_reporting (bool): whether to print the homogenization progress.

            Returns:
                self
            """
            self._silent = not is_reporting
            return self

        def with_delta_r(self, delta_r: int) -> 'Homogenizer.Builder':
            """
            Changes the difference between the inner and outer radii of the annuli.

            Args:
                delta_r (int). The new difference.

            Returns:
                self
            """
            self._delta_r = delta_r
            return self

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

        def remove_background(self,
                              is_removing_background: bool = True,
                              removing_function: Callable[[DiffusionArray], DiffusionArray] = None
                              ) -> 'Homogenizer.Builder':
            """
            Configures the background removal options. If a removing function is provided the boolean flag indicating
            background removal would be set to True. The default removing function is without arguments the
            DiffusionArray.background_removed() function

            Args:
                is_removing_background (bool, optional): Whether to enable background removal.
                removing_function (Callable[[DiffusionArray], DiffusionArray], optional): The new background remover
                function.

            Returns:
                self
            """
            self._is_removing_background = is_removing_background
            if removing_function is not None:
                self._background_removing_function = removing_function
                self._is_removing_background = True
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

        def filter_out_high_intensities(self, cutoff_percentile: float = 98,
                                        compute_for_each_frame: bool = False) -> 'Homogenizer.Builder':
            """
            This method creates an aggregating mask extractor that filters indexes in the input array based on intensity
            values. An intensity value is considered too high if it exceeds the `cutoff_percentile`-th quantile of the
            intensity distribution. The cutoff percentile determines the threshold for high intensity values.

            If `compute_for_each_frame` is set to True, the intensity distribution is computed separately for each
            frame, and indexes are filtered based on their intensity in each frame.
            If False, only the intensity values from the first frame are used as a reference, and indexes with the same
            x, y coordinates would be filtered out for the other frames.

            If another aggregating mask extractor is already applied only indexes selected by both masks would be used
            for aggregation.

            Args:
                cutoff_percentile (float): The percentile value (0 to 100) used as the threshold for high intensities.
                compute_for_each_frame (bool): Whether to compute the intensity distribution for each frame
                individually.

            Returns:
                self
            """

            def star_mask_extractor_same_for_frames(diffusion_array, *_):
                return Mask.cutoff(
                    diffusion_array.frame(0),
                    np.percentile(diffusion_array.frame(0), cutoff_percentile)
                ).flip()

            def star_mask_extractor_different_for_frames(diffusion_array, *_):
                return Mask.cutoff(
                    diffusion_array,
                    np.percentile(
                        np.percentile(diffusion_array, cutoff_percentile, axis=2),
                        cutoff_percentile,
                        axis=1
                    )
                ).flip()

            if compute_for_each_frame:
                star_mask_extractor = star_mask_extractor_different_for_frames
            else:
                star_mask_extractor = star_mask_extractor_same_for_frames

            if self._aggregating_mask_extractor is None:
                self._aggregating_mask_extractor = star_mask_extractor
            else:
                self._aggregating_mask_extractor = lambda *args: (
                        self._aggregating_mask_extractor(*args) & star_mask_extractor(*args)
                )

            return self

        def only_homogenize_on_cells(self, cutoff_percentile=66, compute_for_each_frame=False) -> 'Homogenizer.Builder':
            """
            This method creates an aggregating mask extractor that filters indexes in the input array based on intensity
            values. An intensity value is considered too low if it does not exceed the `cutoff_percentile`-th quantile
            of the intensity distribution.

            If `compute_for_each_frame` is set to True, the intensity distribution is computed separately for each
            frame, and indexes are filtered based on their intensity in each frame.
            If False, only the intensity values from the first frame are used as a reference, and indexes with the same
            x, y coordinates would be filtered out for the other frames.

            If another aggregating mask extractor is already applied only indexes selected by both masks would be used
            for aggregation.

            Args:
                cutoff_percentile (float): The percentile value (0 to 100) used as the threshold for low intensities.
                compute_for_each_frame (bool): Whether to compute the intensity distribution for each frame
                individually.

            Returns:
                self
            """

            def cell_mask_extractor_same_for_frames(diffusion_array, _, first_frame, *__):
                return Mask.cutoff(
                    diffusion_array,
                    np.percentile(diffusion_array.frame(first_frame), cutoff_percentile)
                ).for_frame(first_frame + 1)

            def cell_mask_extractor_different_for_frames(diffusion_array, *_):
                return Mask.cutoff(
                    diffusion_array,
                    np.percentile(
                        np.percentile(diffusion_array, cutoff_percentile, axis=2),
                        cutoff_percentile,
                        axis=1
                    )
                )

            if compute_for_each_frame:
                cell_mask_extractor = cell_mask_extractor_same_for_frames
            else:
                cell_mask_extractor = cell_mask_extractor_different_for_frames

            if self._aggregating_mask_extractor is None:
                self._aggregating_mask_extractor = cell_mask_extractor
            else:
                self._aggregating_mask_extractor = lambda *args: (
                        self._aggregating_mask_extractor(*args) & cell_mask_extractor(*args)
                )

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
                self._center,
                self._start_frame,
                self._aggregating_mask_extractor,
                self._homogenizing_mask_extractor,
                self._aggregating_function,
                self._background_removing_function,
                self._delta_r,
                self._silent,
                self._is_removing_background,
                self._is_normalizing
            )
