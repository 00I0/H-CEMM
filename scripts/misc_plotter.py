import functools
import inspect
import math
from math import ceil, sqrt
from typing import List, Sequence, Iterator, Iterable

import numpy as np
from lmfit import Model, Parameters
from matplotlib import pyplot as plt, gridspec, patches
from matplotlib.axes import Axes

from core.analyzer import Analyzer
from core.diffusion_array import DiffusionArray
from core.homogenizer import Homogenizer
from core.mask import Mask
from core.radial_time_profile import RadialTimeProfile
from core.step_widget import PipeLineWidget, StartFrameWidget, ClippingWidget, StartPlaceWidget, \
    BackgroundRemovalWidget, NormalizingWidget


def configurable_plot(function):
    # noinspection PyUnresolvedReferences
    """
        Decorator that enhances a plotting function with additional configurable options for display and saving.
        It extracts specific keyword arguments to control plot appearance and behavior, allowing for adjustments.

        Args:
            function (Callable): The plotting function to be decorated. This function must be capable of accepting
                arbitrary args and kwargs, with kwargs potentially including keys relevant to this decorator.

        Returns:
            Callable: A wrapper function that adds functionality for managing plot
            configuration such as saving, showing, and adjusting title and margins
            based on provided keyword arguments.

        Keyword Args:
            save (bool): If True, the plot will be saved to a file. Default is False.
            show (bool): If True, display the plot window. Default is True.
            title (str): The title of the plot. If None, no title is set. Default is None.
            save_title (str): The filename for saving the plot. If None, it defaults to the function name
                with '_plot.png'. If `title` is provided and `save_title` is None, `title` is used as the filename.
                Default is None.
            thin_margins (bool): If True, adjusts the plot margins to be minimal. Default is False.
        """

    def wrapper(*args, **kwargs):
        signature = inspect.signature(function)
        params = signature.parameters

        def extract_value(name, default_value):
            if name in params:
                default_value = params[name].default

            return kwargs.pop(name, default_value)

        save: bool = extract_value('save', False)
        show: bool = extract_value('show', True)
        title: str = extract_value('title', None)
        save_title: str = extract_value('save_title', None)
        thin_margins: bool = extract_value('thin_margins', False)

        result = function(*args, **kwargs)

        fig = plt.gcf()  # Get the current figure
        if title is not None:
            fig.suptitle(title)

        if thin_margins:
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95 if title else 0.98)

        if save:
            if save_title is None and title is None:
                save_title = f'{function.__name__}_plot.png'
            if save_title is None and title is not None:
                save_title = title

            print(save_title)
            plt.savefig(save_title, dpi=300)

        if show:
            plt.show()

        return result

    return wrapper


@configurable_plot
def plot_start_neighbourhood(
        darr: DiffusionArray,
        start_frame: int = None,
        start_place: tuple[float, float] = None,
        show_start_place: bool = True
):
    """
    Plots the diffusion start neighborhood across three frames: the detected start frame, one frame before,
    and one frame after. It optionally highlights the start place in all three frames.

    Args:
        darr (DiffusionArray): The diffusion data.
        start_frame (int): The frame number identified as the start of diffusion.
            If None, it will be automatically detected. Defaults to None.
        start_place (tuple[float, float]): The coordinates (x, y) identifying the start place of diffusion.
            If None, it will be automatically detected. Defaults to None.
        show_start_place (bool): If True, highlights the start place on the plots. Default is True.
    """
    analyzer = Analyzer(darr.channel(0))
    if start_frame is None:
        start_frame = analyzer.detect_diffusion_start_frame()
    if start_place is None:
        start_place = analyzer.detect_diffusion_start_place()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    v_max = max(
        darr.frame(start_frame - 1)[:].max(),
        darr.frame(start_frame)[:].max(),
        darr.frame(start_frame + 1)[:].max()
    )

    v_min = min(
        darr.frame(start_frame - 1)[:].min(),
        darr.frame(start_frame)[:].min(),
        darr.frame(start_frame + 1)[:].min()
    )

    axes[1].imshow(darr.channel(0).frame(start_frame), vmax=v_max, vmin=v_min)
    axes[1].tick_params(label1On=False, tick1On=False)
    axes[1].set_title(f'detected start (t = {start_frame:2d})')

    axes[0].imshow(darr.channel(0).frame(start_frame - 1), vmax=v_max, vmin=v_min)
    axes[0].tick_params(label1On=False, tick1On=False)
    axes[0].set_title('start - 1')

    axes[2].imshow(darr.channel(0).frame(start_frame + 1), vmax=v_max, vmin=v_min)
    axes[2].tick_params(label1On=False, tick1On=False)
    axes[2].set_title('start + 1')

    if show_start_place:
        for i in range(3):
            axes[i].scatter(*start_place, color='red', alpha=.5)

    plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.02)


@configurable_plot
def plot_starting_place_finder_comparisons(
        diffusion_arrays: Iterator[DiffusionArray] | List[DiffusionArray] | DiffusionArray,
        strategies: Sequence[str] = (
                'biggest-difference',
                'connected-components (use_inner: True)',
                'connected-components (use_inner: False)',
                'weighted-centroid'
        ),
        length: int | None = None,
        legend: Iterable[str] | None = (
                'legnagyobb különbség',
                'összefüggő komponensek (belső)',
                'összefüggő komponensek (külső)',
                'súlyozott centroid'
        )
) -> None:
    """
    This method creates a plot to show the difference between the different start place finder algorithms of the
    Analyzer class. The method shows the start_frame with the background removed for each diffusion array. The start
    places found by the different algorithms will also be shown with a uniquely colored circle.

    Args:
        diffusion_arrays (Iterator[DiffusionArray]| List[DiffusionArray] | DiffusionArray): The diffusion arrays on
        which the algorithms should be compared. If a single DiffusionArray is provided it will be treated as a list
        with only one entry. If an Iterator object is passed len must also be passed

        strategies (Sequence[str]): The name of the different algorithms, used as the 'strategy' parameter in
        Analyzer.detect_diffusion_start_place.

        length (int | None): number of DiffusionArrays in the input, if diffusion_arrays is not an Iterator it will be
        ignored.

        legend (Iterable[str]):
    """

    if isinstance(diffusion_arrays, DiffusionArray):
        diffusion_arrays = [diffusion_arrays]

    colors = ['orange', 'blue', 'green', 'red', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'yellow', 'lime',
              'black']

    if isinstance(diffusion_arrays, Iterator):
        array_length = length
    else:
        array_length = len(diffusion_arrays)

    height = ceil((-1 + sqrt(1 + 3 * array_length)) / 2)
    width = round(4 / 3 * height)

    if width * height < array_length:
        height += 1
        width = round(4 / 3 * height)

    fig = plt.figure(figsize=(width * 5, height * 5))
    gs = gridspec.GridSpec(height, width)

    for i, diffusion_array in enumerate(diffusion_arrays):
        diffusion_array = diffusion_array.channel(0)

        # noinspection PyTypeChecker
        ax: Axes = plt.subplot(gs[i // width, (i % width)])
        if array_length != 1:
            ax.set_title(diffusion_array.meta.name)
        ax.tick_params(label1On=False, tick1On=False)

        analyzer = Analyzer(diffusion_array)
        start_frame = analyzer.detect_diffusion_start_frame()
        ax.imshow(diffusion_array.frame(start_frame), vmin=0, vmax=1)

        for color, strategy in zip(colors, strategies):
            kw_dict = {}
            if '(' in strategy:
                kw_list = strategy.replace(':', '=').split(' (')[-1].strip('()').split(',')
                kw_dict = {x.split('=')[0].strip(): eval(x.split('=')[1].strip()) for x in kw_list}
                strategy = strategy.split(' (')[0]
            point = analyzer.detect_diffusion_start_place(strategy=strategy, **kw_dict)
            ax.scatter(*point, color=color, alpha=.9)

    fig.subplots_adjust(wspace=0.01)
    fig.legend(
        [patches.Circle((0, 0), radius=0.2, facecolor=c) for c in colors[:len(strategies)]],
        legend,
        loc='lower right',
        fontsize='large'
    )
    plt.tight_layout()


@configurable_plot
def plot_fit(
        data: np.ndarray,
        t_start: int,
        func: str = 'diffusion_decay',
):
    """
    Fits a model to the given data starting from `t_start`. The function supports two types of decay models:
    exponential and diffusion decay, which can be selected using the `func` parameter.

    Args:
        data (np.ndarray): A 1D array of the aggregated diffusion data
        t_start (int): The start frame of the diffusion process.
        func (str): The name of the function to use for fitting. Can be either 'exponential_decay' or 'diffusion_decay'.

    Raises:
        ValueError: If the provided `data` is not 1D.
        ValueError: If `func` is not a recognized function name.
    """
    if data.ndim != 1:
        raise ValueError(f'data must be 1D, but it was: {data.ndim}')

    average_before_start = np.mean(data[:t_start - 1])
    t_max = np.argmax(data)
    t_star = t_max - t_start + 1

    def exponential_decay(x, a, b):
        return a * np.exp(-b * x) + average_before_start

    def diffusion_decay(x, mu, c, ):
        p = c / mu * (1 - np.exp(-mu * x))
        p[x > t_star] = (c / mu * (1 - np.exp(-mu * t_star))) * np.exp(-mu * (x[x > t_star] - t_star))
        return p + average_before_start

    params = Parameters()
    fit_first_frame = t_start
    if isinstance(func, str):
        if func not in ('exponential_decay', 'diffusion_decay'):
            raise ValueError(
                f'If function is given as a string it must be either: "exponential_decay" or "diffusion_decay"'
            )

        elif func == 'exponential_decay':
            func = exponential_decay
            params.add('a', 2, min=-5, max=1e6)
            params.add('b', 0.3, min=-5, max=1e6)
            fit_first_frame = t_max

        elif func == 'diffusion_decay':
            func = diffusion_decay
            params.add('c', 0.5, min=-5, max=1e6)
            params.add('mu', 0.1, min=-5, max=1e6)
            fit_first_frame = t_start

    model = Model(func)
    result = model.fit(data[fit_first_frame:], params, x=np.arange(0, len(data) - fit_first_frame))
    plt.plot(data, 'bo', label='Data')
    plt.plot(np.arange(fit_first_frame, len(data)), result.best_fit, 'r-', label='Best Fit')
    plt.axvline(x=t_start, label='Start Frame', color='green')
    plt.legend()

    for param in result.params.values():
        print(f'{param.name}: {param.value:2.2f}')
    print(f'\nr squared: {result.rsquared: .4f}')


@configurable_plot
def plot_different_homogenization_centers(
        diffusion_array: DiffusionArray,
        darr_with_bcg: DiffusionArray,
        builder: Homogenizer.Builder = Homogenizer.Builder().report_progress(False).use_mask(functools.partial(
            Mask.threshold_percentile_high,
            percentile=40
        )).with_delta_r(10).aggregating_function(functools.partial(
            np.mean,
            axis=1,
        ))
) -> None:
    """
    This method shows the effect of homogenizing on different centers. It creates 12 subplots to show the original
    diffusion array at the start of the process, the diffusion array with the background removed, the diffusion array
    homogenized on the detected center as well as 9 additional homogenizations each with a different homogenization
    center.

    Args:
        diffusion_array (DiffusionArray): The diffusion array on which the homogenization should be shown.
        darr_with_bcg (DiffusionArray): The diffusion array with the background.
        builder (Homogenizer.Builder): A builder object to customize the homogenization process. By default, the
            progress of the homogenization will not be reported.

    Returns:
        None
    """
    diffusion_array = diffusion_array.channel(0)
    analyzer = Analyzer(diffusion_array)
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame() + 1
    diffusion_array = diffusion_array.background_removed().frame(f'0:{start_frame + 1}')

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 24))

    ax = axs[0][0]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(darr_with_bcg.frame(start_frame))
    ax.scatter(*start_place, color='red')

    ax = axs[0][1]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(diffusion_array.frame(start_frame))
    ax.scatter(*start_place, color='red')

    ax = axs[0][2]
    ax.tick_params(label1On=False, tick1On=False)
    homogenizer = builder.start_frame(start_frame).center_point(start_place).build()
    homogenized = homogenizer.homogenize(diffusion_array).to_diffusion_array().frame(start_frame)
    ax.imshow(homogenized)
    ax.scatter(*start_place, color='red')

    test_points = [
        (round(y * diffusion_array.height / 4), round(x * diffusion_array.width / 4))
        for x in range(1, 4) for y in range(1, 4)
    ]

    for i, center_point in enumerate(test_points):
        homogenizer = builder.start_frame(start_frame).center_point(center_point).build()
        homogenized = homogenizer.homogenize(diffusion_array).to_diffusion_array().frame(start_frame)

        ax = axs[i // 3 + 1][i % 3]
        ax.tick_params(label1On=False, tick1On=False)
        ax.imshow(homogenized)
        ax.scatter(*center_point, color='red')

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)


def compose_pixelwise(diffusion_arrays: List[DiffusionArray]) -> DiffusionArray:
    """
    Creates a pixelwise sum of the given diffusion arrays. The diffusion arrays will be translated to have their
    detected starting points be on the same point. The output diffusion array will have the biggest possible height and
    width as long as this rectangle is not outside any of the individual input diffusion arrays.

    Args:
        diffusion_arrays (List[DiffusionArray]): The input diffusion arrays which are to be composed

    Returns:
        DiffusionArray: A new diffusion array created as the pixelwise sum of the translated input arrays.

    """
    left = right = top = bottom = before = after = math.inf

    diffusion_arrays: List[DiffusionArray] = list(map(lambda darr: darr.channel(0), diffusion_arrays))

    for diffusion_array in diffusion_arrays:
        print(diffusion_array.meta.name, diffusion_array.shape)
        analyzer = Analyzer(diffusion_array)
        start_frame = analyzer.detect_diffusion_start_frame()
        start_x, start_y = analyzer.detect_diffusion_start_place()
        width, height = diffusion_array.frame(0).shape

        left = min(left, start_x)
        right = min(right, width - start_x)
        top = min(top, start_y)
        bottom = min(bottom, height - start_y)
        before = min(before, start_frame)
        after = min(after, diffusion_array.number_of_frames - start_frame)

    aggregate = np.zeros((before + after, left + right, top + bottom), dtype=np.float32)

    for diffusion_array in diffusion_arrays:
        analyzer = Analyzer(diffusion_array)
        start_frame = analyzer.detect_diffusion_start_frame()
        start_x, start_y = analyzer.detect_diffusion_start_place()
        start_x = round(start_x)
        start_y = round(start_y)

        diffusion_array = diffusion_array.cropped(
            (start_x - left, start_y - top),
            (start_x + right, start_y + bottom)
        )
        diffusion_array = diffusion_array.background_removed()
        diffusion_array = diffusion_array.normalized(0, 4048)
        diffusion_array = diffusion_array.frame(f'{start_frame - before}:{start_frame + after}')

        aggregate += diffusion_array

    return DiffusionArray(path=None, ndarray=aggregate)


@configurable_plot
def plot_front_speed(rtp: RadialTimeProfile, start_frame: int):
    """
    Visualizes the ATP fronts in a Radial Time Profile.

    Args:
        rtp (RadialTimeProfile): The homogenized values along a radius.
        start_frame (int): The frame at which the diffusion process starts.

    The function performs the following steps:
    - Extracts a sub-array, excluding the first and last 10 units for clearer analysis.
    - Calculates a background value based on the mean and maximum values before the start frame.
    - Identifies points where the intensity is above the background.
    - Filters these points, selecting the highest points in each column to avoid multiple selections per frame.
    - Determines a 'peak' point that is local maxima in the time dimension.
    - Highlights those points in red on that precede the peak by frame index.
    - Fits a linear model to the peak points and plots the resulting best fit line.
    """
    min_c = np.min(rtp)
    max_c = np.max(rtp)

    plt.imshow(np.array(rtp).T, aspect='auto', vmin=min_c, vmax=max_c)
    plt.gca().invert_yaxis()
    plt.ylabel('Distance from origin')
    plt.xlabel('Frame')
    plt.ylim(0, rtp.width)

    np_array_rtp__t = np.array(rtp).T[10:-10]
    background = max(np.mean(np_array_rtp__t[:, start_frame - 5:start_frame + 3]),
                     np.max(np_array_rtp__t[:, :start_frame]))

    np_array_rtp__t[np_array_rtp__t > background] = 1
    cols, rows = np.where(np_array_rtp__t > background)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)

    occupied_cols = [False] * rtp.number_of_frames
    selected_rows = []
    selected_cols = []

    for row, col in sorted(zip(rows, cols), reverse=True):
        if occupied_cols[row]:
            continue

        selected_rows.append(row)
        selected_cols.append(col)
        occupied_cols[row] = True

    cords = list()

    for row, col in sorted(zip(selected_rows, selected_cols), key=lambda tup: tup[1]):
        cords.append((row, col + 10))

    cords = list(sorted(cords))
    red_cords = {cords[0]}
    highest = 0
    for i in range(1, len(cords) - 1):
        prev = cords[i - 1]
        this = cords[i]
        next = cords[i + 1]
        if prev[1] < this[1] and next[1] < this[1]:
            this = (this[0], (prev[1] + next[1]) / 2)

        if highest <= this[1] <= np_array_rtp__t.shape[0] - 10:
            red_cords.add(this)

        highest = max(highest, this[1])
        cords[i] = this

    first_x = min(x for x, y in set(cords) - red_cords)
    true_red = []
    for row, col in cords:
        color = 'orange'
        if (row, col) in red_cords and row < first_x:
            color = 'red'
            true_red.append((row, col))

        plt.scatter(row, col, color=color)

    def line(x, m, b):
        return m * x + b

    red_cords = list(sorted(true_red, key=lambda tup: tup[0]))
    model = Model(line)
    params = model.make_params(m=1, b=0)
    result = model.fit([cord[1] for cord in red_cords], params, x=[cord[0] for cord in red_cords])

    for param in params:
        print(f'{param:} = {result.params[param].value: .4f}')

    print(f'\nr squared = {result.rsquared: .4f}')
    plt.plot([cord[0] for cord in red_cords], result.best_fit, color='lime')
    plt.tight_layout()


def test_resizing(file_name: str) -> None:
    """
    This method tests the `DiffusionArray.resize` method by plotting a frame with the original and the resized size.
    """
    darr = DiffusionArray(file_name)
    plt.imshow(darr.channel(0).frame(12))
    plt.show()
    plt.imshow(darr.resized(50, 50).channel(0).frame(12))
    plt.show()


# noinspection DuplicatedCode
def main(diffusion_array_path):
    darr_with_background = DiffusionArray(diffusion_array_path).channel(0)

    pipeline = PipeLineWidget(None, display_mode=False)
    pipeline.add_step(ClippingWidget())
    pipeline.add_step(StartFrameWidget())
    pipeline.add_step(StartPlaceWidget())
    pipeline.add_step(BackgroundRemovalWidget())
    pipeline.add_step(NormalizingWidget())

    darr, start_frame, start_place = pipeline.apply_pipeline(darr_with_background)
    rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)

    plot_start_neighbourhood(darr, start_frame, start_place)
    plot_starting_place_finder_comparisons(darr)
    plot_different_homogenization_centers(darr, darr_with_background, title='Different homogenization centers compared')
    plot_front_speed(rtp, start_frame)


if __name__ == '__main__':
    main('G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl018.nd2')
