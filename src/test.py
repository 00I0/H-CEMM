import functools
import inspect
import math
import os
import re
from math import ceil, sqrt
from multiprocessing import Pool

import numpy as np
from lmfit import Model, Parameters, create_params
from matplotlib import pyplot as plt, gridspec, patches
from matplotlib.axes import Axes
from typing import List, Sequence, Tuple, Iterator

from analyzer import Analyzer
from diffusion_array import DiffusionArray
from reader import ND2Reader
from src.diffusion_PDEs import LogisticDiffusionPDE, LinearDiffusivityPDE, SigmoidDiffusivityPDE
from src.homogenizer import Homogenizer
from src.mask import Mask
from src.optimizer import Optimizer
from src.pde_solver import SymmetricIVPPDESolver
from src.radial_time_profile import RadialTimeProfile


# directory = 'G:\\rost\\Ca2+_laser'
def create_files_in_directory(directory: str):
    for filename in os.listdir(directory):
        if filename.endswith(".nd2"):
            file_path = os.path.join(directory, filename)
            npz_filename = os.path.splitext(filename)[0] + ".npz"
            npz_filepath = os.path.join(directory, npz_filename)

            meta_filename = os.path.splitext(filename)[0] + "_meta.txt"
            meta_filepath = os.path.join(directory, meta_filename)

            DiffusionArray(file_path).save(npz_filepath)
            with open(meta_filepath, "w", encoding="utf-8") as meta_file:
                meta = str(ND2Reader().meta(file_path))
                meta_file.write(meta)


def homogenize_directory(
        input_directory: str,
        select_regex: str = '.*\\.nd2',
        output_suffix: str = '_homogenized_average.npz',
        output_directory: str | None = None,
        homogenizer_builder: Homogenizer.Builder = Homogenizer.Builder().remove_background()
) -> None:
    """
    This method creates the homogenized versions of specific files in a directory. You can specify the diffusion process
    by changing the default homogenizer_builder. For example if you want to exclude the highest intensities you can use
    Homogenizer.Builder().filter_out_high_intensities(). To load the diffusion data from a directory this function
    delegates to DiffusionArray.get_all_from_directory.

    Args:
        input_directory (str): The director where the input data is located.
        select_regex (str): Only files with names matching this regex in the input_directory will be used as inputs.
        output_suffix (str): The suffix for each output with extension. The original files extension will be replaced
        with this. Example (output_suffix = 'avg.npz'): '1472@laser001.nd2' -> '1472@laser001_avg.npz'.
        output_directory (str | None): The directory where the homogenized data is to be saved. If None the input
        directory will be used.
        homogenizer_builder (Homogenizer.Builder): A builder object to customize the homogenization process.

    Returns:
        None
    """

    if output_directory is None:
        output_directory = input_directory

    diffusion_arrays = DiffusionArray.get_all_from_directory(input_directory, regex=select_regex)

    for diffusion_array in diffusion_arrays:
        analyzer = Analyzer(diffusion_array.channel(0))
        start_place = analyzer.detect_diffusion_start_place()
        start_frame = analyzer.detect_diffusion_start_frame()

        homogenized_name = diffusion_array.meta.name + output_suffix
        homogenized_path = os.path.join(output_directory, homogenized_name)

        homogenizer = homogenizer_builder.start_frame(start_frame).center_point(start_place).build()
        homogenized = homogenizer.homogenize(diffusion_array)
        homogenized.save(homogenized_path)


def configurable_plot(function):
    """TODO"""

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

        # plt.clf()

        if title is not None:
            plt.title(title)

        if thin_margins:
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

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
def plot_start_neighbourhood(darr: DiffusionArray, start_frame=None, start_place=None, show_start_place=True):
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
        length: int | None = None
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

    Returns:
        None
    """

    if isinstance(diffusion_arrays, DiffusionArray):
        diffusion_arrays = [diffusion_arrays]

    colors = ['orange', 'black', 'purple', 'red', 'pink', 'brown', 'cyan', 'magenta', 'yellow', 'lime', 'silver']

    # TODO analyzer.detect_diffusion_start_place(strategy='connected-components', use_inner=True)

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

        ax: Axes = plt.subplot(gs[i // width, (i % width)])
        if array_length != 1:
            ax.set_title(diffusion_array.meta.name)
        ax.tick_params(label1On=False, tick1On=False)

        analyzer = Analyzer(diffusion_array)
        start_frame = analyzer.detect_diffusion_start_frame()
        ax.imshow(diffusion_array.frame(start_frame))

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
        strategies,
        loc='lower right',
        fontsize='large'
    )
    plt.tight_layout()


@configurable_plot
def plot_fit(
        data: np.ndarray,
        t_start: int,
        t_max: int,
        func: str = 'exponential_decay',
):
    if data.ndim != 1:
        raise ValueError(f'data must be 1D, but it was: {data.ndim}')

    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    t_star = t_max - t_start

    def diffusion_decay(x, mu, c):
        p = c / mu * (1 - np.exp(-mu * x))
        p[x > t_star] = (c / mu * (1 - np.exp(-mu * t_star))) * np.exp(-mu * (x[x > t_star] - t_star))
        return p

    params = Parameters()
    fit_first_frame = t_start
    if isinstance(func, str):
        if func not in ('exponential_decay', 'diffusion_decay'):
            raise ValueError(
                f'If function is given as a string it must be either: "exponential_decay" or "diffusion_decay"'
            )

        elif func == 'exponential_decay':
            func = exponential_decay
            params.add('a', 2, min=0, max=1e6)
            params.add('b', 0.3, min=0, max=1e6)
            fit_first_frame = t_max

        elif func == 'diffusion_decay':
            func = diffusion_decay
            params.add('c', 0.5, min=0, max=1e6)
            params.add('mu', 0.1, min=0, max=1e6)
            fit_first_frame = t_start

    model = Model(func)
    result = model.fit(data[fit_first_frame:], params, x=np.arange(0, len(data) - fit_first_frame))
    plt.plot(data, 'bo', label='Data')
    plt.plot(np.arange(fit_first_frame, len(data)), result.best_fit, 'r-', label='Best Fit')
    plt.axvline(x=t_start, label='Start Frame', color='green')
    y = 0.85
    for param in result.params.values():
        plt.text(0.8 * len(data), y, f'{param.name}: {param.value:2.2f}')
        y -= 0.1

    plt.text(0.8 * len(data), y, f'mse: {np.average((data[fit_first_frame:] - result.best_fit) ** 2):2.2f}')
    plt.legend()
    # TODO


@configurable_plot
def plot_homogenized(directories):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4)
    i = 0
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('_avg.npz'):
                print('processing: ', filename)
                file_path = os.path.join(directory, filename)
                darr = DiffusionArray(file_path).channel(0)

                analyzer = Analyzer(darr.channel(0))

                ax1 = plt.subplot(gs[i // 4, (i % 4)])
                ax1.tick_params(label1On=False, tick1On=False)

                new_filename = re.sub(r'_homogenized.*$', '', filename)

                ax1.set_title(new_filename)

                start_frame_number = analyzer.detect_diffusion_start_frame()
                frame = darr.frame(start_frame_number + 1)
                frame = frame[:] - np.mean(darr.frame(slice(start_frame_number - 1, start_frame_number + 1)), axis=0)
                ax1.imshow(frame)

                i = i + 1

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.94, bottom=0.01)


@configurable_plot
def plot_different_homogenization_centers(
        diffusion_array: DiffusionArray,
        darr_with_bcg: DiffusionArray,
        builder: Homogenizer.Builder = Homogenizer.Builder().only_homogenize_on_cells().report_progress(False),
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
    # background_removed_array = diffusion_array.background_removed()
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()
    diffusion_array = diffusion_array.frame(f'0:{start_frame + 1}')
    darr_with_bcg = darr_with_bcg.frame(f'0:{start_frame + 1}')

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
    homogenized = homogenizer.homogenize(diffusion_array).frame(start_frame)
    v_min = 0
    v_max = 1
    ax.imshow(
        homogenized,
        vmin=v_min,
        vmax=v_max
    )
    ax.scatter(*start_place, color='red')

    test_points = [
        (round(y * diffusion_array.height / 4), round(x * diffusion_array.width / 4))
        for x in range(1, 4) for y in range(1, 4)
    ]

    center_point: tuple
    for i, center_point in enumerate(test_points):
        homogenizer = builder.start_frame(start_frame).center_point(center_point).build()
        homogenized = homogenizer.homogenize(diffusion_array).frame(start_frame)

        ax = axs[i // 3 + 1][i % 3]
        ax.tick_params(label1On=False, tick1On=False)
        ax.imshow(
            homogenized,
            vmin=v_min,
            vmax=v_max
        )
        ax.scatter(*center_point, color='red')

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)


@configurable_plot
def plot_start_cells(directories):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4)
    i = 0
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".nd2"):
                print('processing: ', filename)
                file_path = os.path.join(directory, filename)
                darr = DiffusionArray(file_path).channel(0)

                ax = plt.subplot(gs[i // 4, (i % 4)])
                analyzer = Analyzer(darr.channel(0))
                start_place = analyzer.detect_diffusion_start_place()
                start_frame = analyzer.detect_diffusion_start_frame()

                darr_wo_bcg = darr.updated_ndarray(darr[:] - np.mean(darr.frame('0:3'), axis=0))

                cut = Mask.cutoff(darr_wo_bcg[:], np.percentile(darr.frame(start_frame), 66))

                ax.tick_params(label1On=False, tick1On=False)
                parts = filename.split("_")
                new_filename = "_".join(parts[:-2])
                ax.set_title(new_filename)

                # --

                ax.imshow(darr.frame(start_frame + 1))
                ax.imshow(cut.for_frame(start_frame + 1), cmap='jet', alpha=.3)

                i = i + 1

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)


@configurable_plot
def plot_inner_radi(filename):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    darr = DiffusionArray(filename)
    darr = darr.channel(0)

    analyzer = Analyzer(darr)
    start_frame = analyzer.detect_diffusion_start_frame()
    start_place = analyzer.detect_diffusion_start_place()

    darr = DiffusionArray(filename.replace('.nd2', '') + '_homogenized_avg_centroid.npz').channel(0)

    for idx, i in enumerate([1, 5, 10]):
        ax = axs[0, idx]
        radial = darr.frame(start_frame + i)[start_place[1].astype(int), start_place[0].astype(int):]
        inner_radius = np.where(radial >= 0)[0][0]
        outer_radius = np.where(radial >= .6 * np.max(radial))[0][0]

        ax.imshow(darr.frame(start_frame + i))
        inner_circle = plt.Circle(start_place, inner_radius, color='red', fill=False)
        outer_circle = plt.Circle(start_place, outer_radius, color='purple', fill=False)
        ax.add_artist(inner_circle)
        ax.add_artist(outer_circle)

        ax.legend([f'inner radius = {inner_radius}', f'outer radius = {outer_radius}'], handlelength=0)
        ax.set_title(f'start frame + {i}')
        ax.tick_params(label1On=False, tick1On=False)

    inner_radii = []
    outer_radii = []
    for i in range(start_frame, darr.number_of_frames - 1):
        radial = darr.frame(i)[start_place[1].astype(int), start_place[0].astype(int):]
        inner_radius = np.where(radial >= 0)[0][0]
        outer_radius = np.where(radial >= .6 * np.max(radial))[0][0]
        inner_radii.append(inner_radius)
        outer_radii.append(outer_radius)

    axs[1, 0].plot(inner_radii, color='red')
    axs[1, 0].plot(outer_radii, color='purple')

    axs[1, 0].legend([f'inner radii', f'outer radius'], handlelength=0)

    axs[1, 1].remove()
    axs[1, 2].remove()

    plt.tight_layout()


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


def concatenate_homogenized_radii(
        homogenized_array: DiffusionArray,
        start_place: Tuple[int | float, int | float]
) -> np.ndarray:
    """
    This method takes the intensities in the homogenized array along the longest line segment that has an end point in
    the given start place and is parallel to the sides of the diffusion array for each frame. Then stacks those line
    segments next to each other in a ndarray and returns it.

    Args:
          homogenized_array (DiffusionArray): A preferably homogenized diffusion_array whose line segment-wise
          intensities are to be stacked.
          start_place (Tuple[int | float, int | float]): The point around which the array was homogenized.

    Returns:
        np.ndarray: The line segment-wise intensities stacked next to each other in a 2D NumPy array.

    """
    # TODO debug this
    homogenized_array = homogenized_array.channel(0)
    x, y = (round(coordinate) for coordinate in start_place)
    width = homogenized_array.width
    height = homogenized_array.height

    edge_distances = [x, y, width - x, height - y]
    print(edge_distances)
    max_distance = max(edge_distances)

    if x == max_distance:
        print('x')
        return np.flip(homogenized_array.ndarray[:, 0, y, :x].T, axis=0)

    if y == max_distance:
        print('y')
        return homogenized_array.ndarray[:, 0, :y, x].T

    if width - x == max_distance:
        print('-x')
        return homogenized_array.ndarray[:, 0, y, x:].T

    if height - y == max_distance:
        print('-y')
        return homogenized_array.ndarray[:, 0, y:, x].T


@configurable_plot
def plot_concatenated_radii(array: np.ndarray) -> None:
    """
    Plots the 2D NumPy array created by concatenate_homogenized_radii(). Should only be used on the output of that
    function.

    Args:
        array (np.ndarray): A 2D ndarray, the output of concatenate_homogenized_radii().

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(array, aspect='auto')
    plt.gca().invert_yaxis()
    plt.ylabel('Distance from origin')
    plt.xlabel('Frame')


@configurable_plot
def font_speed(diffusion_array, homogenized):
    # file_name = 'G:\\rost\\kozep\\super_1472_5_laser_EC1flow_laserabl018.nd2'
    # darr = DiffusionArray(file_name).channel(0)
    analyzer = Analyzer(diffusion_array)
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()
    # homogenized = DiffusionArray(file_name.replace('.nd2', '_homogenized_avg.npz'))

    print(start_place)
    concatenated_radii = concatenate_homogenized_radii(homogenized, start_place)
    # concatenated_radii[20:25, 20:25] = 999

    background = np.max(concatenated_radii[:, :start_frame - 1])
    # concatenated_radii[:, :start_frame - 1] = 9999
    rows, cols = np.where(concatenated_radii > 2 * background)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    occupied_cols = [False for _ in range(diffusion_array.number_of_frames)]

    selected_rows = []
    selected_cols = []
    for row, col in sorted(zip(rows, cols), reverse=True):

        if occupied_cols[col]:
            continue
        selected_rows.append(row)
        selected_cols.append(col)

        occupied_cols[col] = True

        if row == col:
            print(f'==({row}, {col})')

    # concatenated_radii[selected_rows, selected_cols] = 1.33 * np.max(concatenated_radii)
    cords = set()
    red_cords = set()

    plot_concatenated_radii(concatenated_radii, show=False)
    highest = 0
    for row, col in sorted(zip(selected_rows, selected_cols), key=lambda tup: tup[1]):
        if row == concatenated_radii.shape[0] - 1 or row <= highest:
            pass
        else:
            red_cords.add((col, row))

        cords.add((col, row))
        highest = max(row, highest)

    first_x = min([x for x, y in cords - red_cords])
    true_red = []
    for row, col in cords:
        color = 'orange'
        if (row, col) in red_cords and row < first_x:
            color = 'red'
            true_red.append((row, col))

        plt.scatter(row, col, color=color)

    def line(x, a, b):
        return a * x + b

    red_cords = true_red
    model = Model(line)
    params = model.make_params(a=1, b=0)
    result = model.fit([cord[1] for cord in red_cords], params, x=[cord[0] for cord in red_cords])
    y_max = concatenated_radii.shape[0] - 1
    plt.plot([cord[0] for cord in red_cords], np.clip(result.best_fit, a_min=None, a_max=y_max), color='lime')
    r_squared_text = (
        f'R-squared: {result.rsquared:.4f}\n '
        f'a: {result.best_values["a"]:.2e}, '
        f'b: {result.best_values["b"]:.2f}'
    )
    plt.legend(title=r_squared_text)


def test_resizing() -> None:
    """
    This method tests the `DiffusionArray.resize` method by plotting a frame with the original and the resized size.
    """
    file_name = 'G:\\rost\\kozep\\super_1472_5_laser_EC1flow_laserabl018.nd2'
    darr = DiffusionArray(file_name)
    plt.imshow(darr.channel(0).frame(12))
    plt.show()
    plt.imshow(darr.resized(200, 100).channel(0).frame(12))
    plt.show()


def optimeze_a_single_pde(homogenized_filename):
    start_place = Analyzer(DiffusionArray('G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec006.nd2').channel(
        0).resized(256, 256)).detect_diffusion_start_place()
    darr = DiffusionArray(homogenized_filename).channel(0).resized(256, 256).normalized().frame(f'{0}:{40}')
    analyzer = Analyzer(darr)

    frame_of_max_intensity = np.argmax(analyzer.apply_for_each_frame(np.max, normalize=True))
    target_frame = 40

    sec_per_frame = 0.2
    frames = target_frame - frame_of_max_intensity
    t_range = frames * sec_per_frame

    rtp = RadialTimeProfile(darr, start_place)
    start_radius = rtp.frame(frame_of_max_intensity)
    target_radius = rtp.frame(f'{frame_of_max_intensity}:{target_frame + 1}')

    pde = LinearDiffusivityPDE()

    i = 0
    for i in range(len(start_radius)):
        if start_radius[i] > start_radius[i + 1] and start_radius[i] > start_radius[i + 2]:
            break

    ivp = SymmetricIVPPDESolver(pde, start_radius, inner_radius=i)

    optimizer = Optimizer(ivp, target_radius, t_range)

    params = Parameters()
    params.add('diffusivity', 0.2636, min=0, max=20)
    # params.add('mu', 0.8, min=0.02, max=10)
    beta_bound = np.log((1 - 0.01) / 0.01)
    # params.add('beta', 0, min=-beta_bound, max=beta_bound)
    opt_params = optimizer.optimize(params, max_iter=1)
    print(opt_params)

    print('-' * 20)
    print(optimizer.optimal_parameters)
    print(optimizer.optimal_mse)
    print(optimizer.optimal_mse / (len(start_radius) - i))

    plt.plot(start_radius)
    plt.plot(rtp.frame(target_frame), color='red')
    plt.plot(optimizer.optimal_solution[3], color='green')
    plt.show()

    optimizer.plot_profile_comparisons('Ca2+_006_linear_40_prev.mp4', frame_of_max_intensity)


def _async_optimize_pdes_to_files(darr_path, hom_path, pdes, target_frame):
    start_place = Analyzer(
        DiffusionArray(darr_path).channel(0).resized(256, 256)
    ).detect_diffusion_start_place()
    darr = (
        DiffusionArray(hom_path)
        .channel(0)
        .resized(256, 256)
        .normalized().frame(f'{0}:{target_frame + 1}')
        .updated_ndarray()
    )

    frame_of_max_intensity = np.argmax(Analyzer(darr).apply_for_each_frame(np.max, normalize=True))

    sec_per_frame = 0.2
    frames = target_frame - frame_of_max_intensity
    t_range = frames * sec_per_frame

    rtp = RadialTimeProfile(darr, start_place)
    start_radius = rtp.frame(frame_of_max_intensity)
    target_radius = rtp.frame(f'{frame_of_max_intensity}:{target_frame + 1}')

    i = 0
    for i in range(len(start_radius)):
        if start_radius[i] > start_radius[i + 1] and start_radius[i] > start_radius[i + 2]:
            break

    for (pde, params) in pdes:
        ivp = SymmetricIVPPDESolver(pde, start_radius, inner_radius=i)

        optimizer = Optimizer(ivp, target_radius, t_range)
        optimizer.optimize(params, max_iter=50, report_progress=False)

        print(f'{darr.meta.name}: \n'
              f'pde: {type(pde)} \n'
              f'{optimizer.optimal_parameters} \n'
              f'{optimizer.optimal_mse}\n')


def optimize_pdes_asynchronously():
    files = [
        (
            'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec006.nd2',
            'G:\\rost\\Ca2+_laser\\homogenized\\1133_3_laser@30sec006_homogenized_avg.npz'
        ),
        (
            'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec007.nd2',
            'G:\\rost\\Ca2+_laser\\homogenized\\1133_3_laser@30sec007_homogenized_avg.npz'
        ),
        (
            'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec008.nd2',
            'G:\\rost\\Ca2+_laser\\homogenized\\1133_3_laser@30sec008_homogenized_avg.npz'
        ),
        (
            'G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl010.nd2',
            'G:\\rost\\kozep\\homogenized\\super_1472_5_laser_EC1flow_laserabl010_homogenized_avg.npz'
        ),
        (
            'G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl017.nd2',
            'G:\\rost\\kozep\\homogenized\\super_1472_5_laser_EC1flow_laserabl017_homogenized_avg.npz'
        ),
        (
            'G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl018.nd2',
            'G:\\rost\\kozep\\homogenized\\super_1472_5_laser_EC1flow_laserabl018_homogenized_avg.npz'
        ),
        (
            'G:\\rost\\sarok\\raw_data\\1472_4_laser@30sec001.nd2',
            'G:\\rost\\sarok\\homogenized\\1472_4_laser@30sec001_homogenized_avg.npz',
        ),
        (
            'G:\\rost\\sarok\\raw_data\\1472_4_laser@30sec002.nd2',
            'G:\\rost\\sarok\\homogenized\\1472_4_laser@30sec002_homogenized_avg.npz',
        ),
        (
            'G:\\rost\\sarok\\raw_data\\1472_4_laser@30sec003.nd2',
            'G:\\rost\\sarok\\homogenized\\1472_4_laser@30sec003_homogenized_avg.npz',
        ),
        (
            'G:\\rost\\sarok\\raw_data\\1472_4_laser@30sec004.nd2',
            'G:\\rost\\sarok\\homogenized\\1472_4_laser@30sec004_homogenized_avg.npz',
        ),
    ]

    beta_bound = np.log((1 - 0.01) / 0.01)
    pdes = [
        (
            LogisticDiffusionPDE(),
            create_params(
                diffusivity={'value': 0.1674, 'min': 0, 'max': 1},
                lambda_term={'value': 0.2284, 'min': 0, 'max': 1}
            ),
        ),
        (
            LinearDiffusivityPDE(),
            create_params(
                diffusivity={'value': 0.2636, 'min': 0, 'max': 20},
                mu={'value': 0.0200, 'min': -10, 'max': 10}
            )
        ),
        (
            SigmoidDiffusivityPDE(),
            create_params(
                diffusivity={'value': 13.2528, 'min': -20, 'max': 20},
                mu={'value': 0.2500, 'min': -10, 'max': 10},
                beta={'value': 3.9847, 'min': -beta_bound, 'max': beta_bound}
            )
        )
    ]

    target_frame = 36

    with Pool(processes=4) as pool:
        pool.starmap(functools.partial(_async_optimize_pdes_to_files, pdes=pdes, target_frame=target_frame), files[-4:])


def main():
    # directories = ['G:\\rost\\Ca2+_laser\\raw_data', 'G:\\rost\\kozep\\raw_data', 'G:\\rost\\sarok\\raw_data']
    #
    # # directories = ['G:\\rost\\sarok\\raw_data']
    #
    # darr_files = [
    #     'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec006.nd2',
    #     'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec007.nd2',
    #     'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec008.nd2',
    #     'G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl017.nd2',
    #     'G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl018.nd2',
    #     'G:\\rost\\sarok\\raw_data\\1472_4_laser@30sec001.nd2',
    # ]
    #
    #
    # def darrs():
    #     for file in darr_files:
    #         darr = DiffusionArray(file).channel(0)
    #         start_frame = Analyzer(darr).detect_diffusion_start_frame()
    #         darr = darr.background_removed(f'{start_frame - 3}:{start_frame}')
    #         yield darr
    #
    #
    # plot_starting_place_finder_comparisons(darrs(), thin_margins=True, length=6)

    # -------------------------------------------------

    filename = 'G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec006.nd2'
    darr_w_bcg = DiffusionArray(filename).channel(0).normalized()
    analyzer = Analyzer(darr_w_bcg)
    start_frame = analyzer.detect_diffusion_start_frame()
    darr = darr_w_bcg.background_removed(f'{start_frame - 3}:{start_frame}')
    analyzer = Analyzer(darr)

    # plot_start_neighbourhood(darr, start_frame=start_frame, thin_margins=True, show_start_place=False)

    # percentile = analyzer.apply_for_each_frame(functools.partial(np.percentile, q=100))
    # plt.figure(figsize=(6, 6), dpi=100)
    # plt.title('max intensities by frame')
    # plt.xlabel('frame')
    # plt.ylabel('intensity')
    # plt.plot(percentile)
    # plt.show()

    # plot_starting_place_finder_comparisons(darr)

    # sums = analyzer.apply_for_each_frame(np.sum, normalize=True)
    # plot_fit(sums, start_frame, np.argmax(sums))

    plot_different_homogenization_centers(darr, darr_w_bcg)


if __name__ == '__main__':
    main()
