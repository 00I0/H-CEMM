import math
import os
import re
from math import ceil, sqrt

import numpy as np
from lmfit import Model
from matplotlib import pyplot as plt, gridspec, patches
from matplotlib.axes import Axes
from typing import List, Sequence, Tuple

from analyzer import Analyzer
from diffusion_array import DiffusionArray
from reader import ND2Reader
from src.homogenizer import Homogenizer
from src.mask import Mask


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


def plot_start_neighbourhood(darr: DiffusionArray, start_frame=None, start_place=None, title=None):
    analyzer = Analyzer(darr.channel(0))
    if start_frame is None:
        start_frame = analyzer.detect_diffusion_start_frame()
    if start_place is None:
        start_place = analyzer.detect_diffusion_start_place()

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    v_max = max(darr.frame(start_frame - 1)[:].max(),
                darr.frame(start_frame)[:].max(),
                darr.frame(start_frame + 1)[:].max()
                )

    v_min = min(darr.frame(start_frame - 1)[:].min(),
                darr.frame(start_frame)[:].min(),
                darr.frame(start_frame + 1)[:].min()
                )

    axes[1].imshow(darr.channel(0).frame(start_frame), vmax=v_max, vmin=v_min)
    axes[1].scatter(start_place[1], start_place[0], color='red', alpha=.5)
    axes[1].tick_params(label1On=False, tick1On=False)
    axes[1].set_title('detected start')

    axes[0].imshow(darr.channel(0).frame(start_frame - 1), vmax=v_max, vmin=v_min)
    axes[0].tick_params(label1On=False, tick1On=False)
    axes[0].set_title('start - 1')
    axes[2].imshow(darr.channel(0).frame(start_frame + 1), vmax=v_max, vmin=v_min)
    axes[2].tick_params(label1On=False, tick1On=False)
    axes[2].set_title('start + 1')
    axes[2].scatter(start_place[1], start_place[0], color='red', alpha=.5)
    plt.subplots_adjust(wspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05)

    if title is not None:
        fig.suptitle(title)
    plt.show()


def plot_exp_fit(darr):
    analyzer = Analyzer(darr.channel(0))
    start_frame = analyzer.detect_diffusion_start_frame()
    maxes = analyzer.apply_for_each_frame(np.max, remove_background=True, normalize=True)

    def exponential_decay(x, a, b):
        a * np.exp(-b * x)

    model = Model(exponential_decay)
    y = maxes[np.argmax(maxes):]
    x = np.arange(len(y)) + np.argmax(maxes)
    params = model.make_params(a=2.0, b=0.3)
    params['a'].min = 0
    params['b'].min = 0
    result = model.fit(y, params, x=x)

    r_squared_text = (f'R-squared: {result.rsquared:.4f}\n a: {result.best_values["a"]:.2f}, '
                      f'b: {result.best_values["b"]:.2f}')
    plt.plot(maxes, 'bo', label='Data')
    plt.plot(x, result.best_fit, 'r-', label='Best Fit')
    plt.axvline(x=start_frame, label='Start Frame', color='green')
    plt.legend(title=r_squared_text)
    plt.show()


def plot_starting_place_finder_comparisons(
        diffusion_arrays: List[DiffusionArray] | DiffusionArray,
        strategies: Sequence[str] = ('biggest-difference', 'connected-components', 'weighted-centroid'),
        title: str = 'start place detection comparisons'
) -> None:
    """
    This method creates a plot to show the difference between the different start place finder algorithms of the
    Analyzer class. The method shows the start_frame with the background removed for each diffusion array. The start
    places found by the different algorithms will also be shown with a uniquely colored circle.

    Args:
        diffusion_arrays (List[DiffusionArray] | DiffusionArray): The diffusion arrays on which the algorithms should be
        compared. If a single DiffusionArray is provided it will be treated as a list with only one entry.
        strategies (Sequence[str]): The name of the different algorithms, used as the 'strategy' parameter in
        Analyzer.detect_diffusion_start_place.
        title (str): The title for the plot.

    Returns:
        None
    """

    if isinstance(diffusion_arrays, DiffusionArray):
        diffusion_arrays = [diffusion_arrays]

    colors = ['lime', 'silver', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta', 'yellow']

    # TODO analyzer.detect_diffusion_start_place(strategy='connected-components', use_inner=True)

    height = ceil((-1 + sqrt(1 + 3 * len(diffusion_arrays))) / 2)
    width = round(4 / 3 * height)

    if width * height < len(diffusion_arrays):
        height += 1
        width = round(4 / 3 * height)

    fig = plt.figure(figsize=(width * 5, height * 5))
    gs = gridspec.GridSpec(height, width)

    for i, diffusion_array in enumerate(diffusion_arrays):
        diffusion_array = diffusion_array.channel(0)

        ax: Axes = plt.subplot(gs[i // width, (i % width)])
        if len(diffusion_arrays) != 1:
            ax.set_title(diffusion_array.meta.name)
        ax.tick_params(label1On=False, tick1On=False)

        analyzer = Analyzer(diffusion_array)
        start_frame = analyzer.detect_diffusion_start_frame()
        diffusion_array = diffusion_array.background_removed()
        ax.imshow(diffusion_array.frame(start_frame))

        for color, strategy in zip(colors, strategies):
            point = analyzer.detect_diffusion_start_place(strategy=strategy)
            ax.scatter(*point, color=color, alpha=.5)

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.90, bottom=0.01)
    fig.suptitle(title)
    fig.legend(
        [patches.Circle((0, 0), radius=0.2, facecolor=c) for c in colors[:len(strategies)]],
        strategies,
        loc='lower right'
    )
    plt.tight_layout()
    plt.show()


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
    title = 'homogenized on centroids'
    fig.suptitle(title)
    plt.savefig(f'{title}.pdf')
    plt.show()


def plot_different_homogenization_centers(
        diffusion_array: DiffusionArray,
        builder: Homogenizer.Builder = Homogenizer.Builder().report_progress(False),
        title: str = 'homogenization different starts'
) -> None:
    """
    This method shows the effect of homogenizing on different centers. It creates 12 subplots to show the original
    diffusion array at the start of the process, the diffusion array with the background removed, the diffusion array
    homogenized on the detected center as well as 9 additional homogenizations each with a different homogenization
    center.

    Args:
        diffusion_array (DiffusionArray): The diffusion array on which the homogenization should be shown.
        builder (Homogenizer.Builder): A builder object to customize the homogenization process. By default, the
        progress of the homogenization will not be reported.
        title (str): The title for the plot.

    Returns:
        None
    """
    diffusion_array = diffusion_array.channel(0)
    analyzer = Analyzer(diffusion_array)
    background_removed_array = diffusion_array.background_removed()
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()
    diffusion_array = diffusion_array.frame(f'0:{start_frame + 1}')

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 24))

    ax = axs[0][0]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(diffusion_array.frame(start_frame))
    ax.scatter(*start_place, color='red')

    ax = axs[0][1]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(background_removed_array.frame(start_frame))
    ax.scatter(*start_place, color='red')

    ax = axs[0][2]
    ax.tick_params(label1On=False, tick1On=False)
    homogenizer = builder.start_frame(start_frame).center_point(start_place).remove_background().build()
    homogenized = homogenizer.homogenize(diffusion_array).frame(start_frame)
    v_min = np.min(homogenized)
    v_max = np.max(homogenized)
    ax.imshow(homogenized, vmin=v_min, vmax=v_max)
    ax.scatter(*start_place, color='red')

    test_points = [
        (round(y * diffusion_array.height / 4), round(x * diffusion_array.width / 4))
        for x in range(1, 4) for y in range(1, 4)
    ]

    center_point: tuple
    for i, center_point in enumerate(test_points):
        homogenizer = builder.start_frame(start_frame).center_point(center_point).remove_background().build()
        homogenized = homogenizer.homogenize(diffusion_array).frame(start_frame)

        ax = axs[i // 3 + 1][i % 3]
        ax.tick_params(label1On=False, tick1On=False)
        ax.imshow(homogenized, vmin=v_min, vmax=v_max)
        ax.scatter(*center_point, color='red')

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)
    fig.suptitle(title)
    plt.show()


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

                darr_wo_bcg = darr.updated_ndarray(darr - np.mean(darr.frame('0:3'), axis=0))

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
    title = 'cell masks'
    fig.suptitle(title)
    plt.savefig(f'{title}.pdf')
    plt.show()


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
    plt.savefig('inner vs outer radii (Ca2_008).pdf')
    plt.show()


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

        diffusion_array = diffusion_array.resized(
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
    homogenized_array = homogenized_array.channel(0)
    x, y = (round(coordinate) for coordinate in start_place)
    width = homogenized_array.width
    height = homogenized_array.height

    edge_distances = [x, y, width - x, height - y]
    max_distance = max(edge_distances)

    if x == max_distance:
        return homogenized_array.ndarray[:, 0, y, :x].T

    if y == max_distance:
        return homogenized_array.ndarray[:, 0, :y, x].T

    if width - x == max_distance:
        return homogenized_array.ndarray[:, 0, y, x:].T

    if height - y == max_distance:
        return homogenized_array.ndarray[:, 0, y:, x].T


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
    plt.show()


def main():
    # generate_homogenized_npz(directory='G:\\rost\\kozep')
    # generate_homogenized_npz(directory='G:\\rost\\Ca2+_laser')
    # generate_homogenized_npz(directory='G:\\rost\\sarok')

    directories = ['G:\\rost\\Ca2+_laser', 'G:\\rost\\kozep', 'G:\\rost\\sarok']
    # plot_homogenized(directories)

    filename = 'G:\\rost\\kozep\\super_1472_5_laser_EC1flow_laserabl018.nd2'
    # filename = 'G:\\rost\\Ca2+_laser\\1133_3_laser@30sec007.nd2'
    # filename = 'G:\\rost\\sarok\\1472_4_laser@30sec004.nd2'

    composite = compose_pixelwise(DiffusionArray.get_all_from_directory('G:\\rost\\sarok', regex='.*\\.nd2'))

    plot_starting_place_finder_comparisons(composite, title='Start place finders for composite')

    plot_different_homogenization_centers(composite, title='Different centers of homogenization for composite')

    analyzer = Analyzer(composite)
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()
    homogenizer = Homogenizer.Builder().remove_background().report_progress(False).center_point(
        start_place).start_frame(start_frame).build()
    homogenized = homogenizer.homogenize(composite)

    plot_concatenated_radii(concatenate_homogenized_radii(homogenized, start_place))


if __name__ == '__main__':
    main()
