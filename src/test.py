import os
import re
import warnings
from math import ceil, sqrt

import numpy as np
from lmfit import Model
from matplotlib import pyplot as plt, gridspec, patches

from analyzer import Analyzer
from diffusion_array import DiffusionArray
# it's just a playground... doesn't do anything related to the project
from reader import ND2Reader
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


def generate_homogenized_npz(directory: str, diff=2):
    for filename in os.listdir(directory):
        if filename.endswith(".nd2"):
            file_path = os.path.join(directory, filename)
            npz_filename = os.path.splitext(filename)[0] + "_homogenized_avg_centroid_cell.npz"
            npz_filepath = os.path.join(directory, npz_filename)

            darr = DiffusionArray(file_path).channel(0)
            analyzer = Analyzer(darr)
            start_place = analyzer.detect_diffusion_start_place()
            start_frame = analyzer.detect_diffusion_start_frame()
            darr_wo_bcg = darr.updated_ndarray(darr - np.mean(darr.frame('0:3'), axis=0))
            analyzer = Analyzer(darr_wo_bcg)
            copy = darr_wo_bcg.copy()

            size = darr.channel(0).frame(0).shape[0]  # assuming square
            ordered = sorted([start_place[0], start_place[1], size - start_place[0], size - start_place[1]])
            max_distance = ceil(sqrt(ordered[-1] ** 2 + ordered[-2] ** 2))
            cell_mask = Mask.cell(darr_wo_bcg[:], np.percentile(darr.frame(start_frame), 66)).for_frame(start_frame + 1)
            for i in range(0, max_distance, diff):
                if i % 100 == 0:
                    print(f'{filename:40s}--{i:4d}/{max_distance}')
                ring_mask = Mask.ring(darr.shape, start_place, i, i + diff)
                combined_mask = ring_mask & cell_mask
                combined_mask = ring_mask

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    avg = np.nan_to_num(np.nanmean(copy[combined_mask]), nan=0).astype(int)
                    copy[ring_mask] = avg

            copy.save(npz_filepath)


def plot_start_neighbourhood(darr: DiffusionArray, start_frame=None, start_place=None, super_title=None):
    analyzer = Analyzer(darr.channel(0))
    if start_frame is None:
        start_frame = analyzer.detect_diffusion_start_frame()
    if start_place is None:
        start_place = analyzer.detect_diffusion_start_place()

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    vmax = max(darr.frame(start_frame - 1)[:].max(),
               darr.frame(start_frame)[:].max(),
               darr.frame(start_frame + 1)[:].max()
               )

    vmin = min(darr.frame(start_frame - 1)[:].min(),
               darr.frame(start_frame)[:].min(),
               darr.frame(start_frame + 1)[:].min()
               )

    axes[1].imshow(darr.channel(0).frame(start_frame), vmax=vmax, vmin=vmin)
    axes[1].scatter(start_place[1], start_place[0], color='red', alpha=.5)
    axes[1].tick_params(label1On=False, tick1On=False)
    axes[1].set_title('detected start')

    axes[0].imshow(darr.channel(0).frame(start_frame - 1), vmax=vmax, vmin=vmin)
    axes[0].tick_params(label1On=False, tick1On=False)
    axes[0].set_title('start - 1')
    axes[2].imshow(darr.channel(0).frame(start_frame + 1), vmax=vmax, vmin=vmin)
    axes[2].tick_params(label1On=False, tick1On=False)
    axes[2].set_title('start + 1')
    axes[2].scatter(start_place[1], start_place[0], color='red', alpha=.5)
    plt.subplots_adjust(wspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05)

    if super_title is not None:
        fig.suptitle(super_title)
    # plt.show()


def plot_exp_fit(darr):
    analyzer = Analyzer(darr.channel(0))
    start_frame = analyzer.detect_diffusion_start_frame()
    maxes = analyzer.apply_for_each_frame(np.max, remove_background=True, normalize=True)

    exponential_decay = lambda x, a, b: a * np.exp(-b * x)

    model = Model(exponential_decay)
    y = maxes[np.argmax(maxes):]
    x = np.arange(len(y)) + np.argmax(maxes)
    params = model.make_params(a=2.0, b=0.3)
    params['a'].min = 0
    params['b'].min = 0
    result = model.fit(y, params, x=x)

    r_squared_text = f'R-squared: {result.rsquared:.4f}\n a: {result.best_values["a"]:.2f}, b: {result.best_values["b"]:.2f}'
    plt.plot(maxes, 'bo', label='Data')
    plt.plot(x, result.best_fit, 'r-', label='Best Fit')
    plt.axvline(x=start_frame, label='Start Frame', color='green')
    plt.legend(title=r_squared_text)
    plt.show()


def plot_start(directories):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4)
    i = 0
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".nd2"):
                print('processing: ', filename)
                file_path = os.path.join(directory, filename)
                darr = DiffusionArray(file_path).channel(0)

                analyzer = Analyzer(darr.channel(0))

                ax1 = plt.subplot(gs[i // 4, (i % 4)])
                ax1.tick_params(label1On=False, tick1On=False)
                ax1.set_title(filename)

                # --

                start_frame_number = analyzer.detect_diffusion_start_frame()
                frame = darr.frame(start_frame_number + 1)
                frame = frame - np.mean(darr.frame(slice(start_frame_number - 1, start_frame_number + 1)), axis=0)
                ax1.imshow(frame)

                diff_x, diff_y = analyzer.detect_diffusion_start_place(strategy='biggest-difference')
                ax1.scatter(diff_x, diff_y, color='red', alpha=.5)

                center_x, center_y = analyzer.detect_diffusion_start_place(strategy='connected-components')
                ax1.scatter(center_x, center_y, color='orange', alpha=.5)

                closest_x, closest_y = analyzer.detect_diffusion_start_place(strategy='connected-components',
                                                                             use_inner=True)
                ax1.scatter(closest_x, closest_y, color='pink', alpha=.5)

                old_start_x, old_start_y = analyzer.detect_diffusion_start_place(strategy='weighted-centroid')
                ax1.scatter(old_start_x, old_start_y, color='green', alpha=.5)

                # --

                i = i + 1

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)
    title = 'cc centroids'
    fig.suptitle(title)
    fig.legend(
        [patches.Circle((0, 0), radius=0.2, facecolor=c) for c in ('red', 'orange', 'pink', 'green')],
        ['biggest-difference', 'connected-components', 'connected-components_inner', 'weighted-centroid'],
        loc='lower right'
    )
    plt.savefig(f'{title}.pdf')
    plt.show()


def plot_homogenized(directories):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4)
    i = 0
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".npz") \
                    and 'centroid' in filename and 'homogenized' in filename and 'cell' not in filename:
                print('processing: ', filename)
                file_path = os.path.join(directory, filename)
                darr = DiffusionArray(file_path).channel(0)

                analyzer = Analyzer(darr.channel(0))

                ax1 = plt.subplot(gs[i // 4, (i % 4)])
                ax1.tick_params(label1On=False, tick1On=False)

                new_filename = re.sub(r'_homogenized.*$', '', filename)

                ax1.scatter(*Analyzer(DiffusionArray(os.path.join(directory, new_filename) + '.nd2').channel(0))
                            .detect_diffusion_start_place(),
                            color='red')
                ax1.set_title(new_filename)

                start_frame_number = analyzer.detect_diffusion_start_frame()
                frame = darr.frame(start_frame_number + 1)
                frame = frame - np.mean(darr.frame(slice(start_frame_number - 1, start_frame_number + 1)), axis=0)
                ax1.imshow(frame)

                i = i + 1

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.94, bottom=0.01)
    title = 'homogenized on centroids'
    fig.suptitle(title)
    plt.savefig(f'{title}.pdf')
    plt.show()


def plot_hom_different_starts(darr, diff):
    analyzer = Analyzer(darr)
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()

    darr_wo_bcg = darr.updated_ndarray(darr - np.mean(darr.frame('0:3'), axis=0))

    analyzer = Analyzer(darr_wo_bcg)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(24, 32))
    axes = [axs[0][0], axs[0][1]] + [axs[i][j] for i in range(1, 4) for j in range(0, 3)]

    ax = axs[0][0]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(darr.frame(start_frame + 1))
    ax.scatter(*start_place, color='red')

    ax = axs[0][1]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(darr_wo_bcg.frame(start_frame + 1))

    ax.scatter(*start_place, color='red')

    ax = axs[0][2]
    copy = darr_wo_bcg.copy()
    size = darr_wo_bcg.channel(0).frame(0).shape[0]
    ordered = sorted([start_place[0], start_place[1], size - start_place[0], size - start_place[1]])
    max_distance = ceil(sqrt(ordered[-1] ** 2 + ordered[-2] ** 2))
    for j in range(0, max_distance, diff):
        if j % 100 == 0:
            print(f'{j:4d}/{max_distance}')
        ring_mask = Mask.ring(darr.shape, start_place, j, j + diff)
        avg = analyzer.apply_for_each_frame(np.average, mask=ring_mask)
        copy[ring_mask] = avg

    ax.tick_params(label1On=False, tick1On=False)
    im_max = np.max(copy[start_frame + 1])
    im_min = np.min(copy[start_frame + 1])
    ax.imshow(copy[start_frame + 1], vmax=im_max, vmin=im_min)
    ax.scatter(*start_place, color='red')

    for i in range(2, 11):
        ax = axes[i]

        copy = darr_wo_bcg.copy()

        size = darr_wo_bcg.channel(0).frame(0).shape[0]  # assuming square
        start_place = (256 * ((i - 2) % 3 + 1), 256 * ((i - 2) // 3 + 1))

        ordered = sorted([start_place[0], start_place[1], size - start_place[0], size - start_place[1]])
        max_distance = ceil(sqrt(ordered[-1] ** 2 + ordered[-2] ** 2))
        for j in range(0, max_distance, diff):
            if j % 100 == 0:
                print(f'{j:4d}/{max_distance}')
            ring_mask = Mask.ring(darr.shape, start_place, j, j + diff)
            avg = analyzer.apply_for_each_frame(np.average, mask=ring_mask)
            copy[ring_mask] = avg

        ax.tick_params(label1On=False, tick1On=False)
        ax.imshow(copy[start_frame + 1], vmax=im_max, vmin=im_min)
        ax.scatter(*start_place, color='red')

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)
    title = 'homogenization different starts'
    fig.suptitle(title)
    plt.savefig(f'{title}.pdf')
    plt.show()


def plot_cell_hom_different_starts(darr, diff):
    analyzer = Analyzer(darr)
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()

    darr_wo_bcg = darr.updated_ndarray(darr - np.mean(darr.frame('0:3'), axis=0))

    analyzer = Analyzer(darr_wo_bcg)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(24, 32))
    axes = [axs[0][0], axs[0][1]] + [axs[i][j] for i in range(1, 4) for j in range(0, 3)]

    ax = axs[0][0]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(darr.frame(start_frame + 1))
    ax.scatter(*start_place, color='red')

    ax = axs[0][1]
    ax.tick_params(label1On=False, tick1On=False)
    ax.imshow(darr_wo_bcg.frame(start_frame + 1))

    ax.scatter(*start_place, color='red')

    ax = axs[0][2]
    copy = darr_wo_bcg.copy()
    size = darr_wo_bcg.channel(0).frame(0).shape[0]
    ordered = sorted([start_place[0], start_place[1], size - start_place[0], size - start_place[1]])
    max_distance = ceil(sqrt(ordered[-1] ** 2 + ordered[-2] ** 2))
    cell_mask = Mask.cell(darr_wo_bcg[:], np.percentile(darr.frame(start_frame), 66)).for_frame(start_frame + 1)
    for j in range(0, max_distance, diff):
        if j % 100 == 0:
            print(f'{j:4d}/{max_distance}')
        ring_mask = Mask.ring(darr.shape, start_place, j, j + diff)
        combined_mask = ring_mask & cell_mask

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            avg = np.nan_to_num(np.nanmean(copy[combined_mask]), nan=0).astype(int)
            copy[ring_mask] = avg

    ax.tick_params(label1On=False, tick1On=False)
    im_max = np.max(copy[start_frame + 1])
    im_min = np.min(copy[start_frame + 1])
    ax.imshow(copy[start_frame + 1], )
    ax.scatter(*start_place, color='red')

    for i in range(2, 11):
        ax = axes[i]

        copy = darr_wo_bcg.copy()

        size = darr_wo_bcg.channel(0).frame(0).shape[0]  # assuming square
        start_place = (256 * ((i - 2) % 3 + 1), 256 * ((i - 2) // 3 + 1))

        ordered = sorted([start_place[0], start_place[1], size - start_place[0], size - start_place[1]])
        max_distance = ceil(sqrt(ordered[-1] ** 2 + ordered[-2] ** 2))
        cell_mask = Mask.cell(darr_wo_bcg[:], np.percentile(darr.frame(start_frame), 66)).for_frame(start_frame + 1)
        for j in range(0, max_distance, diff):
            if j % 100 == 0:
                print(f'{j:4d}/{max_distance}')
            ring_mask = Mask.ring(darr.shape, start_place, j, j + diff)
            combined_mask = ring_mask & cell_mask

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                avg = np.nan_to_num(np.nanmean(copy[combined_mask]), nan=0).astype(int)
                copy[ring_mask] = avg

        ax.tick_params(label1On=False, tick1On=False)
        ax.imshow(copy[start_frame + 1], vmax=im_max, vmin=im_min)
        ax.scatter(*start_place, color='red')

    fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.96, bottom=0.01)
    title = 'cell homogenization different starts'
    fig.suptitle(title)
    plt.savefig(f'{title}.pdf')
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

                cut = Mask.cell(darr_wo_bcg[:], np.percentile(darr.frame(start_frame), 66))

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
    plt.savefig('inner vs outer radii.pdf')
    plt.show()


def main():
    # generate_homogenized_npz(directory='G:\\rost\\kozep')
    # generate_homogenized_npz(directory='G:\\rost\\Ca2+_laser')
    # generate_homogenized_npz(directory='G:\\rost\\sarok')
    directories = ['G:\\rost\\Ca2+_laser', 'G:\\rost\\kozep', 'G:\\rost\\sarok']
    # filename = 'G:\\rost\\Ca2+_laser\\1133_3_laser@30sec008.nd2'
    # filename = 'G:\\rost\\kozep\\super_1472_5_laser_EC1flow_laserabl017.nd2'  # 17 dead w/ 50%   10 dead w/ 5%
    # filename = 'G:\\rost\\sarok\\1472_4_laser@30sec004.nd2'  # 001 dead if inverted 60%

    diff = 2
    r = 300

    # darr = DiffusionArray(filename)
    # darr = darr.channel(0)

    # plot_inner_radi(filename)

    plot_homogenized(directories)


if __name__ == '__main__':
    main()
