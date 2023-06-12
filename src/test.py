import os

import numpy as np
from lmfit import Model
from matplotlib import pyplot as plt

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
            npz_filename = os.path.splitext(filename)[0] + "_homogenized_avg.npz"
            npz_filepath = os.path.join(directory, npz_filename)

            darr = DiffusionArray(file_path).channel(0)
            darr = darr.update_ndarray(darr - np.mean(darr.frame('0:3'), axis=0))
            analyzer = Analyzer(darr)
            start_place = analyzer.detect_diffusion_start_place()
            copy = darr.copy()

            size = darr.channel(0).frame(0).shape[0]  # assuming square
            max_distance = max(start_place[0], start_place[1], size - start_place[0], size - start_place[1])
            max_distance = int((max_distance + 5) * 2 ** (1 / 2))
            for i in range(0, max_distance, diff):
                if i % 100 == 0:
                    print(f'{filename:40s}--{i:4d}/{max_distance}')
                ring_mask = Mask.ring(darr.shape, start_place, i, i + diff)
                avg = analyzer.apply_for_each_frame(np.average, mask=ring_mask)
                copy[ring_mask] = avg

            copy.save(npz_filepath)


def plot_start_neighbourhood(darr: DiffusionArray, start_frame=None, start_place=None):
    analyzer = Analyzer(darr.channel(0))
    if start_frame is None:
        start_frame = analyzer.detect_diffusion_start_frame()
    if start_place is None:
        start_place = analyzer.detect_diffusion_start_place()

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    axes[1].imshow(darr.channel(0).frame(start_frame))
    axes[1].scatter(start_place[1], start_place[0], color='red')
    axes[1].tick_params(label1On=False, tick1On=False)
    axes[1].set_title('detected start')

    axes[0].imshow(darr.channel(0).frame(start_frame - 1))
    axes[0].tick_params(label1On=False, tick1On=False)
    axes[0].set_title('start - 1')
    axes[2].imshow(darr.channel(0).frame(start_frame + 1))
    axes[2].tick_params(label1On=False, tick1On=False)
    axes[2].set_title('start + 1')
    # plt.subplots_adjust(wspace=0.4)
    plt.show()


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


def main():
    # downloader = Downloader.from_json('1NEfEFK86jqWvNdPuLStjOi18dH9P1faN')
    # print(downloader.list_file_names())
    # downloader.download_file('super_1472_5%laser_EC1flow_laserabl017.nd2')
    # meta = downloader.file_meta_for('super_1472_5%laser_EC1flow_laserabl017.nd2')
    # darr = DiffusionArray(meta)
    # Sum of intensities by frames

    filename = 'G:\\rost\\Ca2+_laser\\1133_3_laser@30sec006_homogenized_avg.npz'
    # generate_homogenized_npz(directory='G:\\rost\\sarok')
    # filename = 'G:\\rost\\kozep\\super_1472_5_laser_EC1flow_laserabl017.nd2'
    # filename = 'G:\\rost\\sarok\\1472_4_laser@30sec004.nd2'
    darr = DiffusionArray(filename)
    # best for homo: 008 & 006   |   018 & 017    |    003 & 004

    darr = darr.channel(0)

    plt.imshow(darr.frame(14))
    plt.show()

    # darr = darr.update_ndarray(darr - np.mean(darr.frame('0:3'), axis=0))

    analyzer = Analyzer(darr)
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()
    # plt.show()
    # quarter_mask = Mask.bottom_right_quarter(darr.shape, start_place)
    # darr[quarter_mask] = 4444
    # plt.scatter(start_place[1], start_place[0], color='red')
    # plt.imshow(darr.frame(start_frame))

    plt.title('asd')

    # import cupy as np
    # copy = darr.copy()
    # for i in range(0, 300, 2):
    #     if i % 50 == 0:
    #         print(i)
    #     ring_mask = Mask.ring(darr.shape, start_place, i, i + 2)
    #     avg = analyzer.apply_for_each_frame(np.average, mask=ring_mask)
    #     copy[ring_mask] = avg

    # plot_start_neighbourhood(copy, start_frame=start_frame, start_place=start_place)
    # plot_start_neighbourhood(darr, start_frame=start_frame, start_place=start_place)
    #
    fig, axs = plt.subplots(1, 2)

    maxes = analyzer.apply_for_each_frame(np.max)
    sums = analyzer.apply_for_each_frame(np.sum)
    axs[0].plot(maxes[start_frame:])
    axs[1].plot(sums[start_frame:])

    axs[0].set_title('Max by frame')
    axs[1].set_title('Sum by frame')
    plt.title(filename)
    plt.show()


if __name__ == '__main__':
    main()

# analyzer = Analyzer(data.channel(0))
# start_place = analyzer.detect_diffusion_start_place()
# start_frame = analyzer.detect_diffusion_start_frame()
#
#
# def biggest_jump(x, **kwargs):
#   sorted_array = np.sort(x, **kwargs)
#   max_jump_idx = np.argmax(np.diff(sorted_array, **kwargs), **kwargs)
#   return sorted_array[np.arange(sorted_array.shape[0]), max_jump_idx]
#
# circle_mask = analyzer.circle_mask(start_place, 300)
# cut_ofs = analyzer.apply_for_each_frame(np.mean, mask=circle_mask)
# cell_mask = analyzer.cell_mask(circle_mask, cut_ofs)
# arr = np.array(data.channel(0))
#
# fig, ax = plt.subplots(figsize=(10, 8))
#
# ax.imshow(arr[start_frame + 4])
# ax.imshow(cell_mask[start_frame + 4], cmap='jet', alpha=.3)
# plt.show()
