import os

import numpy as np
from lmfit import Model
from matplotlib import pyplot as plt

from analyzer import Analyzer
from diffusion_array import DiffusionArray
# it's just a playground... doesn't do anything related to the project
from reader import ND2Reader


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


def plot_start_neighbourhood(darr: DiffusionArray):
    analyzer = Analyzer(darr.channel(0))
    place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[1].imshow(darr.channel(0).frame(start_frame))
    axes[1].scatter(place[1], place[0], color='red')
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
    result = model.fit(y, params, x=x)

    # print(result.fit_report()['R-squared'])

    r_squared_text = f'R-squared: {result.rsquared:.4f}\n a: {result.best_values["a"]:.2f}, b: {result.best_values["b"]:.2f}'
    plt.plot(maxes, 'bo', label='Data')
    plt.plot(x, result.best_fit, 'r-', label='Best Fit')
    plt.axvline(x=start_frame, label='Start Frame')
    plt.legend(title=r_squared_text)
    plt.show()


def main():
    # downloader = Downloader.from_json('1NEfEFK86jqWvNdPuLStjOi18dH9P1faN')
    # print(downloader.list_file_names())
    # Sum of intensities by frames

    # darr = DiffusionArray('G:\\rost\\Ca2+_laser\\1133_3_laser@30sec008.nd2')
    # super_1472_5_laser_EC1flow_laserabl010.nd2
    darr = DiffusionArray('G:\\rost\\kozep\\super_1472_5_laser_EC1flow_laserabl017.nd2')
    # 1472_4_laser@30sec001.nd2
    # darr = DiffusionArray('G:\\rost\\sarok\\1472_4_laser@30sec003.nd2')

    # darrs = []
    # directories = ['G:\\rost\\Ca2+_laser', 'G:\\rost\\kozep', 'G:\\rost\\sarok']
    # directories = ['G:\\rost\\Ca2+_laser']
    # for directory in directories:
    #     for filename in os.listdir(directory):
    #         if filename.endswith(".nd2"):
    #             print(filename)
    #             file_path = os.path.join(directory, filename)
    #             darr = DiffusionArray(file_path)
    #             analyzer = Analyzer(darr.channel(0))
    #             start_frame = analyzer.detect_diffusion_start_frame()
    #             maxes = analyzer.apply_for_each_frame(np.sum, remove_background=True, normalize=True)
    #             plt.plot(maxes[start_frame:])
    #
    # plot_start_neighbourhood(darr)

    analyzer = Analyzer(darr.channel(0))
    start_place = analyzer.detect_diffusion_start_place()
    start_frame = analyzer.detect_diffusion_start_frame()
    maxes = analyzer.apply_for_each_frame(np.max, remove_background=True, normalize=True)

    # print(result.fit_report()['R-squared'])

    def biggest_jump(x, **kwargs):
        sorted_array = np.sort(x, **kwargs)
        max_jump_idx = np.argmax(np.diff(sorted_array, **kwargs), **kwargs)
        return sorted_array[np.arange(sorted_array.shape[0]), max_jump_idx]

    circle_mask = analyzer.circle_mask(start_place, 200)
    cut_ofs = analyzer.apply_for_each_frame(np.mean, mask=circle_mask)
    cell_mask = analyzer.cell_mask(circle_mask, cut_ofs)
    arr = np.array(darr.channel(0))

    fig, ax = plt.subplots()

    ax.imshow(arr[start_frame + 4])
    ax.imshow(cell_mask[start_frame + 4], cmap='jet', alpha=0.3)
    plt.show()


if __name__ == '__main__':
    main()
