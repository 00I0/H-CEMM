import os

from diffusion_array import DiffusionArray
# it's just a playground... doesn't do anything related to the project
from reader import ND2Reader


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


def main():
    # downloader = Downloader.from_json('1NEfEFK86jqWvNdPuLStjOi18dH9P1faN')
    # print(downloader.list_file_names())
    darr = DiffusionArray('../data/1133_3%laser@30sec007.nd2')

    directory = 'G:\\rost\\Ca2+_laser'
    # create_files_in_directory(directory)

    # darr.save('data/super_1472_5_laser_EC1flow_laserabl010.npz')

    # with nd2.ND2File('data/super_1472_5_laser_EC1flow_laserabl010.nd2') as nd_file:

    # nd_file
    # print(nd_file.metadata)
    # for i in range(nd_file.metadata.contents.frameCount):
    # frame channel
    # print(nd_file.asarray()[0][0][400][30])

    # print(darr.ndarray[0, :, :, :].shape())
    # print(darr.frame(0)[:].shape)

    # print(darr.number_of_channels())
    # print(darr[:].dtype)
    # # plt.imshow(darr.frame(41).channel(0))
    # # plt.show()
    #
    # darr = darr.channel(0)
    # print(np.mean(darr.frame('0:6'), axis=0).shape)
    # plt.imshow(darr.frame(13)[:] - np.mean(darr.frame('0:6'), axis=0))
    # # plt.imshow(darr.frame(13))
    # plt.show()

    print(darr.channel(0).frame('0:10').shape)


if __name__ == '__main__':
    main()