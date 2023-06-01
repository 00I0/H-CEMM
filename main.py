import numpy as np
from matplotlib import pyplot as plt

from diffusion_array import DiffusionArray
# it's just a playground... doesn't do anything related to the project
from downloader import Downloader


def main():
    downloader = Downloader.from_json('1LOnjyZYbOBwZIqzFnwx3sJf9HM7J1A1R')
    print(downloader.list_file_names())
    darr = DiffusionArray('a.nd2')
    # darr.save('super_1472_5_laser_EC1flow_laserabl010.npz')

    # with nd2.ND2File('super_1472_5_laser_EC1flow_laserabl010.nd2') as nd_file:

    # nd_file
    # print(nd_file.metadata)
    # for i in range(nd_file.metadata.contents.frameCount):
    # frame channel
    # print(nd_file.asarray()[0][0][400][30])

    # print(darr.ndarray[0, :, :, :].shape())
    # print(darr.frame(0)[:].shape)

    print(darr.number_of_channels())
    print(darr[:].dtype)
    # plt.imshow(darr.frame(41).channel(0))
    # plt.show()

    darr = darr.channel(0)
    print(np.mean(darr.frame('0:6'), axis=0).shape)
    plt.imshow(darr.frame(20)[:] - np.mean(darr.frame('0:6'), axis=0))
    plt.show()


if __name__ == '__main__':
    main()
