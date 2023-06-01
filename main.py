from matplotlib import pyplot as plt

from diffusion_array import DiffusionArray


# it's just a playground... doesn't do anything related to the project
def main():
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
    print(darr[:].dtype)
    plt.imshow(darr.frame(0).channel(0))
    plt.show()

    # plt.imshow(nd_file.asarray()[7, 0, :] - np.mean(nd_file.asarray()[:7, 0, :], axis=0))
    # plt.show()


if __name__ == '__main__':
    main()
