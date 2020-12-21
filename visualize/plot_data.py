import enum

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize


class ConterName(enum.Enum):
    SHAPE = 0
    THICKNESS_REDUCTION_RATE = enum.auto()
    WRINKLES = enum.auto()
    VON_MISES = enum.auto()
    STRAIN = enum.auto()


class VisualizeData:

    def __init__(self):
        print(ConterName(0))
        print(ConterName(1))
        print(ConterName(2))
        print(ConterName(3))
        print(ConterName(4))

    @staticmethod
    def plot_processed_data(data_path, channel=None):
        a = np.load(data_path)
        print(a.shape)

        axes = {}
        if channel is None:
            fig = plt.figure(figsize=(20, 10))
            for i in range(a.shape[2]):
                axes[i] = fig.add_subplot(2, 3, i + 1)
                data = a[:, :, i]
                axes[i].imshow(data)
            fig.show()

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            data = a[:, :, channel]
            plt.imshow(data)
            plt.colorbar()
            plt.show()


            #
            # # mesh = ax.imshow(data)
            # x, y = np.mgrid[:data.shape[1], :data.shape[0]]
            # mappable0 = ax.pcolormesh(x, y, data.T, norm=Normalize(vmin=-1, vmax=2))  # ここがポイント！
            # pp = fig.colorbar(mappable0, ax=ax, orientation="vertical")
            # mappable0.set_clim(-1, 2)
            # fig.show()
            #
            # #
            # print(np.average(data))
            # print(np.max(data))
            # print(np.average(data))
            # new_data = np.vectorize(VisualizeData.max_1)(data)
            # plt.imshow(new_data)
            # plt.colorbar()
            # plt.show()
            # print(new_data.shape)
            # print(type(new_data))
            # print(np.max(new_data))
            # print(np.average(new_data))

    # @staticmethod
    # def max_1(input):
    #     return min(input, 1)
