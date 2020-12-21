import enum

import numpy as np
import matplotlib.pyplot as plt


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
        fig = plt.figure(figsize=(20, 10))
        axes = {}
        if channel is None:
            for i in range(a.shape[2]):

                axes[i] = fig.add_subplot(2, 3, i + 1)  # 2行3列の1番目
                data = a[:, :, i]
                axes[i].imshow(data)
            fig.show()

        else:
            plt.imshow(a[:,:,channel])
            plt.show()
