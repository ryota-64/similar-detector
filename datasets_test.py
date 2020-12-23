import torch
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

from similar_detector.config.config import Config
from similar_detector.datasets import DataSet



if __name__ == '__main__':
    opt = Config()
    print(opt.train_root, opt.train_list)
    train_dataset = DataSet(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=opt.train_batch_size,
                                              shuffle=True,
                                              num_workers=opt.num_workers)
    to_img = T.ToPILImage()

    for ii, train_batch in enumerate(trainloader):
        data_input, label = train_batch


        print(data_input.shape, label)

        layer = np.array(to_img(data_input[0][1:2, :, :]))
        print(layer.shape)
        plt.imshow(layer)
        plt.show()
        # for i in range(data_input[0].shape[0]):
        #     layer = np.array(to_img(data_input[0][i:i+1, :, :]))
        #     print(layer.shape)
        #     plt.imshow(layer)
        #     plt.show()
