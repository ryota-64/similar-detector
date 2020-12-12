import torch

from config import Config
from data import DataSet


if __name__ == '__main__':
    opt = Config()
    print(opt.train_root, opt.train_list)
    train_dataset = DataSet(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=opt.train_batch_size,
                                              shuffle=True,
                                              num_workers=opt.num_workers)

    for ii, train_batch in enumerate(trainloader):
        data_input, label = train_batch
        print(data_input.shape, label)
