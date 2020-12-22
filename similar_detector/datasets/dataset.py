import json
import os
import pathlib

from PIL import Image
import numpy as np
from torch.utils import data
from torchvision import transforms as T

from .transforms import RandomRotationTensor


class DataSet(data.Dataset):

    def __init__(self, root, labels_json_path, phase='train', input_shape=(1, 128, 128),
                 data_is_image=False, random_erase=True):
        self.phase = phase
        self.input_shape = input_shape
        self.data_is_image = data_is_image

        with open(labels_json_path, 'rb') as fd:
            labels_json = json.load(fd)
        data_arrays = [os.path.join(root, data_array_name) for data_array_name in labels_json['data']]
        data_arrays = np.random.permutation(data_arrays)
        print('len {}'.format(len(data_arrays)))
        self.data_arrays = [path for path in data_arrays if pathlib.Path(path).exists()]
        print('len {}'.format(len(self.data_arrays)))
        self.label_dict = labels_json['data']

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            compose_list = [
                PadingWithLongerSize(),
                T.Resize(self.input_shape[1:])] if self.data_is_image else []
            compose_list.extend([
                T.ToTensor(),
                RandomRotationTensor([-45,45]),
                normalize,
                # T.RandomErasing(),
                ])
            self.transforms = T.Compose(compose_list)

        else:
            compose_list = [
                PadingWithLongerSize(),
                T.Resize(self.input_shape[1:])] if self.data_is_image else []

            compose_list.extend([
                T.ToTensor(),
                RandomRotationTensor([-45, 45]),
                normalize,
                # T.RandomErasing(),
            ])
            self.transforms = T.Compose(compose_list)

    def __getitem__(self, index):
        data_array_path = self.data_arrays[index]

        if self.data_is_image:
            data_array = Image.open(data_array_path)
            # data_array = data_array.convert('L')
        else:
            data_array = np.load(data_array_path)
        # data_array = data_array.reshape([data_array.shape[2], data_array.shape[0], data_array.shape[1]])
        # img_data = img_data.convert('L')
        data_array = self.transforms(data_array)
        label = np.array([value for value in self.label_dict[pathlib.Path(data_array_path).name].values()])
        return data_array.float(), label

    def __len__(self):
        return len(self.data_arrays)


class PadingWithLongerSize(object):
    """
    長い方の辺を基準に正方形になるようにpaddingする
    左上が基準
    """

    def __init__(self):
        pass

    def __call__(self, image):
        """
        Args:
            image (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        size = image.size[:2]
        longer_flip = np.argmax(size)
        padding_flip = np.argmin(size)
        padding_num = size[longer_flip] - size[padding_flip]
        right_pad = padding_num if padding_flip == 0 else 0
        bottom_pad = padding_num if padding_flip == 1 else 0
        self.transform = [
            T.Pad((0, 0, right_pad, bottom_pad))
        ]
        for t in self.transform:
            image = t(image)
        return image
