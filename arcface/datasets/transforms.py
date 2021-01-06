import random

import numpy as np
from torch import torch
import torchvision.transforms as transforms


class RandomRotationTensor(torch.nn.Module):
    def __init__(self, degrees):
        super().__init__()

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.degree_range = degrees

    def __call__(self, tensor):
        degree = random.uniform(self.degree_range[0], self.degree_range[1])
        output = np.empty((0,tensor.shape[1], tensor.shape[2]))

        for i in range(tensor.shape[0]):
            layer_data = tensor[i:i + 1, :, :]
            pil_layer_data = self.to_pil(layer_data)
            rotate_layer = pil_layer_data.rotate(degree)
            output = np.concatenate([output, np.array(rotate_layer)[np.newaxis, :, :]])
        return torch.from_numpy(output)
