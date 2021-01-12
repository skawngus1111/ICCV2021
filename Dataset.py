import torch
from torch.utils.data import Dataset

import numpy as np

class CustomDataset(Dataset) :
    def __init__(self, image_size=256, transform=None):
        self.image_size = image_size
        self.transform = transform

        self.images = np.load('./dataset/test_data.npy')
        self.labels = np.load('./dataset/test_label.npy')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = image.reshape((self.image_size, self.image_size, 3))

        if self.transform is not None :
            image = self.transform(image)

        return image, label