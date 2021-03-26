import torch
import torch.utils
from torch.utils.data import Dataset
from random import *
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class TrainDataset(Dataset):
    def __init__(self):
        self.train = dsets.MNIST(root='MNIST_data/', train=True,
                                 transform=transforms.ToTensor(), download=True)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.train[idx]
