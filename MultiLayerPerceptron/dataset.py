import torch
import torch.utils
from torch.utils.data import Dataset
from random import *
import numpy as np


class TrainDataset(Dataset):
    def __init__(self):
        xData = [[0, 0], [0, 1], [1, 0], [1, 1]]
        yData = [[0], [1], [1], [0]]
        self.x = xData
        self.y = yData

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y
