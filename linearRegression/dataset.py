import torch
import torch.utils
from torch.utils.data import Dataset
from random import *
import numpy as np


def innerProduct(list1, list2):
    result = 0
    for i in range(len(list1)):
        result += list1[i]*list2[i]
    return result


def makeDataset(parameterList, numData):
    size = len(parameterList)
    xData = [[randint(1, 100) for j in range(size-1)]
             for i in range(numData)]
    yData = [[innerProduct(randomData, parameterList[:size-1]) + parameterList[size-1]]
             for randomData in xData]
    return (xData, yData)


class TrainDataset(Dataset):
    def __init__(self):
        (xData, yData) = makeDataset([1, 7, 8, 10], 100)
        print(xData, yData)
        self.x = xData
        self.y = yData

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y
