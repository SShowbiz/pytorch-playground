import torch
import os
from linearRegression.dataset import TrainDataset
from linearRegression.model import LinearRegression
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train():
    numEpoch = 100

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    numBatch = len(dataloader)

    linearRegressionModel = LinearRegression(3)
    optimizer = torch.optim.SGD(linearRegressionModel.parameters(), lr=1e-5)

    loss = nn.MSELoss()
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            x, y = data
            batchLoss = loss(linearRegressionModel(x), y)

            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()
            print('EPOCH: {}/{} BATCH: {}/{} MSEloss: {:.6f}'.format(epoch,
                                                                     numEpoch, batch, numBatch, batchLoss.item()))
    for name, param in linearRegressionModel.named_parameters():
        if param.requires_grad:
            print(name, param.data)
