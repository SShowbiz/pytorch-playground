import torch
import os
from logisticRegression.dataset import TrainDataset
from logisticRegression.model import LogisticRegression
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train():
    numEpoch = 100

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    numBatch = len(dataloader)

    LogisticRegressionModel = LogisticRegression(3)
    optimizer = torch.optim.SGD(LogisticRegressionModel.parameters(), lr=1e-5)

    loss = nn.BCELoss()
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            x, y = data
            batchLoss = loss(LogisticRegressionModel(x), y)

            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()
            print('EPOCH: {}/{} BATCH: {}/{} BCEloss: {:.6f}'.format(epoch,
                                                                     numEpoch, batch, numBatch, batchLoss.item()))
    for name, param in LogisticRegressionModel.named_parameters():
        if param.requires_grad:
            print(name, param.data)
