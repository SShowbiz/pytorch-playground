import torch
import os
from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train():
    numEpoch = 20

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    numBatch = len(dataloader)

    linearRegressionModel = LinearRegression(3)
    optimizer = torch.optim.SGD(linearRegressionModel.parameters(), lr=1e-5)

    loss = nn.MSELoss()
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            # print(batch)
            # print(data)
            x, y = data
            batchLoss = loss(linearRegressionModel(x), y)

            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()
            print('EPOCH: {}/{} BATCH: {}/{} MSEloss: {:.6f}'.format(epoch,
                                                                     numEpoch, batch, numBatch, batchLoss.item()))

    input = torch.FloatTensor([[73, 80, 75]])
    output = linearRegressionModel(input)

    print('input이 {}일 때 output이 {}입니다'.format(input, output))
    torch.save(linearRegressionModel,
               './linearRegression/saveModel')


train()
