import torch
import os
from SoftmaxRegression.dataset import TrainDataset
from SoftmaxRegression.model import SoftmaxRegression
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train():
    numEpoch = 100
    batchSize = 4

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    numBatch = len(dataloader)

    SoftmaxRegressionModel = SoftmaxRegression(3, 3)
    optimizer = torch.optim.SGD(SoftmaxRegressionModel.parameters(), lr=1e-2)

    loss = nn.NLLLoss()
    correct_number = 0.0
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            x, y = data
            hypothesis = SoftmaxRegressionModel(x)

            batchLoss = loss(hypothesis, y.max(dim=1)[1])

            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()

            correct_prediction = hypothesis.max(dim=1)[1] == y.max(dim=1)[1]

            correct_number += correct_prediction.sum()
            accuracy = correct_number / \
                ((epoch+1)*batchSize*numBatch + (batch+1)*batchSize) * 100
            print('EPOCH: {}/{} BATCH: {}/{} BCEloss: {:.6f} ACCURACY: {}%'.format(epoch,
                                                                                   numEpoch, batch, numBatch, batchLoss.item(), accuracy))
    for name, param in SoftmaxRegressionModel.named_parameters():
        if param.requires_grad:
            print(name, param.data)
