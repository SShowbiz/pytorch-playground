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
    batchSize = 4

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    numBatch = len(dataloader)

    LogisticRegressionModel = LogisticRegression(3)
    optimizer = torch.optim.SGD(LogisticRegressionModel.parameters(), lr=1e-5)

    loss = nn.BCELoss()
    correct_number = 0.0
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            x, y = data
            hypothesis = LogisticRegressionModel(x)
            batchLoss = loss(hypothesis, y)

            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()

            prediction = hypothesis >= torch.FloatTensor([0.5])
            correct_prediction = prediction.float() == y
            correct_number += correct_prediction.sum()
            accuracy = correct_number / \
                ((epoch+1)*batchSize*numBatch + (batch+1)*batchSize) * 100
            print('EPOCH: {}/{} BATCH: {}/{} BCEloss: {:.6f} ACCURACY: {}%'.format(epoch,
                                                                                   numEpoch, batch, numBatch, batchLoss.item(), accuracy))
    for name, param in LogisticRegressionModel.named_parameters():
        if param.requires_grad:
            print(name, param.data)
