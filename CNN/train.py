import torch
import os
from CNN.dataset import TrainDataset
from CNN.model import CNN
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train():
    numEpoch = 15
    batchSize = 100

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    numBatch = len(dataloader)

    CNNModel = CNN()
    optimizer = torch.optim.SGD(
        CNNModel.parameters(), lr=1e-3)

    loss = nn.CrossEntropyLoss()
    correct_number = 0.0
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = data
            hypothesis = CNNModel(x)
            batchLoss = loss(hypothesis, y)
            # print('1', hypothesis)
            # print(torch.argmax(hypothesis, dim=1).view(-1, 10, 1))
            # print(y)
            correct_prediction = torch.argmax(
                hypothesis, dim=1).view(100,) == y
            correct_number += correct_prediction.sum()
            accuracy = correct_number / \
                ((epoch+1)*batchSize*numBatch + (batch+1)*batchSize) * 100

            batchLoss.backward()
            optimizer.step()

            print('EPOCH: {}/{} BATCH: {}/{} BCEloss: {:.6f} ACCURACY: {}%'.format(epoch,
                                                                                   numEpoch, batch, numBatch, batchLoss.item(), accuracy))
    for name, param in CNNModel.named_parameters():
        if param.requires_grad:
            print(name, param.data)
