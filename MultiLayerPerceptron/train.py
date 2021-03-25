import torch
import os
from MultiLayerPerceptron.dataset import TrainDataset
from MultiLayerPerceptron.model import MultiLayerPerceptron
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train():
    numEpoch = 10001
    batchSize = 4

    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    numBatch = len(dataloader)

    MultiLayerPerceptronModel = MultiLayerPerceptron()
    optimizer = torch.optim.SGD(
        MultiLayerPerceptronModel.parameters(), lr=1)

    loss = nn.BCELoss()
    correct_number = 0.0
    for epoch in range(numEpoch + 1):
        for batch, data in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = data
            hypothesis = MultiLayerPerceptronModel(x)
            batchLoss = loss(hypothesis, y)

            prediction = hypothesis >= torch.FloatTensor([0.5])
            correct_prediction = prediction.float() == y
            correct_number += correct_prediction.sum()
            accuracy = correct_number / \
                ((epoch+1)*batchSize*numBatch + (batch+1)*batchSize) * 100

            batchLoss.backward()
            optimizer.step()

            print('EPOCH: {}/{} BATCH: {}/{} BCEloss: {:.6f} ACCURACY: {}%'.format(epoch,
                                                                                   numEpoch, batch, numBatch, batchLoss.item(), accuracy))
    for name, param in MultiLayerPerceptronModel.named_parameters():
        if param.requires_grad:
            print(name, param.data)
