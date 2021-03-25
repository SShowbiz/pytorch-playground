import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiLayerPerceptron = nn.Sequential(
            nn.Linear(2, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):  # forward propagation
        return self.multiLayerPerceptron(x)
