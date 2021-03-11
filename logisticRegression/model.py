import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, n):  # n 개의 인자를 가진 다중 선형 회귀의 이중 분류를 구현
        super().__init__()
        self.linearRegressionModel = nn.Linear(n, 1, bias=True)
        self.logisticRegressionModel = nn.Sigmoid()

    def forward(self, x):  # forward propagation
        return self.logisticRegressionModel(self.linearRegressionModel(x))
