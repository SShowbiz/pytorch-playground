import torch
import torch.nn as nn


class SoftmaxRegression(nn.Module):
    def __init__(self, n, m):  # n 개의 인자를 가진 다중 선형 회귀의 m중 분류를 구현
        super().__init__()
        self.linearRegressionModel = nn.Linear(n, m, bias=True)
        self.softmaxRegressionModel = nn.Softmax()

    def forward(self, x):  # forward propagation
        return self.softmaxRegressionModel(self.linearRegressionModel(x))
