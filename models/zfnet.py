import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class contrastNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        eps = 1e-4
        mu = torch.mean(x, (-1, -2), True)
        sig = torch.var(x, (-1, -2), False, True)
        x = (x - mu).div(torch.sqrt(sig + eps))
        return x

class zfNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, 2)
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.contrastnorm1 = contrastNorm()
        self.conv2 = nn.Conv2d(96, 256, 5, 2)
        self.maxpool2 = nn.MaxPool2d(3, 2)
        self.contrastnorm2 = contrastNorm()
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(3, 2)
        self.linear1 = nn.Linear(6*6*256, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.contrastnorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.contrastnorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = torch.flatten(x ,1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.softmax(x)
        




