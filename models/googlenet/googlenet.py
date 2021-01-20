import torch
import torch.nn as nn
from inceptionModule import inception

class googlenet(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.mxpool1 = nn.MaxPool2d(3, 2)
        self.lrn1 = nn.LocalResponseNorm(3, 1.0, 0.75, 1.0)
        self.conv21 = nn.Conv2d(64, 64, 1, 1)
        self.conv2 = nn.Conv2d(64, 192, 3, 1)
        self.lrn2 = nn.LocalResponseNorm(3, 1.0, 0.75, 1.0)
        self.mxpool2 = nn.Maxpool2(3, 2)
        self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)
        self.mxpool3 = nn.Maxpool2(3, 2)
        self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception(512, 112, 224, 24, 64, 64)
        self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)
        self.mxpool4 = nn.MaxPool2d(3, 2)
        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(7, 1)
        self.drpout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(1024, 1000)
        self.softmax = nn.Softmax(1000)