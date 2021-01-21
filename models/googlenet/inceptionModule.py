import torch
import torch.nn as nn
import torch.nn.functional as F
class Inception(nn.Module):
    def __init__(self, infmap, fmap1, fmap13, fmap3, fmap15, fmap5, fpool, auxclassifier = False):
        super().__init__(self)  
        self.conv1 = nn.Conv2d(infmap, fmap1, 1, 1)
        self.conv13 = nn.Conv2d(infmap, fmap13, 1, 1)
        self.conv15 = nn.Conv2d(infmap, fmap15, 1, 1)
        self.pool1 = nn.MaxPool2d(3, 1, 1)
        self.conv3 = nn.Conv2d(fmap13, fmap3, 3, 1, 1)
        self.conv5 = nn.Conv2d(fmap15, fmap5, 5, 1, 2)
        self.convpool = nn.Conv2d(infmap, fpool, 1, 1)
        #intermediate softmax output
        self.auxclassifier = auxclassifier
        if auxclassifier == True:
            self.avgpool = nn.AvgPool2d(5, 3)
            self.auxconv = nn.Conv2d(infmap, 128, 1, 1)
            self.linear1 = nn.Linear(128, 1024)
            self.dropout = nn.Dropout(0.7)
            self.linear2 = nn.Linear(1024, 1000)
            self.softmax = nn.Softmax(1000)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = F.relu(y1)
        y3 = self.conv13(x)
        y3 = F.relu(y3)
        y3 = self.conv3(y3)
        y3 = F.relu(y3)
        y5 = self.conv15(x)
        y5 = F.relu(y5)
        y5 = self.conv5(y5)
        y5 = F.relu(y5)
        ypool = self.pool1(x)
        ypool = self.convpool(ypool)
        ypool = F.relu(ypool)
        y = torch.cat([y1, y3, y5, ypool], 1)
        if self.auxclassifier == True:
            ax = self.avgpool(x)
            ax = self.auxconv(ax)
            ax = F.relu(ax)
            ax = self.linear1(ax)
            ax = F.relu(ax)
            ax = self.dropout(ax)
            ax = self.linear2(ax)
            F.relu(ax)
            ax = self.softmax(ax)
            return y, ax
        else:
            return y