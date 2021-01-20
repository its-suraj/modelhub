#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#my logic for LRNorm
# def LRNorm(A, k, alpha, beta, n):
#     N = A.shape[1]
#     B = torch.zeros(A.shape)
#     for i in range(N):
#         kindexStart = max(0, math.floor(i - n/2))
#         kindexStop = min(N - 1, math.ceil(i + n/2))
#         B[:, i, :, :] = A[:, i, :, :]/(k + alpha*A[:, kindexStart:kindexStop, :, :].sum())**beta
#     return B

#LRN module from : https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
class LRN(nn.Module):
    def __init__(self, local_size=1, k=1.0, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class alexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.lrn1 = nn.LRN(5, 2, 1e-4)
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.lrn2 = nn.LRN(5, 2, 1e-4)
        self.maxpool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(3, 2)
        self.linear1 = nn.Linear(9216, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.lrn1(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.softmax(x)
        return x



