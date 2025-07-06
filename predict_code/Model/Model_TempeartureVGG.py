import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


class ECANet(nn.Module):
    def __init__(self, in_channels, b=1, gamma=2):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.b = b
        self.gamma = gamma

        # 根据公式计算自适应卷积核大小
        self.kernel_size = int(abs((math.log(self.in_channels, 2) + self.b) / self.gamma))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(self.in_channels, in_channels, kernel_size=self.kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 全局平均池化，将每个通道的特征图压缩为一个值，得到形状为(batch_size, channels)的张量
        y = self.avg_pool(x).view(batch_size, channels)

        # 调整形状为 (channels, 1)，以便进行1D卷积
        y = y.reshape(batch_size, channels, -1)

        # 进行1D卷积操作
        y = self.conv1d(y)
        # print(y.shape)

        # sigmoid激活，得到每个通道的权重系数
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # 将得到的权重与原始输入特征图进行对应通道的乘法操作
        return x * y.expand_as(x)

class VGG(nn.Module):
    def __init__(self, num_classes=1000, in_channels=64):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.eca = ECANet(32)
        self.Classifier_eca = ECANet(256)
        self.classifier = nn.Sequential(
            nn.Linear(256 , 125),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(125, 25),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        # x = self.eca(x)
        x = self.features(x)
        x = self.Classifier_eca(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
