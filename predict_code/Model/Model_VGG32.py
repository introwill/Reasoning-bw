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

# class VGG11(nn.Module):
#     def __init__(self, num_classes, in_channels, epoch):
#         super(VGG11, self).__init__()
#         self.epoch = epoch
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=1, padding=1),
#             # nn.ReLU(inplace=False),
#             # 改成tanh激活
#             nn.Tanh(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(64, 128, kernel_size=1, padding=1),
#             # nn.ReLU(inplace=False),
#             nn.Tanh(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(128, 256, kernel_size=1, padding=1),
#             # nn.ReLU(inplace=False),
#             nn.Tanh(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(256, 512, kernel_size=1, padding=1),
#             # nn.ReLU(inplace=False),
#             nn.Tanh(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # nn.Conv2d(512, 1024, kernel_size=1, padding=1),
#             # nn.ReLU(inplace=False),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.Classifier_eca = ECANet(512)

#         self.classifier = nn.Sequential(
#             # nn.Linear(1024, 512),
#             # nn.ReLU(inplace=False),
#             # nn.Dropout(),

#             nn.Linear(512, 256),
#             # nn.ReLU(inplace=False),
#             nn.Tanh(),
#             nn.Dropout(),

#             nn.Linear(256, 125),
#             # nn.ReLU(inplace=False),
#             nn.Tanh(),
#             nn.Dropout(),

#             nn.Linear(125, num_classes),
#         )

#     def forward(self, x):
#         x = x.reshape(x.shape[0], x.shape[1], 1, 1)
#         x = self.features(x)
#         x = self.Classifier_eca(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         x = x * (1 + math.log(self.epoch+1))
#         return x

class VGG11(nn.Module):
    def __init__(self, num_classes=32, in_channels=209, epoch=350):
        super(VGG11, self).__init__()
        self.epoch = epoch
        self.features = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),  # 输出32个类别
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # 不使用sigmoid激活，因为使用CrossEntropyLoss

class Trained_VGG11(nn.Module):
    def __init__(self, model, epoch, mean=None, std=None, device=None):
        super(Trained_VGG11, self).__init__()
        self.model = model
        self.epoch = epoch
        self.mean = mean
        self.std = std
        self.device = device
        
    def forward(self, x):
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        x = self.model(x)
        return x

# class Trained_VGG11(nn.Module):
#     def __init__(self, model, epoch, mean_val, std_val, device):
#         super(Trained_VGG11, self).__init__()
#         self.model = model
#         self.epoch = epoch
#         self.device = device
#         self.mean = mean_val
#         self.std = std_val
        

#     def forward(self, x):
#         if self.mean is not None and self.std is not None:
#             x = (x - self.mean) / self.std
#         x = self.model(x)
#         return x
