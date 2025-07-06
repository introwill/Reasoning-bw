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

class VGG11(nn.Module):
    def __init__(self, num_classes, in_channels, epoch=305):
        super(VGG11, self).__init__()
        self.epoch = epoch
        self.conv_layer1 = self._make_conv_1(in_channels, 32)
        self.conv_layer2 = self._make_conv_1(32,64)
        self.conv_layer3 = self._make_conv_2(64,128)
        self.conv_layer4 = self._make_conv_2(128,256)
        self.conv_layer5 = self._make_conv_2(256,512)
        self.Classifier_eca = ECANet(512)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 125),    # 这里修改一下输入输出维度
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(125, 25),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(25, num_classes)
            # 使用交叉熵损失函数，pytorch的nn.CrossEntropyLoss()中已经有过一次softmax处理，这里不用再写softmax
        )

    def _make_conv_1(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=2),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        return layer
    def _make_conv_2(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=2),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels, kernel_size=2, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
              )
        return layer

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        # 32*32 channel == 3
        x = self.conv_layer1(x)
        # 16*16 channel == 64
        x = self.conv_layer2(x)
        # 8*8 channel == 128
        x = self.conv_layer3(x)
        # 4*4 channel == 256
        x = self.conv_layer4(x)
        # 2*2 channel == 512
        x = self.conv_layer5(x)
        # 1*1 channel == 512
        x = self.Classifier_eca(x)
        x = x.view(x.size(0), -1)
        # 512
        x = self.classifier(x)
        # 10
        x = x * (1 + math.log(self.epoch + 1))
        return x

class Trained_VGG11(nn.Module):
    def __init__(self, model, epoch, mean_val, std_val, device):
        super(Trained_VGG11, self).__init__()
        self.model = model
        self.epoch = epoch
        self.device = device
        self.mean = mean_val
        self.std = std_val

    def forward(self, input):
        input = (input - self.mean) / (self.std + 1e-8)

        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32).to(self.device)

        logits = self.model(input, self.epoch)
        probs_sigmoid = torch.sigmoid(logits)
        probs = probs_sigmoid.cpu().detach().numpy()
        return probs
