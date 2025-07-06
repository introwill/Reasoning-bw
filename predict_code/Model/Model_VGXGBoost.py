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
    def __init__(self, num_classes=5, in_channels=207, epoch=350):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.epoch = epoch
        
        # 特征提取层
        self.features = nn.Sequential(
            # 将1D特征重塑为2D特征图
            nn.Unflatten(1, (1, in_channels)),  # 将[batch, features]变为[batch, 1, features]
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # 计算特征提取后的特征维度
        # 假设输入特征数为in_channels，经过5次池化层（每次缩小一半）
        feature_size = in_channels // (2**5)
        if feature_size < 1:
            feature_size = 1
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        """
        提取特征，用于XGBoost模型
        
        参数:
        x -- 输入数据张量 [batch_size, features]
        
        返回:
        特征向量
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        # 返回倒数第二层特征
        x = self.classifier[:-2](x)  # 使用除了最后一个线性层之外的所有层
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Trained_VGG11(nn.Module):
    def __init__(self, model, epoch, mean_val, std_val, device):
        super(Trained_VGG11, self).__init__()
        self.model = model
        self.epoch = epoch
        self.device = device
        self.mean = mean_val
        self.std = std_val

    def forward(self, input):
        # input = (input - self.mean) / (self.std + 1e-8)

        # if isinstance(input, np.ndarray):
        #     input = torch.tensor(input, dtype=torch.float32).to(self.device)

        # logits = self.model(input, self.epoch)
        # probs_sigmoid = torch.sigmoid(logits)
        # probs = probs_sigmoid.cpu().detach().numpy()
        # return probs
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32).to(self.device)

        # 确保均值和标准差在正确的设备上
        if isinstance(self.mean, np.ndarray):
            self.mean = torch.tensor(self.mean, dtype=torch.float32).to(self.device)
        if isinstance(self.std, np.ndarray):
            self.std = torch.tensor(self.std, dtype=torch.float32).to(self.device)
        
        if len(input.shape) == 2:  # 如果输入是二维的 [batch_size, features]
            # 标准化输入
            input = (input - self.mean) / self.std
        
        # 根据模型的forward方法参数需求进行调用
        try:
            # 尝试使用两个参数调用
            if hasattr(self, 'epoch') and self.epoch is not None:
                logits = self.model(input, self.epoch)
            else:
                # 如果没有epoch参数，只使用input调用
                logits = self.model(input)
        except TypeError:
            # 如果上面的调用失败，尝试只用一个参数调用
            logits = self.model(input)
            
        return logits
