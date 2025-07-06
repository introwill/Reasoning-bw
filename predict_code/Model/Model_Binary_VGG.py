import torch
import torch.nn as nn

class BinaryVGG11(nn.Module):
    def __init__(self, in_channels, epoch=350):
        super(BinaryVGG11, self).__init__()
        self.epoch = epoch
        self.in_channels = in_channels
        
        # 特征提取部分保持不变
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv1d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv1d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # 计算卷积后的特征维度
        # 经过5次下采样，特征维度变为原来的1/32
        feature_size = in_channels // 32
        if feature_size == 0:
            feature_size = 1
        
        # 分类器部分修改为单类别输出
        self.classifier = nn.Sequential(
            nn.Linear(1024 * feature_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),  # 输出单个类别
        )
        
    def forward(self, x):
        # 调整输入形状以适应一维卷积
        x = x.unsqueeze(1)  # [batch_size, 1, in_channels]
        x = self.features(x)
        
        # 打印特征形状以便调试
        # print(f"Feature shape: {x.shape}")
        
        x = x.view(x.size(0), -1)
        
        # 打印展平后的形状以便调试
        # print(f"Flattened shape: {x.shape}")
        
        x = self.classifier(x)
        return x

class Trained_BinaryVGG11(nn.Module):
    def __init__(self, model, epoch, mean, std, device):
        super(Trained_BinaryVGG11, self).__init__()
        self.model = model
        self.epoch = epoch
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)
        self.device = device

    def forward(self, x):
        # 标准化输入
        x = (x - self.mean) / self.std
        x = x.to(self.device)
        return self.model(x)