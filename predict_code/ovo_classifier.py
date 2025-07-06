import torch
import torch.nn as nn
import numpy as np
import copy
import itertools

class OvOClassifier:
    def __init__(self, base_model_class, num_classes, in_channels, device, covariates_length=0):
        """
        初始化一对一分类器
        
        Args:
            base_model_class: 基础模型类（不是实例）
            num_classes: 类别数量
            in_channels: 输入通道数
            device: 计算设备
            covariates_length: 协变量特征长度
        """
        self.num_classes = num_classes
        self.device = device
        self.binary_models = {}
        self.class_pairs = list(itertools.combinations(range(num_classes), 2))
        
        # 为每对类别创建一个二分类模型
        for i, j in self.class_pairs:
            model = base_model_class(num_classes=1, in_channels=in_channels, 
                                    Covariates_features_length=covariates_length)
            self.binary_models[(i, j)] = model.to(device)
    
    def train(self):
        """将所有模型设置为训练模式"""
        for model in self.binary_models.values():
            model.train()
    
    def eval(self):
        """将所有模型设置为评估模式"""
        for model in self.binary_models.values():
            model.eval()
    
    def parameters(self):
        """返回所有模型参数，用于优化器"""
        params = []
        for model in self.binary_models.values():
            params.extend(model.parameters())
        return params
    
    def forward(self, x, epoch=None):
        """前向传播，返回每个二分类器的输出和投票结果"""
        batch_size = x.size(0)
        votes = torch.zeros(batch_size, self.num_classes, device=self.device)
        outputs = {}
        
        for (i, j), model in self.binary_models.items():
            output = model(x, epoch)
            outputs[(i, j)] = output
            
            # 根据输出进行投票
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).float()
            
            # 更新投票
            for idx in range(batch_size):
                if pred[idx] > 0.5:  # 预测为类别i
                    votes[idx, i] += 1
                else:  # 预测为类别j
                    votes[idx, j] += 1
        
        return outputs, votes
    
    def predict(self, x, epoch=None):
        """预测样本类别"""
        _, votes = self.forward(x, epoch)
        # 选择得票最多的类别
        _, predicted_class = torch.max(votes, dim=1)
        return predicted_class
    
    def to(self, device):
        """将所有模型移动到指定设备"""
        self.device = device
        for key in self.binary_models:
            self.binary_models[key] = self.binary_models[key].to(device)
        return self