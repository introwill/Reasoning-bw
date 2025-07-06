import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import math

class ModelWithTemperature(nn.Module):
    """
    为32类分类模型添加温度缩放
    """
    def __init__(self, model, num_classes):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.num_classes = num_classes
        
    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)
        
    def temperature_scale(self, logits):
        """
        对logits应用温度缩放
        """
        # 对于32类分类，我们直接缩放logits
        return logits / self.temperature
        
    def set_temperature(self, valid_loader, probs_Switcher):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.model.eval()
        nll_criterion = nn.NLLLoss()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                # 确保输入转换为 float32
                input = input.to(device).float()
                label = label.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list).cuda()
            
        # 计算未缩放的NLL损失
        before_temperature_nll = nll_criterion(logits, torch.argmax(labels, dim=1)).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))
        
        # 优化温度值
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), torch.argmax(labels, dim=1))
            loss.backward()
            return loss
            
        optimizer.step(eval)
        
        # 计算缩放后的NLL损失
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), torch.argmax(labels, dim=1)).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f' % (after_temperature_nll))
        
        self.cpu()
        return self

class MultiLabelHammingLoss(nn.Module):
    """
    Calculates the Multi-label Hamming Loss.
    It measures the fraction of incorrect label predictions across all labels for multi-label classification tasks.
    """

    def __init__(self):
        super(MultiLabelHammingLoss, self).__init__()

    def forward(self, logits, labels, probs_Switcher):
        """
        Calculate the Multi-label Hamming Loss based on given logits and labels.

        Args:
            logits (torch.Tensor): Model output logits of shape (batch_size, num_labels) for multi-label classification.
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_labels) where values are 0 or 1.

        Returns:
            torch.Tensor: The calculated Multi-label Hamming Loss.
        """
        probs_Switcher = torch.from_numpy(probs_Switcher).float().cuda()
        probabilities = torch.sigmoid(logits).cuda()  # 将logits转换为概率值，多标签场景下直接使用sigmoid
        predictions = (probabilities >= probs_Switcher).float()  # 根据概率以0.5为阈值确定预测结果

        # 计算预测标签与真实标签不一致的元素个数
        diff = (predictions != labels).float().sum(dim=1)
        num_labels = labels.size(1)  # 获取总标签数量

        # 计算每个样本的汉明损失（不一致元素个数除以标签数量），再求平均得到总体多标签汉明损失
        multi_label_hamming_loss = diff / num_labels
        return multi_label_hamming_loss.mean()