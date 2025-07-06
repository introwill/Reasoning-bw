import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import math


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 0.25)
        # print(self.temperature)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        temperature = torch.abs(temperature)
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, probs_Switcher):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        # self.cpu()
        nll_criterion = nn.MultiLabelSoftMarginLoss()
        multi_label_hamming_loss_criterion = MultiLabelHammingLoss()  # 实例化多标签汉明损失计算类

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input, label = input.float().to('cuda'), label.long().to('cuda')
                # input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and Multi-label Hamming Loss before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_multi_label_hamming_loss = multi_label_hamming_loss_criterion(logits, labels, probs_Switcher).item()
        print('Before temperature - NLL: %.3f, Multi-label Hamming Loss: %.3f' % (
            before_temperature_nll, before_temperature_multi_label_hamming_loss))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.0125, max_iter=16)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and Multi-label Hamming Loss after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_multi_label_hamming_loss = multi_label_hamming_loss_criterion(
            self.temperature_scale(logits), labels, probs_Switcher).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, Multi-label Hamming Loss: %.3f' % (
            after_temperature_nll, after_temperature_multi_label_hamming_loss))

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
        # probs_Switcher = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        print(probs_Switcher)

        probs_Switcher = torch.from_numpy(probs_Switcher).float().cuda()
        probabilities = torch.sigmoid(logits).cuda()  # 将logits转换为概率值，多标签场景下直接使用sigmoid
        predictions = (probabilities >= probs_Switcher).float()  # 根据概率以0.5为阈值确定预测结果

        # 计算预测标签与真实标签不一致的元素个数
        diff = (predictions != labels).float().sum(dim=1)
        # print(f'Differ:\n{diff}')
        # print(f"predictions:\n{predictions}")
        # print(f'labels:\n{labels}')
        num_labels = labels.size(1)  # 获取总标签数量

        # 计算每个样本的汉明损失（不一致元素个数除以标签数量），再求平均得到总体多标签汉明损失
        multi_label_hamming_loss = diff / num_labels
        return multi_label_hamming_loss.mean()
