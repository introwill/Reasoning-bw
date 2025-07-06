import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(203, 64, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print("After conv1 and pool:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print("After conv2 and pool:", x.shape)
        x = x.view(64, -1)
        # print("After view:", x.shape)
        x = F.relu(self.fc1(x))
        # print("After fc1:", x.shape)
        x = self.fc2(x)
        # print("Output shape:", x.shape)
        return x