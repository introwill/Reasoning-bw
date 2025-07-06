import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        """
        初始化DNN模型
        
        参数:
        input_dim -- 输入特征的维度
        hidden_dims -- 隐藏层维度的列表，例如[512, 256, 128]
        output_dim -- 输出维度（类别数量）
        dropout_rate -- Dropout比率，用于防止过拟合
        """
        super(DNN, self).__init__()
        
        # 构建层列表
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层之间的连接
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 将所有层打包为一个ModuleList
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        x -- 输入数据，形状为 [batch_size, input_dim]
        
        返回:
        输出预测，形状为 [batch_size, output_dim]
        """
        # 确保输入是2D的
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # 通过所有层
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def get_features(self, x):
        """
        获取特征表示（最后一个隐藏层的输出）
        
        参数:
        x -- 输入数据，形状为 [batch_size, input_dim]
        
        返回:
        特征表示，形状为 [batch_size, hidden_dims[-1]]
        """
        # 确保输入是2D的
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # 通过除了最后一层以外的所有层
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            
        return x