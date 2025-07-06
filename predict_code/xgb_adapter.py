import torch
import numpy as np
import xgboost as xgb

class XGBoostAdapter:
    """
    适配器类，将XGBoost模型包装成可与温度缩放兼容的形式
    """
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def forward(self, x):
        """
        接收numpy数组或torch张量，返回预测的logits
        """
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # 创建DMatrix
        dmatrix = xgb.DMatrix(x_np)
        
        # 获取预测值（概率）
        probs = self.xgb_model.predict(dmatrix)
        
        # 将概率转换为logits: logit = log(p/(1-p))
        # 添加一个小常数避免除零或log(0)
        epsilon = 1e-7
        probs = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs / (1 - probs))
        
        # 处理批量数据
        if len(logits.shape) == 1:
            # 将一维数组reshape为[batch_size, 1]
            logits = logits.reshape(-1, 1)
        
        # 转换为torch张量
        logits_tensor = torch.tensor(logits).float().to(self.device)
        
        return logits_tensor
    
    def __call__(self, x):
        return self.forward(x)
