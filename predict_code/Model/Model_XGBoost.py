import xgboost as xgb
import numpy as np
import torch
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV

class XGBoostWrapper:
    def __init__(self, num_classes, lr=0.00275, max_depth=6, subsample=0.8):
        self.lr = lr
        self.max_depth = max_depth
        self.subsample = subsample
        self.base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            num_class=num_classes,       # 添加类别数参数
            learning_rate=lr,
            max_depth=max_depth,
            subsample=subsample,
            n_estimators=1000,
            early_stopping_rounds=15  # 将早停参数移至构造函数
        )
        self.model = MultiOutputClassifier(self.base_model)
        
    def fit(self, X_train, y_train, X_val, y_val):
        # # 修正后的参数传递
        # self.base_model = xgb.XGBClassifier(
        #     objective='binary:logistic',
        #     learning_rate=self.lr,
        #     max_depth=self.max_depth,
        #     subsample=self.subsample,
        #     n_estimators=1000,
        # )
        # self.model = MultiOutputClassifier(self.base_model)
        # self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=15)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose = False)
        
        # 温度缩放校准（保持不变）
        self.scaled_model = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
        self.scaled_model.fit(X_val, y_val)
    
    def predict_proba(self, X):
        if self.scaled_model:
            return self.scaled_model.predict_proba(X)
        return self.model.predict_proba(X)
    
    def save(self, path):
        joblib.dump({'base': self.model, 'scaled': self.scaled_model}, path)
        
    @classmethod
    def load(cls, path):
        loaded = joblib.load(path)
        model = cls(num_classes=loaded['base'].n_features_out_)
        model.model = loaded['base']
        model.scaled_model = loaded['scaled']
        return model