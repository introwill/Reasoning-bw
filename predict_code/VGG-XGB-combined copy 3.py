import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from Model import Model_VGG
import torch.optim as optim
from numpy import argmax
from sklearn.metrics import precision_recall_curve
from Model import mlsmote
import matplotlib
import matplotlib.pyplot as plt
from temperature_scaling import ModelWithTemperature
from Multilable_temerature_scaling import ModelWithTemperature
import shap
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
import statsmodels.formula.api as smf
import os
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import time
from datetime import datetime
import itertools

matplotlib.use('TKAgg')  # 用于解决绘图时的报错问题
seed_value = 3407
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

# 数据集处理
def encode_labels(File, columns_to_encode):
    one_hot_encoded_df = pd.get_dummies(File[columns_to_encode], columns=columns_to_encode, prefix_sep='_')
    selected_columns = [col for col in one_hot_encoded_df.columns if col.endswith('_1')]
    filtered_df = one_hot_encoded_df[selected_columns].to_numpy()
    return filtered_df


def open_excel(filename, columns_to_encode):
    readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
    nplist = readbook.T.to_numpy()
    data = nplist[1:-5].T
    data = np.float64(data)
    target = encode_labels(readbook, columns_to_encode=columns_to_encode)
    all_feature_names = readbook.columns[1:-5]
    Covariates_features = readbook.columns[1:4]
    print(all_feature_names)
    print(Covariates_features)
    return data, target, all_feature_names, Covariates_features

# 自定义数据集类
class NetDataset(Dataset):
    def __init__(self, features, labels):
        self.Data = features
        self.label = labels

    def __getitem__(self, index):
        return self.Data[index], self.label[index]

    def __len__(self):
        return len(self.Data)


# 数据标准化与交叉验证划分
def split_data_5fold(input_data):
    np.random.seed(3407)
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    fold_size = len(input_data) // 5
    folds_data_index = []
    for i in range(5):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        validation_indices = indices[(i + 1) * fold_size: (i + 2) * fold_size] if i < 4 else indices[4 * fold_size:]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 2) * fold_size:]]) if i < 4 else indices[:4 * fold_size]
        folds_data_index.append((train_indices, validation_indices, test_indices))
    return folds_data_index

# 添加绘制混淆矩阵的函数
def plot_confusion_matrix(cm, classes, fold, epoch, title='Confusion Matrix', cmap=plt.cm.Blues, is_sum=False):
    """
    绘制并显示混淆矩阵，使用归一化处理
    
    参数:
    cm -- 混淆矩阵，对于多标签问题，形状为(n_classes, 2, 2)
    classes -- 类别名称列表
    fold -- 当前fold编号
    epoch -- 当前epoch编号
    title -- 图表标题
    cmap -- 颜色映射
    is_sum -- 是否为累积混淆矩阵
    """
    # 确保figure目录存在
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # 处理多标签混淆矩阵
    if cm.ndim == 3 and cm.shape[1] == 2 and cm.shape[2] == 2:
        # 多标签情况，为每个类别绘制单独的混淆矩阵
        for i, class_name in enumerate(classes):
            plt.figure(figsize=(8, 6))
            
            # 提取当前类别的混淆矩阵
            cm_i = cm[i]
            
            # 创建原始混淆矩阵的副本
            cm_original = cm_i.copy()
            
            # 归一化混淆矩阵
            cm_row_sum = cm_i.sum(axis=1)
            cm_normalized = np.zeros_like(cm_i, dtype=float)
            for j in range(cm_i.shape[0]):
                if cm_row_sum[j] > 0:
                    cm_normalized[j] = cm_i[j] / cm_row_sum[j]
            
            # 绘制归一化后的混淆矩阵
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
            plt.title(f'{title} - {class_name} (Fold {fold+1}, Epoch {epoch})')
            plt.colorbar()
            
            # 设置标签
            labels = ['Negative', 'Positive']
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels)
            plt.yticks(tick_marks, labels)
            
            # 在每个单元格中添加原始数值和归一化后的百分比
            thresh = cm_normalized.max() / 2.
            for j, k in itertools.product(range(cm_i.shape[0]), range(cm_i.shape[1])):
                plt.text(k, j, f"{cm_original[j, k]}\n({cm_normalized[j, k]:.2f})",
                        horizontalalignment="center", 
                        color="white" if cm_normalized[j, k] > thresh else "black", 
                        fontsize=10)
            
            plt.tight_layout()
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # 保存图像
            if is_sum:
                plt.savefig(f'figure/confusion_matrix_{class_name}_fold{fold+1}_sum.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'figure/confusion_matrix_{class_name}_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
            
            plt.close()
        
        # 创建一个汇总的混淆矩阵可视化
        plt.figure(figsize=(15, 10))
        
        # 计算每个类别的性能指标
        metrics = []
        for i, class_name in enumerate(classes):
            tn, fp = cm[i][0, 0], cm[i][0, 1]
            fn, tp = cm[i][1, 0], cm[i][1, 1]
            
            # 计算指标
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append([class_name, accuracy, precision, recall, specificity, f1])
        
        # 创建表格
        columns = ['Class', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
        cell_text = [[f"{m[0]}", f"{m[1]:.2f}", f"{m[2]:.2f}", f"{m[3]:.2f}", f"{m[4]:.2f}", f"{m[5]:.2f}"] for m in metrics]
        
        # 绘制表格
        table = plt.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.axis('off')
        
        plt.title(f'Performance Metrics Summary (Fold {fold+1})', fontsize=14)
        
        # 保存汇总图像
        if is_sum:
            plt.savefig(f'figure/confusion_matrix_summary_fold{fold+1}_sum.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figure/confusion_matrix_summary_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    else:
        # 单标签情况，原始代码
        plt.figure(figsize=(12, 10))
        
        # 创建原始混淆矩阵的副本
        cm_original = cm.copy()
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # 处理可能的除零问题
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # 绘制归一化后的混淆矩阵
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.title(f'{title} (Fold {fold+1}, Epoch {epoch})')
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 在每个单元格中添加原始数值和归一化后的百分比
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if np.sum(cm_original[i]) > 0:  # 避免除以零
                plt.text(j, i, f"{cm_original[i, j]}\n({cm_normalized[i, j]:.2f})",
                        horizontalalignment="center", 
                        color="white" if cm_normalized[i, j] > 0.5 else "black", 
                        fontsize=8)
        
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 保存图像
        if is_sum:
            plt.savefig(f'figure/confusion_matrix_normalized_fold{fold+1}_sum.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figure/confusion_matrix_normalized_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
        
        plt.close()

def generate_multilabel_confusion_matrix(y_true, y_pred, fold, epoch):
    """
    生成多标签分类的32×32混淆矩阵
    
    参数:
    y_true -- 真实标签，形状为(n_samples, 5)
    y_pred -- 预测标签，形状为(n_samples, 5)
    fold -- 当前fold编号
    epoch -- 当前epoch编号
    """
    # 确保figure目录存在
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # 将多标签转换为单一类别标签（0-31）
    def multilabel_to_class(labels):
        return np.sum(labels * np.array([2**i for i in range(labels.shape[1])]), axis=1).astype(int)
    
    # 转换真实标签和预测标签
    y_true_class = multilabel_to_class(y_true)
    y_pred_class = multilabel_to_class(y_pred)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_class, y_pred_class, labels=range(32))
    
    # 创建类别标签
    class_names = []
    for i in range(32):
        # 将数字转换为二进制表示，然后填充到5位
        binary = format(i, '05b')
        # 创建标签，例如 "10110" 表示第1、3、4类疾病存在
        class_names.append(binary)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(20, 18))
    
    # 归一化混淆矩阵
    cm_row_sum = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if cm_row_sum[i] > 0:
            cm_normalized[i] = cm[i] / cm_row_sum[i]
    
    # 绘制归一化后的混淆矩阵
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'32×32 Confusion Matrix (Fold {fold+1}, Epoch {epoch})', fontsize=16)
    plt.colorbar()
    
    # 设置标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    # 在每个单元格中添加数值
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:  # 只显示非零值
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center", 
                    color="white" if cm_normalized[i, j] > thresh else "black", 
                    fontsize=6)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    # 保存图像
    plt.savefig(f'figure/confusion_matrix_32x32_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算每个类别的性能指标
    precision = np.zeros(32)
    recall = np.zeros(32)
    f1 = np.zeros(32)
    
    # 计算每个类别的精确率、召回率和F1分数
    for i in range(32):
        # 计算真阳性、假阳性和假阴性
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        # 计算精确率和召回率
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0
            
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0
            
        # 计算F1分数
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0
    
    return cm, precision, recall, f1


# 在test_xgb函数中添加对新函数的调用
def test_xgb(X_test, y_test, models, probs_Switcher):
    all_probs = np.zeros((X_test.shape[0], len(models)))
    
    # 对每个类别使用对应的模型进行预测
    for i, model in enumerate(models):
        dtest = xgb.DMatrix(X_test)
        all_probs[:, i] = model.predict(dtest)
    
    test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(all_probs, y_test, probs_Switcher)
    
    # 计算AUC-ROC
    auc_scores = []
    for i in range(y_test.shape[1]):
        try:
            auc_score = roc_auc_score(y_test[:, i], all_probs[:, i])
            auc_scores.append(auc_score)
        except ValueError:
            print(f"Class {i} has only one class present in test set.")
            auc_scores.append(float('nan'))
    
    # 计算宏平均和加权平均AUC
    macro_auc = np.nanmean(auc_scores)
    weighted_auc = roc_auc_score(y_test, all_probs, average='weighted', multi_class='ovr')
    
    return test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, y_test, preds

# 阈值计算函数
def Probs_Switcher(probs, labels):
    probs_Switcher = np.array([])
    
    for i in range(labels.shape[1]):
        split_labels_np = labels[:, i]
        split_probs_np = probs[:, i]
        precision, recall, thresholds = precision_recall_curve(split_labels_np, split_probs_np)
        precision = precision * 0.6  # 可以根据需要调整这个权重
        recall = recall
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        index = argmax(f1_scores)
        if len(thresholds) > index:
            probs_Switcher = np.append(probs_Switcher, thresholds[index])
        else:
            # 如果index超出thresholds范围，使用默认阈值0.5
            probs_Switcher = np.append(probs_Switcher, 0.5)

    return probs_Switcher

# 指标计算函数
def f1_score_func(probs, labels, probs_Switcher):
    preds = np.float64(probs >= probs_Switcher)

    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # 特异性计算
    cm = multilabel_confusion_matrix(labels, preds)
    specificities = []
    for i in range(cm.shape[0]):
        tn, fp = cm[i][0, 0], cm[i][0, 1]
        if (tn + fp) == 0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
        specificities.append(specificity)
    specificity = np.mean(specificities) * 100  # 使用宏平均

    return f1 * 100, accuracy * 100, precision * 100, recall * 100, preds, specificity

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

# 修改VGG特征提取器类
class DNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=92):
        super(DNNFeatureExtractor, self).__init__()
        # 创建一个新的特征提取网络，适合我们的输入维度
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),  # 输入维度为93（PCA特征数量）
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
        
    def forward(self, x):
        return self.feature_extractor(x)

# DNN训练函数
def train_dnn_extractor(model, train_loader, val_loader, device, num_epochs=100):
    model.train()
    criterion = nn.MSELoss()  # 使用MSE损失进行特征学习
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data, _ in train_loader:  # 我们不需要标签，进行无监督学习
            data = data.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)  # 自编码器式训练
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.float().to(device)
                output = model(data)
                val_loss += criterion(output, data).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # 早停机制
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_dnn_extractor.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_dnn_extractor.pth'))
    return model

# 修改特征提取函数
def extract_vgg_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.float().to(device)
            # 提取特征
            feature = model(data)
            
            features.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    features = np.vstack(features)
    labels = np.vstack(labels)
    
    return features, labels

# 测试函数
def test_xgb(X_test, y_test, models, probs_Switcher):
    all_probs = np.zeros((X_test.shape[0], len(models)))
    
    # 对每个类别使用对应的模型进行预测
    for i, model in enumerate(models):
        dtest = xgb.DMatrix(X_test)
        all_probs[:, i] = model.predict(dtest)
    
    test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(all_probs, y_test, probs_Switcher)
    
    # 计算AUC-ROC
    auc_scores = []
    for i in range(y_test.shape[1]):
        try:
            auc_score = roc_auc_score(y_test[:, i], all_probs[:, i])
            auc_scores.append(auc_score)
        except ValueError:
            print(f"Class {i} has only one class present in test set.")
            auc_scores.append(float('nan'))
    
    # 计算宏平均和加权平均AUC
    macro_auc = np.nanmean(auc_scores)
    weighted_auc = roc_auc_score(y_test, all_probs, average='weighted', multi_class='ovr')
    
    return test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, y_test, preds

# SHAP分析
def global_shap_analysis(models, background_data, test_data, feature_names, class_names, 
                        pca_model=None, original_feature_names=None, output_dir='figure/global_shap'):
    """
    增强版SHAP分析，支持特征逆向映射
    
    参数:
    models -- 训练好的XGBoost模型列表
    background_data -- 用于SHAP解释器的背景数据
    test_data -- 用于生成SHAP值的测试数据
    feature_names -- 特征名称列表
    class_names -- 类别名称列表
    pca_model -- 使用的PCA模型对象
    original_feature_names -- 原始特征名称列表
    output_dir -- 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查特征名称长度
    if len(feature_names) != test_data.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]
        print(f"警告：自动生成特征名称，长度为 {len(feature_names)}")
    
    # 存储所有类别的SHAP值
    all_shap_values = []
    
    # 为每个类别处理SHAP值
    for i, (class_name, model) in enumerate(zip(class_names, models)):
        print(f"计算类别 {class_name} 的全局SHAP值...")
        
        # 创建解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(test_data)
        all_shap_values.append(shap_values)
        
        # 绘制条形图
        plt.figure(figsize=(10, 8))
        try:
            # 计算特征重要性（平均绝对SHAP值）
            feature_importance = np.abs(shap_values).mean(0)
            # 获取排序索引
            sorted_idx = np.argsort(feature_importance)
            # 选择最重要的特征
            top_features = min(20, len(feature_names))
            plt.barh(range(top_features), feature_importance[sorted_idx[-top_features:]])
            plt.yticks(range(top_features), [feature_names[i] for i in sorted_idx[-top_features:]])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Feature Importance for Category {class_name}')
        except Exception as e:
            print(f"绘制条形图时出错: {e}")
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/global_bar_plot_category_{class_name}.png')
        plt.close()
        
        # 绘制摘要图
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values, test_data, feature_names=feature_names, show=False, max_display=20)
            plt.title(f'SHAP Summary Plot for Category {class_name}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/global_summary_plot_category_{class_name}.png')
        except Exception as e:
            print(f"绘制摘要图时出错: {e}")
        plt.close()
    
    # 绘制多项式图
    mean_abs_shap = np.zeros((len(feature_names), len(class_names)))
    for i, shap_values in enumerate(all_shap_values):
        mean_abs_shap[:, i] = np.abs(shap_values).mean(axis=0)
    
    agg_shap_df = pd.DataFrame(mean_abs_shap, columns=class_names, index=feature_names)
    
    # 按特征重要性总和排序
    feature_order = agg_shap_df.sum(axis=1).sort_values(ascending=False).index
    agg_shap_df = agg_shap_df.loc[feature_order]
    
    plt.figure(figsize=(18, 8))
    bottom = np.zeros(len(agg_shap_df))
    colors = sns.color_palette("tab10", len(class_names))
    
    for i, disease in enumerate(class_names):
        plt.bar(
            agg_shap_df.index,
            agg_shap_df[disease],
            bottom=bottom,
            label=disease,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5
        )
        bottom += agg_shap_df[disease]
    
    plt.xlabel("Top Most Important Features in Predicting Disease", fontsize=12)
    plt.ylabel("mean(|SHAP value|) / average impact on model output magnitude", fontsize=12)
    plt.title("Polynomial-SHAP plot of the data", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.legend(
        title="",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/polynomial_shap_plot.png")
    plt.close()

    # 新增特征逆向映射分析
    if pca_model is not None and original_feature_names is not None and len(original_feature_names) > 0:
        print("\n开始特征重要性逆向映射分析...")
        
        # 计算PCA组件的SHAP重要性
        pca_shap_importance = np.abs(np.array(all_shap_values)).mean(axis=(0,1))
        
        # 确保维度匹配
        if len(pca_shap_importance) != pca_model.components_.shape[0]:
            print(f"警告：SHAP重要性维度({len(pca_shap_importance)})与PCA组件数({pca_model.components_.shape[0]})不匹配，将截断或填充")
            min_dim = min(len(pca_shap_importance), pca_model.components_.shape[0])
            pca_shap_importance = pca_shap_importance[:min_dim]
            pca_components = pca_model.components_[:min_dim, :]
        else:
            pca_components = pca_model.components_
        
        # 计算原始特征重要性 = PCA组件重要性 × PCA载荷矩阵
        original_feature_importance = np.dot(pca_shap_importance, np.abs(pca_components))  # 使用截断后的pca_components
        
        # 创建并保存原始特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': original_feature_names.values if hasattr(original_feature_names, 'values') else original_feature_names,
            'importance': original_feature_importance
        }).sort_values('importance', ascending=False)
        
        # 保存到CSV
        importance_df.to_csv(f'{output_dir}/original_feature_importance.csv', index=False)
        
        # 绘制原始特征重要性图
        plt.figure(figsize=(12, 8))
        top_n = min(30, len(importance_df))
        sns.barplot(
            x='importance', 
            y='feature', 
            data=importance_df.head(top_n),
            palette='viridis'
        )
        plt.title('Top Original Features by SHAP Importance')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Original Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/original_feature_importance.png', dpi=300)
        plt.close()
    
    return all_shap_values

def plot_roc_curves(all_labels, all_probs, class_names):
    plt.figure(figsize=(10, 8))

    # 为每个类别绘制ROC曲线
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'figure/vgg_xgb_roc_curves.png')
    plt.close()

def main():
    """
    主函数，包含程序的主要执行逻辑
    """
    # 设置类别
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    
    # 加载数据
    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)
    
    # 保留原始数据用于测试
    features_val = features.copy()
    labels_val = labels.copy()
    
    # 数据增强
    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)
    
    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)
    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)
    
    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)
    
    # 标准化
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8
    
    features = (features - mean_f) / std_f
    features_val = (features_val - mean_f) / (std_f + 1e-8)
    
    # 方差过滤
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    features_val = selector.transform(features_val)
    
    # 更新特征名称
    mask = selector.get_support()
    filtered_feature_names = all_feature_names[mask]
    
    print("方差过滤后，训练集形状：", features.shape)
    print("方差过滤后，验证集形状：", features_val.shape)
    
    # PCA降维
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features)
    features_val_pca = pca.transform(features_val)
    
    # PCA后再次过滤零方差
    pca_selector = VarianceThreshold(threshold=0.01)  
    features_pca = pca_selector.fit_transform(features_pca)
    features_val_pca = pca_selector.transform(features_val_pca)
    
    # 更新PCA特征名称
    pca_feature_names = [f'PC{i}' for i in range(1, features_pca.shape[1] + 1)]
    print("PCA降维后，训练集形状：", features_pca.shape)
    print("PCA降维后，验证集形状：", features_val_pca.shape)
    
    # 交叉验证划分
    folds_data_index = split_data_5fold(features_val_pca)
    
    # 设置模型参数
    num_classes = len(columns_to_encode)
    num_epochs = 100  # VGG训练轮数
    batch_size = 256
    
    # 让用户输入想要运行的fold编号
    # selected_folds_input = input("请输入想要运行的fold编号（用逗号分隔，例如：1,3,5）：")
    selected_folds_input = '1,2,3,4,5'  # 默认运行所有fold
    # if not selected_folds_input.strip():
    #     selected_folds_input = '1,2,3,4,5'  # 默认运行所有fold
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]
    
    # 存储所有fold的结果
    all_fold_results = []
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue
        
        print(f"\n========== 开始处理 Fold {fold + 1} ==========")
        
        # 准备数据
        trainX = features_pca  # 使用全部增强数据训练
        trainY = labels
        valX = features_val_pca[validation_index]
        valY = labels_val[validation_index]
        testX = features_val_pca[test_indices]
        testY = labels_val[test_indices]
        
        # 创建数据加载器
        Train_data = NetDataset(trainX, trainY)
        Validation_data = NetDataset(valX, valY)
        Test_data = NetDataset(testX, testY)
        
        Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
        Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        
        # 第一阶段：训练DNN模型作为特征提取器
        print("第一阶段：训练DNN模型作为特征提取器...")
        

        
        # # 打印模型参数数量
        # total_num, trainable_num = get_parameter_number(model)
        # print(f'DNN模型总参数: {total_num}, 可训练参数: {trainable_num}')
        
        # # 定义损失函数和优化器
        # criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        
        # # 初始化早停变量
        # best_valid_f1 = 0
        # patience = 10
        # counter = 0
        # best_epoch = 0

        # 初始化DNN特征提取器
        dnn_extractor = DNNFeatureExtractor(input_dim=features_pca.shape[1]).to(device)

        # 打印模型参数数量
        total_num, trainable_num = get_parameter_number(dnn_extractor)
        print(f'DNN模型总参数: {total_num}, 可训练参数: {trainable_num}')

        # 训练DNN特征提取器
        print("训练DNN特征提取器...")
        dnn_extractor = train_dnn_extractor(
            dnn_extractor,
            Train_data_loader,
            Validation_data_loader,
            device,
            num_epochs=50  # 可以调整训练轮数
        )
        
        # 创建特征提取器
        # feature_extractor = VGGFeatureExtractor(model, input_dim=features_pca.shape[1])
        # feature_extractor.to(device)
        
        print("第二阶段：使用DNN提取特征，XGBoost进行预测...")
            
        train_features, _ = extract_vgg_features(dnn_extractor, Train_data_loader, device)
        val_features, _ = extract_vgg_features(dnn_extractor, Validation_data_loader, device)
        test_features, _ = extract_vgg_features(dnn_extractor, Test_data_loader, device)
        
        # 为每个类别训练一个XGBoost模型
        xgb_models = []
        xgb_feature_names = [f'DNN_Feature_{i}' for i in range(train_features.shape[1])]
        
        # 计算阈值
        probs_Switcher = np.zeros(num_classes)
        
        for i in range(num_classes):
            print(f"训练类别 {columns_to_encode[i]} 的XGBoost模型...")
            
            # 准备数据
            dtrain = xgb.DMatrix(train_features, label=train_labels[:, i])
            dval = xgb.DMatrix(val_features, label=val_labels[:, i])
            
            # 设置XGBoost参数
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'alpha': 0.1,
                'lambda': 1,
                'seed': 3407
            }
            
            # 训练模型
            evallist = [(dtrain, 'train'), (dval, 'eval')]
            num_round = 100
            bst = xgb.train(params, dtrain, num_round, evallist, 
                            early_stopping_rounds=10, verbose_eval=10)
            
            # 保存模型
            bst.save_model(f'xgb_model_fold{fold+1}_class{i}.json')
            xgb_models.append(bst)
            
            # 计算阈值
            dval_pred = bst.predict(dval)
            precision, recall, thresholds = precision_recall_curve(val_labels[:, i], dval_pred)
            precision = precision * 0.6
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
            index = argmax(f1_scores)
            if len(thresholds) > index:
                probs_Switcher[i] = thresholds[index]
            else:
                probs_Switcher[i] = 0.5
            
        # 保存阈值
        np.save(f'probs_switcher_fold{fold+1}.npy', probs_Switcher)
        
        # 测试XGBoost模型
        test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels, all_preds = test_xgb(
            test_features, test_labels, xgb_models, probs_Switcher)
        
        # 绘制ROC曲线
        plot_roc_curves(all_labels, all_probs, columns_to_encode)
        
        # 绘制混淆矩阵
        cm = multilabel_confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, columns_to_encode, fold, num_epochs, is_sum=True)

        # 生成32×32的混淆矩阵
        cm_32x32, precision_32, recall_32, f1_32 = generate_multilabel_confusion_matrix(all_labels, all_preds, fold, num_epochs)
        
        # 输出测试结果
        print(
            f'Fold [{fold + 1}/{len(selected_folds)}]: '
            f'VGG+XGBoost Test F1: {test_f1:.2f}% | '
            f'Accuracy: {test_accuracy:.2f}% | '
            f'Precision: {test_precision:.2f}% | '
            f'Recall: {test_recall:.2f}% | '
            f'Specificity: {test_specificity:.2f}% | '
            f'Macro AUC: {macro_auc:.3f} | '
            f'Weighted AUC: {weighted_auc:.3f}'
        )
        
        # 将结果保存到CSV文件
        with open('vgg_xgb_results.csv', 'a') as f:
            f.write(
                f'fold [{fold + 1}/{len(selected_folds)}], '
                f'VGG+XGBoost Test F1: {test_f1:.2f}%, '
                f'Accuracy: {test_accuracy:.2f}%, '
                f'Precision: {test_precision:.2f}%, '
                f'Recall: {test_recall:.2f}%, '
                f'Specificity: {test_specificity:.2f}%, '
                f'Macro AUC: {macro_auc:.3f}, '
                f'Weighted AUC: {weighted_auc:.3f}\n'
            )
        
        # 执行SHAP分析
        print("执行SHAP分析...")
        
        # 选择一部分测试数据用于SHAP分析
        shap_sample_size = min(100, test_features.shape[0])
        shap_indices = np.random.choice(test_features.shape[0], shap_sample_size, replace=False)
        shap_data = test_features[shap_indices]
        
        # 执行全局SHAP分析
        global_shap_analysis(
            models=xgb_models,
            background_data=train_features,
            test_data=test_features,
            feature_names=pca_feature_names,  # PCA特征名称
            class_names=columns_to_encode,    # 类别名称
            pca_model=pca,                    # PCA模型对象
            original_feature_names=filtered_feature_names,  # 原始特征名称
            output_dir='figure/global_shap'
        )
        
        # 清理内存
        del train_features, val_features, test_features
        del train_labels, val_labels, test_labels
        torch.cuda.empty_cache()
    
    print("所有fold训练和测试完成！")
    
    # 汇总所有fold的结果
    print("\n===== 汇总结果 =====")
    
    # 检查结果文件是否存在
    if os.path.exists('vgg_xgb_results.csv') and os.path.getsize('vgg_xgb_results.csv') > 0:
        try:
            # 直接读取文件内容
            with open('vgg_xgb_results.csv', 'r') as f:
                lines = f.readlines()
            
            # 提取指标
            f1_scores = []
            accuracies = []
            precisions = []
            recalls = []
            specificities = []
            macro_aucs = []
            weighted_aucs = []
            
            for i, line in enumerate(lines):
                try:
                    # 使用正则表达式提取指标值
                    import re
                    
                    # 提取F1分数
                    f1_match = re.search(r'Test F1: (\d+\.\d+)%', line)
                    if f1_match:
                        f1_scores.append(float(f1_match.group(1)))
                    
                    # 提取准确率
                    acc_match = re.search(r'Accuracy: (\d+\.\d+)%', line)
                    if acc_match:
                        accuracies.append(float(acc_match.group(1)))
                    
                    # 提取精确率
                    prec_match = re.search(r'Precision: (\d+\.\d+)%', line)
                    if prec_match:
                        precisions.append(float(prec_match.group(1)))
                    
                    # 提取召回率
                    recall_match = re.search(r'Recall: (\d+\.\d+)%', line)
                    if recall_match:
                        recalls.append(float(recall_match.group(1)))
                    
                    # 提取特异性
                    spec_match = re.search(r'Specificity: (\d+\.\d+)%', line)
                    if spec_match:
                        specificities.append(float(spec_match.group(1)))
                    
                    # 提取宏观AUC
                    macro_match = re.search(r'Macro AUC: (\d+\.\d+)', line)
                    if macro_match:
                        macro_aucs.append(float(macro_match.group(1)))
                    
                    # 提取加权AUC
                    weighted_match = re.search(r'Weighted AUC: (\d+\.\d+)', line)
                    if weighted_match:
                        weighted_aucs.append(float(weighted_match.group(1)))
                    
                except Exception as e:
                    print(f"处理第{i+1}行时出错：{e}，跳过该行")
            
            if len(f1_scores) > 0:
                # 计算平均值和标准差
                print(f"平均 F1 分数: {np.mean(f1_scores):.2f}% ± {np.std(f1_scores):.2f}%")
                print(f"平均准确率: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
                print(f"平均精确率: {np.mean(precisions):.2f}% ± {np.std(precisions):.2f}%")
                print(f"平均召回率: {np.mean(recalls):.2f}% ± {np.std(recalls):.2f}%")
                print(f"平均特异性: {np.mean(specificities):.2f}% ± {np.std(specificities):.2f}%")
                print(f"平均宏观AUC: {np.mean(macro_aucs):.3f} ± {np.std(macro_aucs):.3f}")
                print(f"平均加权AUC: {np.mean(weighted_aucs):.3f} ± {np.std(weighted_aucs):.3f}")
                
                # 将汇总结果保存到CSV文件
                with open('vgg_xgb_summary.csv', 'w') as f:
                    f.write("指标,平均值,标准差\n")
                    f.write(f"F1 分数,{np.mean(f1_scores):.2f}%,{np.std(f1_scores):.2f}%\n")
                    f.write(f"准确率,{np.mean(accuracies):.2f}%,{np.std(accuracies):.2f}%\n")
                    f.write(f"精确率,{np.mean(precisions):.2f}%,{np.std(precisions):.2f}%\n")
                    f.write(f"召回率,{np.mean(recalls):.2f}%,{np.std(recalls):.2f}%\n")
                    f.write(f"特异性,{np.mean(specificities):.2f}%,{np.std(specificities):.2f}%\n")
                    f.write(f"宏观AUC,{np.mean(macro_aucs):.3f},{np.std(macro_aucs):.3f}\n")
                    f.write(f"加权AUC,{np.mean(weighted_aucs):.3f},{np.std(weighted_aucs):.3f}\n")
            else:
                print("没有有效的结果数据可以汇总")
        except Exception as e:
            print(f"汇总结果时出错：{e}")
            print("请检查vgg_xgb_results.csv文件格式是否正确")
    else:
        print("结果文件不存在或为空，无法汇总结果")

if __name__ == '__main__':  
    main()