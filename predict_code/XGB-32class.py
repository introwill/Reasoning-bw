import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import argmax
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
import statsmodels.formula.api as smf
import os
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import shap
import time
from datetime import datetime

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

# 将多标签转换为单一类别标签（32类）
def multilabel_to_multiclass(labels):
    """
    将多标签数据转换为多类别数据
    例如：[0,1,0,0,1] -> 9 (二进制01001)
    """
    n_samples = labels.shape[0]
    multiclass_labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # 将二进制标签转换为整数类别
        binary_str = ''.join(map(str, labels[i].astype(int)))
        multiclass_labels[i] = int(binary_str, 2)
    
    return multiclass_labels

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

# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_names, fold, epoch):
    """
    绘制混淆矩阵图片
    
    参数:
    cm -- 混淆矩阵
    class_names -- 类别名称列表
    fold -- 当前fold编号
    epoch -- 当前epoch编号
    """
    # 获取混淆矩阵的实际大小
    n_classes_actual = cm.shape[0]
    
    # 如果实际类别数小于预期类别数，创建一个更大的混淆矩阵并填充0
    if n_classes_actual < len(class_names):
        new_cm = np.zeros((len(class_names), len(class_names)))
        new_cm[:n_classes_actual, :n_classes_actual] = cm
        cm = new_cm
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)  # 添加小值避免除零
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零问题
    
    # 确保figure目录存在
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12))
    
    # 使用seaborn绘制热图
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                xticklabels=range(len(class_names)),
                yticklabels=range(len(class_names)))
    
    plt.title(f'Normalized Confusion Matrix (Fold {fold+1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'figure/xgb_32class_confusion_matrix_fold{fold+1}.png')
    plt.close()
    
    # 如果类别数量较多，绘制一个简化版本，只显示对角线和主要错误
    if len(class_names) > 10:
        # 找出每行中最大的非对角线元素
        top_errors = []
        for i in range(min(cm_normalized.shape[0], len(class_names))):
            if i < cm_normalized.shape[0]:  # 确保索引在范围内
                row = cm_normalized[i].copy()
                if i < row.size:  # 确保对角线元素索引在范围内
                    row[i] = 0  # 忽略对角线元素
                if np.max(row) > 0.1:  # 只关注错误率超过10%的
                    top_errors.append((i, np.argmax(row), np.max(row)))
        
        # 绘制主要错误的热图
        if top_errors:
            plt.figure(figsize=(10, 8))
            error_df = pd.DataFrame(top_errors, columns=['Actual', 'Predicted', 'Error Rate'])
            error_df = error_df.sort_values('Error Rate', ascending=False)
            
            plt.bar(range(len(error_df)), error_df['Error Rate'])
            plt.xticks(range(len(error_df)), [f"{a}->{p}" for a, p in zip(error_df['Actual'], error_df['Predicted'])], rotation=45)
            plt.title(f'Top Misclassifications (Fold {fold+1})')
            plt.xlabel('Actual -> Predicted')
            plt.ylabel('Error Rate')
            plt.tight_layout()
            plt.savefig(f'figure/xgb_32class_top_errors_fold{fold+1}.png')
            plt.close()

# 测试函数
def test(X_test, y_test, model):
    dtest = xgb.DMatrix(X_test)
    
    # 预测概率
    probs = model.predict(dtest)
    
    # 获取预测类别
    if probs.ndim > 1:  # 如果是概率矩阵
        preds = np.argmax(probs, axis=1)
    else:  # 如果已经是类别
        preds = probs
    
    # 计算各种指标
    accuracy = accuracy_score(y_test, preds) * 100
    
    # 计算加权F1、精确率和召回率
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0) * 100
    precision = precision_score(y_test, preds, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, preds, average='weighted', zero_division=0) * 100
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, preds)
    
    # 计算特异性（多类别情况下较复杂）
    # 这里使用一种简化方法：对每个类别计算特异性，然后取平均
    n_classes = len(np.unique(np.concatenate([y_test, preds])))
    specificities = []
    
    for i in range(n_classes):
        # 将问题转化为二分类：当前类别 vs 其他类别
        y_true_binary = (y_test == i).astype(int)
        y_pred_binary = (preds == i).astype(int)
        
        # 计算真阴性和假阳性
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        
        # 计算特异性
        if tn + fp == 0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
        
        specificities.append(specificity)
    
    # 计算平均特异性
    specificity = np.mean(specificities) * 100
    
    # 计算多类别AUC（使用OvR方法）
    macro_auc = 0.0
    weighted_auc = 0.0
    
    if probs.ndim > 1:
        try:
            # 获取测试集中实际存在的类别
            unique_classes = np.unique(y_test)
            
            # 创建one-hot编码的真实标签，但只包含实际存在的类别
            y_test_onehot = np.zeros((len(y_test), probs.shape[1]))
            for i in range(len(y_test)):
                if y_test[i] < probs.shape[1]:  # 确保类别索引在概率矩阵范围内
                    y_test_onehot[i, y_test[i]] = 1
            
            # 只计算实际存在类别的AUC
            aucs = []
            for cls in unique_classes:
                if cls < probs.shape[1]:  # 确保类别索引在概率矩阵范围内
                    fpr, tpr, _ = roc_curve(y_test_onehot[:, cls], probs[:, cls])
                    aucs.append(auc(fpr, tpr))
            
            if aucs:  # 如果有计算出的AUC值
                macro_auc = np.mean(aucs)
                
                # 计算加权AUC
                class_weights = np.array([np.sum(y_test == cls) for cls in unique_classes])
                weighted_auc = np.average(aucs, weights=class_weights)
        except Exception as e:
            print(f"计算AUC时出错: {e}")
    
    return f1, accuracy, precision, recall, specificity, macro_auc, weighted_auc, cm, probs, y_test
def plot_roc_curves(y_test, probs, n_classes):
    plt.figure(figsize=(10, 8))
    
    # 创建one-hot编码的真实标签
    y_test_onehot = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        y_test_onehot[i, y_test[i]] = 1
    
    # 为每个类别绘制ROC曲线
    for i in range(n_classes):
        if np.sum(y_test_onehot[:, i]) > 0:  # 确保该类别在测试集中存在
            fpr, tpr, _ = roc_curve(y_test_onehot[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'figure/XGB_32class_ROC_Curves.png')
    plt.close()

class new_shap_values():
    def __init__(self, shap_values, bool_tf=None, method='sum'):
        self.feature_names = shap_values.feature_names
        if method == 'sum':
            self.data = np.nansum(shap_values.data[bool_tf], axis=0)
            self.values = np.nansum(shap_values.values[bool_tf], axis=0)
            self.base_values = np.nansum(shap_values.base_values[bool_tf], axis=0)
        elif method == 'mean':
            self.data = np.nanmean(shap_values.data[bool_tf], axis=0)
            self.values = np.nanmean(shap_values.values[bool_tf], axis=0)
            self.base_values = np.nanmean(shap_values.base_values[bool_tf], axis=0)
        else:
            print('sry,not right method.')
            return
        self.explanation = shap.Explanation(values=self.values, data=self.data, feature_names=self.feature_names,
                                        base_values=self.base_values)

    def get_explanation(self):
        return self.explanation

if __name__ == '__main__':
    # 创建结果目录
    if not os.path.exists('figure'):
        os.makedirs('figure')
    if not os.path.exists('xgb_models'):
        os.makedirs('xgb_models')
    
    # 记录开始时间
    start_time = time.time()
    
    # 设置要编码的列
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    
    # 加载数据
    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

        # 将多标签转换为32类单标签
    multiclass_labels = multilabel_to_multiclass(labels)
    features_val = features.copy()
    labels_val = labels.copy()
    multiclass_labels_val = multilabel_to_multiclass(labels_val)
    
    # 打印类别分布
    unique_classes, counts = np.unique(multiclass_labels, return_counts=True)
    print("类别分布:")
    for cls, count in zip(unique_classes, counts):
        binary = format(cls, '05b')  # 转换为5位二进制
        print(f"类别 {cls} (二进制: {binary}): {count}个样本")
    
    # 特征处理
    # 方差过滤
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    features_val = selector.transform(features_val)

    # 更新特征名称
    mask = selector.get_support()
    all_feature_names = all_feature_names[mask]

    # 标准化
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8

    features = (features - mean_f) / std_f
    features_val = (features_val - mean_f) / (std_f + 1e-8)
    
    # 降维前的形状
    print("PCA降维前，训练集形状：", features.shape)
    print("PCA降维前，验证集形状：", features_val.shape)

    # PCA降维
    pca = PCA(n_components=0.95)
    
    # PCA后再次过滤零方差
    pca_selector = VarianceThreshold(threshold=0.01)
    features = pca_selector.fit_transform(pca.fit_transform(features))
    features_val = pca_selector.transform(pca.transform(features_val))

    # 更新PCA特征名称
    pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1] + 1)]
    print("PCA降维后，训练集形状：", features.shape)
    print("PCA降维后，验证集形状：", features_val.shape)
    
    # 更新 Shap_features 为 PCA 后的特征
    Shap_features = features_val.copy()

    # 交叉验证划分
    folds_data_index = split_data_5fold(features)

    num_x = len(Covariates_features)
    num_folds = len(folds_data_index)

    # 用于保存不同x的指标
    all_accuracies = [[] for _ in range(num_x)]
    all_precisions = [[] for _ in range(num_x)]
    all_recalls = [[] for _ in range(num_x)]
    all_f1_scores = [[] for _ in range(num_x)]
    all_macro_auc = [[] for _ in range(num_x)]
    all_weighted_auc = [[] for _ in range(num_x)]

    # 让用户输入想要运行的fold编号
    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    # 记录所有fold的结果
    all_fold_results = []
    
    # 创建日志文件
    log_filename = f'xgb_32class_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(log_filename, 'w') as f:
        f.write("fold,x,test_f1,test_accuracy,test_precision,test_recall,test_specificity,macro_auc,weighted_auc\n")

    # 计算类别数量
    n_classes = 2**len(columns_to_encode)  # 32类
    
    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  # 跳过未选择的fold
        
        for x in range(len(Covariates_features)):
            print(f"\n开始训练 Fold {fold+1}, x={x+1}")
            
            # 验证和测试集划分
            Validation_size = len(features_val) // 5
            Test_size = Validation_size

            indices = np.arange(len(features_val))
            np.random.shuffle(indices)
            validation_index = indices[(fold + 1) * Validation_size: (fold + 2) * Validation_size] if fold < 4 else indices[4 * Validation_size:]
            test_indices = indices[fold * Test_size: (fold + 1) * Test_size]

            trainX = features[train_index]
            trainY = multiclass_labels[train_index]
            valX = features_val[validation_index]
            valY = multiclass_labels_val[validation_index]
            testX = features_val[test_indices]
            testY = multiclass_labels_val[test_indices]
            
            # 设置XGBoost参数 - 多分类
            params = {
                'objective': 'multi:softprob',  # 多分类
                'eval_metric': 'mlogloss',      # 多分类损失
                'num_class': n_classes,         # 类别数量
                'eta': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'alpha': 0.1,  # L1正则化
                'lambda': 1.0,  # L2正则化
                'seed': 3407
            }
            
            # 创建DMatrix对象
            dtrain = xgb.DMatrix(trainX, label=trainY)
            dval = xgb.DMatrix(valX, label=valY)
            
            # 设置早停
            evallist = [(dtrain, 'train'), (dval, 'eval')]

            # 创建回调函数来记录训练和验证损失
            train_losses = []
            val_losses = []
            
            # 使用自定义回调类
            class LossCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    train_loss = evals_log['train']['mlogloss'][-1]
                    val_loss = evals_log['eval']['mlogloss'][-1]
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    return False
            
            # 训练模型
            num_round = 350  # 与原始模型的epoch数一致
            bst = xgb.train(params, dtrain, num_round, evallist, 
                            early_stopping_rounds=15, verbose_eval=50, callbacks=[LossCallback()])
            
            # 保存模型
            if not os.path.exists('xgb_models'):
                os.makedirs('xgb_models')
            bst.save_model(f'xgb_models/xgb_32class_model_fold{fold+1}_x{x+1}.json')
            
            # 绘制学习曲线
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
            plt.axvline(x=bst.best_iteration, color='r', linestyle='--', 
                       label=f'Best Iteration: {bst.best_iteration}')
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Log Loss')
            plt.title(f'XGBoost 32-Class Learning Curve (Fold {fold+1}, x={x+1})')
            plt.legend()
            plt.grid(True)
            
            # 确保figure目录存在
            if not os.path.exists('figure'):
                os.makedirs('figure')
                
            plt.savefig(f'figure/xgb_32class_learning_curve_fold{fold+1}_x{x+1}.png')
            plt.close()
            
            # 在测试集上评估
            test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, cm, all_probs, all_labels = test(testX, testY, bst)
            
            # 绘制混淆矩阵
            plot_confusion_matrix(cm, [f'Class_{i}' for i in range(n_classes)], fold, 0)
            
            # 绘制ROC曲线
            plot_roc_curves(all_labels, all_probs, n_classes)
            
            # 保存结果
            fold_result = {
                'fold': fold + 1,
                'x': x + 1,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_specificity': test_specificity,
                'macro_auc': macro_auc,
                'weighted_auc': weighted_auc
            }
            all_fold_results.append(fold_result)
            
            # 记录到日志文件
            with open(log_filename, 'a') as f:
                f.write(f"{fold+1},{x+1},{test_f1:.2f},{test_accuracy:.2f},{test_precision:.2f},{test_recall:.2f},{test_specificity:.2f},{macro_auc:.3f},{weighted_auc:.3f}\n")
            
            # 打印结果
            print(
                f'fold [{fold + 1}/5]: '
                f'Test F1: {test_f1:.2f}% | '
                f'Accuracy: {test_accuracy:.2f}% | '
                f'Precision: {test_precision:.2f}% | '
                f'Recall: {test_recall:.2f}% | '
                f'Specificity: {test_specificity:.2f}% | '
                f'Macro AUC: {macro_auc:.3f} | '
                f'Weighted AUC: {weighted_auc:.3f}'
            )
            
            # 将结果保存到Test.csv文件
            with open('XGB_32class_Test.csv', 'a') as f:
                f.write(
                    f'fold [{fold + 1}/5]: '
                    f'Test F1: {test_f1:.2f}% | '
                    f'Accuracy: {test_accuracy:.2f}% | '
                    f'Precision: {test_precision:.2f}% | '
                    f'Recall: {test_recall:.2f}% | '
                    f'Specificity: {test_specificity:.2f}% | '
                    f'Macro AUC: {macro_auc:.3f} | '
                    f'Weighted AUC: {weighted_auc:.3f}\n'
                )
            
            # 保存当前 x 和 fold 的指标
            all_accuracies[x].append(test_accuracy)
            all_precisions[x].append(test_precision)
            all_recalls[x].append(test_recall)
            all_f1_scores[x].append(test_f1)
            all_macro_auc[x].append(macro_auc)
            all_weighted_auc[x].append(weighted_auc)
    
    # 如果运行了所有fold，绘制比较图
    if len(selected_folds) == 5:
        # 绘制准确率折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_accuracies[x], label=f'Accuracy (x={x + 1})', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('Accuracy (%)')
        plt.title('XGBoost 32-Class: Comparison of Accuracy for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_32class_Comparison_of_Accuracy.png')
        plt.close()

        # 绘制精确率折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_precisions[x], label=f'Precision (x={x + 1})', marker='s')
        plt.xlabel('Fold Number')
        plt.ylabel('Precision (%)')
        plt.title('XGBoost 32-Class: Comparison of Precision for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_32class_Comparison_of_Precision.png')
        plt.close()

        # 绘制召回率折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_recalls[x], label=f'Recall (x={x + 1})', marker='^')
        plt.xlabel('Fold Number')
        plt.ylabel('Recall (%)')
        plt.title('XGBoost 32-Class: Comparison of Recall for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_32class_Comparison_of_Recall.png')
        plt.close()

        # 绘制 F1 分数折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_f1_scores[x], label=f'F1 Score (x={x + 1})', marker='D')
        plt.xlabel('Fold Number')
        plt.ylabel('F1 Score (%)')
        plt.title('XGBoost 32-Class: Comparison of F1 Score for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_32class_Comparison_of_F1_Score.png')
        plt.close()
        
        # 绘制宏平均AUC折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_macro_auc[x], label=f'Macro AUC (x={x + 1})', marker='*')
        plt.xlabel('Fold Number')
        plt.ylabel('Macro AUC')
        plt.title('XGBoost 32-Class: Comparison of Macro AUC for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_32class_Comparison_of_Macro_AUC.png')
        plt.close()
        
        # 绘制加权平均AUC折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_weighted_auc[x], label=f'Weighted AUC (x={x + 1})', marker='p')
        plt.xlabel('Fold Number')
        plt.ylabel('Weighted AUC')
        plt.title('XGBoost 32-Class: Comparison of Weighted AUC for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_32class_Comparison_of_Weighted_AUC.png')
        plt.close()
    
    # SHAP分析
    print("\n开始进行SHAP分析...")
    
    # 选择一个fold的模型进行SHAP分析
    analysis_fold = 0  # 使用第一个fold的模型
    if analysis_fold in selected_folds:
        # 使用PCA特征名称
        pca_feature_names = [f'PC{i}' for i in range(1, features_val.shape[1] + 1)]
        origin = pd.DataFrame(features_val, columns=pca_feature_names)
        COLUMNS = origin.columns
        
        # 使用SHAP对背景数据进行降采样
        K = 16
        background_data = shap.kmeans(features_val, K)
        
        # 随机选择样本进行SHAP分析
        selected_indices = np.random.choice(len(features_val), min(3501, len(features_val)), replace=True)
        selected_features_val = features_val[selected_indices]
        
        # 加载模型
        model_path = f'xgb_models/xgb_32class_model_fold{analysis_fold+1}_x{0+1}.json'
        if os.path.exists(model_path):
            model = xgb.Booster()
            model.load_model(model_path)
            
            print(f"正在使用SHAP评估32类模型的特征重要性...")
            
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(xgb.DMatrix(selected_features_val))
            
            # 对于多类别模型，shap_values是一个列表，每个元素对应一个类别
            if isinstance(shap_values, list):
                # 计算所有类别的平均SHAP值
                avg_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                
                # 创建SHAP解释对象
                shap_exp = shap.Explanation(
                    values=avg_shap_values, 
                    data=selected_features_val,
                    feature_names=pca_feature_names,
                    base_values=np.full(len(selected_features_val), explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value)
                )
            else:
                # 单类别情况
                shap_exp = shap.Explanation(
                    values=shap_values, 
                    data=selected_features_val,
                    feature_names=pca_feature_names,
                    base_values=np.full(len(selected_features_val), explainer.expected_value)
                )
            
            # 绘制SHAP汇总图
            plt.figure(figsize=(12, 8))
            shap.plots.bar(shap_exp, max_display=16, show=False)
            plt.title('XGBoost 32-Class SHAP Summary Plot (PCA Features)')
            plt.tight_layout()
            plt.savefig('figure/XGB_32class_SHAP_Summary_PCA.png')
            plt.close()
            
            # 绘制SHAP蜂群图
            plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(shap_exp, max_display=16, show=False)
            plt.title('XGBoost 32-Class SHAP Beeswarm Plot (PCA Features)')
            plt.tight_layout()
            plt.savefig('figure/XGB_32class_SHAP_Beeswarm_PCA.png')
            plt.close()
            
            # 特征映射 - 将PCA特征的重要性映射回原始特征空间
            # 获取PCA组件
            pca_components = pca.components_
            
            # 计算每个原始特征的重要性
            feature_importance = np.zeros(len(all_feature_names))
            
            # 计算每个SHAP值的平均绝对值
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # 将PCA特征的重要性映射回原始特征
            for j, importance in enumerate(mean_abs_shap):
                if j < len(pca_components):  # 确保索引在范围内
                    # 将PCA特征的重要性分配给原始特征
                    feature_importance += importance * np.abs(pca_components[j])
            
            # 创建原始特征重要性的DataFrame
            original_importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importance
            })
            original_importance_df = original_importance_df.sort_values('Importance', ascending=False)
            
            # 绘制原始特征重要性图
            plt.figure(figsize=(15, 6))
            top_features = original_importance_df.head(min(64, len(original_importance_df)))
            plt.bar(top_features['Feature'], top_features['Importance'])
            plt.xlabel('Features')
            plt.ylabel('Mapped Importance')
            plt.title('XGBoost 32-Class Mapped Feature Importance')
            plt.xticks(rotation=45, fontsize=6)
            plt.tight_layout()
            plt.savefig('figure/XGB_32class_Mapped_Feature_Importance.png')
            plt.close()
            
            # 获取XGBoost模型的原生特征重要性
            importance = model.get_score(importance_type='gain')
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # 绘制XGBoost原生特征重要性图
            plt.figure(figsize=(15, 6))
            xgb_top_features = importance_df.head(min(64, len(importance_df)))
            plt.bar(xgb_top_features['Feature'], xgb_top_features['Importance'])
            plt.xlabel('PCA Features')
            plt.ylabel('Importance')
            plt.title('XGBoost 32-Class Native Feature Importance')
            plt.xticks(rotation=45, fontsize=6)
            plt.tight_layout()
            plt.savefig('figure/XGB_32class_Native_Feature_Importance.png')
            plt.close()
        else:
            print(f"警告：模型文件 {model_path} 不存在")
    
    # 计算运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nXGBoost 32类模型训练和评估完成！")
    print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    # 保存总结果
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('XGB_32class_all_results.csv', index=False)
    
    print("\n所有结果已保存到 XGB_32class_all_results.csv")
