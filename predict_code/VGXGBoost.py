import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
from Model import Model_VGXGBoost
import torch.optim as optim
from numpy import argmax
from sklearn.metrics import precision_recall_curve
from Model import mlsmote
import matplotlib
import matplotlib.pyplot as plt
from temperature_scaling import ModelWithTemperature
from Multilable_temerature_scaling  import ModelWithTemperature
import shap
import seaborn as sns  # 添加seaborn库用于更美观的混淆矩阵绘制
from sklearn.metrics import roc_auc_score, roc_curve, auc
import statsmodels.formula.api as smf
import os
# 添加XGBoost相关依赖
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

matplotlib.use('TKAgg')  # 用于解决绘图时的报错问题python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
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

# ... [其他函数保持不变] ...

# 添加VGG特征提取器类
class VGGFeatureExtractor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()  # 设置为评估模式
        
    def extract_features(self, data_loader):
        features = []
        labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.float().to(self.device)
                # 获取VGG的特征提取部分输出
                feature = self.model.extract_features(data)
                features.append(feature.cpu().numpy())
                labels.append(target.numpy())
                
        return np.vstack(features), np.vstack(labels)

# 添加XGBoost训练和评估函数
def train_xgboost(X_train, y_train, X_val=None, y_val=None, param_grid=None):
    """
    训练XGBoost模型
    
    参数:
    X_train -- 训练特征
    y_train -- 训练标签
    X_val -- 验证特征 (可选)
    y_val -- 验证标签 (可选)
    param_grid -- 参数网格 (可选)
    
    返回:
    训练好的XGBoost模型
    """
    if param_grid is None:
        param_grid = {
            'estimator__learning_rate': [0.1],
            'estimator__max_depth': [3, 5],
            'estimator__n_estimators': [100],
            'estimator__subsample': [0.8],
            'estimator__colsample_bytree': [0.8],
            'estimator__objective': ['binary:logistic']
        }
    
    # 创建基础XGBoost分类器
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # 创建多输出分类器
    multi_target_xgb = MultiOutputClassifier(xgb_clf)
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        multi_target_xgb,
        param_grid,
        cv=3,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1
    )
    
    # 训练模型
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 如果提供了验证集，在验证集上评估
    if X_val is not None and y_val is not None:
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        
        # 计算各种评估指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        print(f"验证集性能:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 详细的分类报告
        print("\n分类报告:")
        print(classification_report(y_val, y_pred, zero_division=0))
    
    return grid_search.best_estimator_

# 评估XGBoost模型
def evaluate_xgboost(model, X_test, y_test, class_names):
    """
    评估XGBoost模型并生成可视化
    
    参数:
    model -- 训练好的XGBoost模型
    X_test -- 测试特征
    y_test -- 测试标签
    class_names -- 类别名称列表
    """
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)
    # 预测类别
    y_pred = model.predict(X_test)
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"测试集性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 详细的分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 计算每个类别的AUC
    auc_scores = []
    for i in range(y_test.shape[1]):
        try:
            # 获取当前类别的预测概率
            class_proba = np.array([estimator.predict_proba(X_test)[:, 1] for estimator in model.estimators_])[i]
            auc_score = roc_auc_score(y_test[:, i], class_proba)
            auc_scores.append(auc_score)
        except ValueError:
            print(f"类别 {i} 在测试集中只有一个类别，无法计算AUC。")
            auc_scores.append(float('nan'))
    
    # 计算宏平均和加权平均AUC
    macro_auc = np.nanmean(auc_scores)
    
    print(f"宏平均AUC: {macro_auc:.4f}")
    print(f"各类别AUC: {auc_scores}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        if not np.isnan(auc_scores[i]):
            # 获取当前类别的预测概率
            class_proba = np.array([estimator.predict_proba(X_test)[:, 1] for estimator in model.estimators_])[i]
            fpr, tpr, _ = roc_curve(y_test[:, i], class_proba)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_scores[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curves')
    plt.legend(loc="lower right")
    
    # 确保figure目录存在
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    plt.savefig('figure/xgboost_roc_curves.png')
    plt.close()
    
    # 绘制混淆矩阵
    for i in range(len(class_names)):
        cm = multilabel_confusion_matrix(y_test, y_pred)[i]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {class_names[i]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'figure/xgboost_confusion_matrix_{class_names[i]}.png')
        plt.close()
    
    return accuracy, precision, recall, f1, macro_auc, auc_scores

# 修改Model_VGG.py中的VGG11类，添加特征提取方法
# 注意：这里假设您已经在Model_VGG.py中添加了extract_features方法
# 如果没有，您需要修改Model_VGG.py文件

if __name__ == '__main__':  
    # columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    # columns_to_encode = ['MCQ160C', 'MCQ160F']
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    # columns_to_encode = ['MCQ160B', 'MCQ160D']

    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

    # 测试集的数据不要改定义！！！（一定要原始的数据集）
    features_val = features
    labels_val = labels

    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)

    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)
    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)  # Getting minority instance of that datframe
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)  # Applying MLSMOTE to augment the dataframe

    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)

    # 数据标准化
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8

    features = (features - mean_f) / std_f
    features_val = (features_val - mean_f) / (std_f + 1e-8)
    features = features.reshape(features.shape[0], -1)
    
    # 降维前的形状
    print("PCA降维前，训练集形状：", features.shape)
    print("PCA降维前，验证集形状：", features_val.shape)

    # 加入PCA降维，保留95%的信息
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    
    pca = PCA(n_components=0.95)
    
    # PCA后再次过滤零方差
    pca_selector = VarianceThreshold(threshold=0.01)
    features = pca_selector.fit_transform(pca.fit_transform(features))
    features_val = pca_selector.transform(pca.transform(features_val))

    # 更新PCA特征名称
    pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1] + 1)]
    print("PCA降维后，训练集形状：", features.shape)
    print("PCA降维后，验证集形状：", features_val.shape)
    
    # 重新计算PCA后数据的均值和标准差
    mean_pca, std_pca = np.mean(features, axis=0), np.std(features, axis=0)

    folds_data_index = split_data_5fold(features)

    num_classes = len(columns_to_encode)
    
    # 让用户输入想要运行的fold编号，用逗号分隔
    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    # 存储所有fold的结果
    all_xgb_results = []

    # 为验证集和测试集创建单独的索引
    folds_val_index = split_data_5fold(features_val)
    
    for fold, (train_index, _, _) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  # 跳过未选择的fold
            
        print(f"\n=== 处理 Fold {fold + 1} ===")
        
        # 获取验证集和测试集的索引
        _, validation_index, test_indices = folds_val_index[fold]
        
        # 准备数据
        trainX = features[train_index]
        trainY = labels[train_index]
        valX = features_val[validation_index]  # 使用正确的验证集索引
        valY = labels_val[validation_index]    # 使用正确的验证集索引
        testX = features_val[test_indices]     # 使用正确的测试集索引
        testY = labels_val[test_indices]       # 使用正确的测试集索引
        
        # 创建数据加载器
        batch_size = 256
        Train_data = NetDataset(trainX, trainY)
        Validation_data = NetDataset(valX, valY)
        Test_data = NetDataset(testX, testY)
        
        Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=False, drop_last=False)
        Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        
        # 初始化VGG模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model_VGXGBoost.VGG11(num_classes=num_classes, in_channels=features.shape[1], epoch=100)
        model.to(device)
        
        # 训练VGG模型（仅用于特征提取）
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0), reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.475)
        
        print("训练VGG特征提取器...")
        num_epochs = 50  # 减少训练轮数，因为我们只需要特征提取器
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(Train_data_loader):
                data, target = data.float().to(device), target.float().to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            
            average_train_loss = train_loss / len(Train_data_loader.dataset)
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}]: Average Train Loss: {average_train_loss:.4f}')
        
        # 保存VGG特征提取器
        torch.save(model.state_dict(), f'single/vgg_feature_extractor_fold{fold+1}.ckpt')
        
        # 创建特征提取器
        feature_extractor = VGGFeatureExtractor(model, device)
        
        # 提取特征
        print("提取VGG特征...")
        X_train_vgg, y_train_vgg = feature_extractor.extract_features(Train_data_loader)
        X_val_vgg, y_val_vgg = feature_extractor.extract_features(Validation_data_loader)
        X_test_vgg, y_test_vgg = feature_extractor.extract_features(Test_data_loader)
        
        print(f"VGG特征形状 - 训练集: {X_train_vgg.shape}, 验证集: {X_val_vgg.shape}, 测试集: {X_test_vgg.shape}")
        
        # 训练XGBoost模型
        print("训练XGBoost模型...")
        xgb_model = train_xgboost(X_train_vgg, y_train_vgg, X_val_vgg, y_val_vgg)
        
        # 评估XGBoost模型
        print("评估XGBoost模型...")
        accuracy, precision, recall, f1, macro_auc, auc_scores = evaluate_xgboost(
            xgb_model, X_test_vgg, y_test_vgg, columns_to_encode
        )
        
        # 保存结果
        fold_results = {
            'fold': fold + 1,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'macro_auc': macro_auc,
            'auc_scores': auc_scores
        }
        all_xgb_results.append(fold_results)
        
        # 保存结果到CSV
        with open('xgboost_results.csv', 'a') as f:
            f.write(f"Fold {fold+1},"
                   f"{accuracy*100:.2f}%,"
                   f"{precision*100:.2f}%,"
                   f"{recall*100:.2f}%,"
                   f"{f1*100:.2f}%,"
                   f"{macro_auc:.4f},"
                   f"{','.join([f'{auc:.4f}' for auc in auc_scores])}\n")
    
    # 计算平均性能
    if all_xgb_results:
        avg_accuracy = np.mean([res['accuracy'] for res in all_xgb_results])
        avg_precision = np.mean([res['precision'] for res in all_xgb_results])
        avg_recall = np.mean([res['recall'] for res in all_xgb_results])
        avg_f1 = np.mean([res['f1'] for res in all_xgb_results])
        avg_macro_auc = np.mean([res['macro_auc'] for res in all_xgb_results])
        
        print("\n=== XGBoost平均性能 ===")
        print(f"平均准确率: {avg_accuracy:.2f}%")
        print(f"平均精确率: {avg_precision:.2f}%")
        print(f"平均召回率: {avg_recall:.2f}%")
        print(f"平均F1分数: {avg_f1:.2f}%")
        print(f"平均宏平均AUC: {avg_macro_auc:.4f}")
        
        # 保存平均结果到CSV
        with open('xgboost_results.csv', 'a') as f:
            f.write(f"Average,"
                   f"{avg_accuracy:.2f}%,"
                   f"{avg_precision:.2f}%,"
                   f"{avg_recall:.2f}%,"
                   f"{avg_f1:.2f}%,"
                   f"{avg_macro_auc:.4f},\n")
        
        # 绘制性能比较图
        plt.figure(figsize=(12, 8))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        vgg_values = [avg_accuracy, avg_precision, avg_recall, avg_f1]
        
        plt.bar(metrics, vgg_values, color='blue', alpha=0.7, label='VGG+XGBoost')
        
        plt.ylabel('Score (%)')
        plt.title('VGG+XGBoost Performance Metrics')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig('figure/xgboost_performance.png')
        plt.close()