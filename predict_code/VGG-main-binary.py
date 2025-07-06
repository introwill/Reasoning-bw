import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
from Model import Model_Binary_VGG
import torch.optim as optim
from numpy import argmax
from sklearn.metrics import precision_recall_curve
from Model import mlsmote
import matplotlib
import matplotlib.pyplot as plt
from temperature_scaling import ModelWithTemperature
import shap
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os
from datetime import datetime

matplotlib.use('TKAgg')
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
    def __init__(self, features, labels, class_idx=None):
        self.Data = features
        if class_idx is not None:
            # 只取指定类别的标签
            self.label = labels[:, class_idx:class_idx+1]
        else:
            self.label = labels

    def __getitem__(self, index):
        return self.Data[index], self.label[index]

    def __len__(self):
        return len(self.Data)

# 数据标准化与交叉验证划分
def split_data_5fold(input_data, original_size=None):
    np.random.seed(3407)
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    fold_size = len(input_data) // 5
    
    # 如果提供了原始数据集大小，则使用它来限制验证和测试索引
    if original_size is not None:
        orig_fold_size = original_size // 5
    else:
        orig_fold_size = fold_size
    
    folds_data_index = []
    for i in range(5):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        validation_indices = indices[(i + 1) * fold_size: (i + 2) * fold_size] if i < 4 else indices[4 * fold_size:]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 2) * fold_size:]]) if i < 4 else indices[:4 * fold_size]
        
        # 确保验证和测试索引不超过原始数据集大小
        if original_size is not None:
            test_indices = test_indices[test_indices < original_size]
            validation_indices = validation_indices[validation_indices < original_size]
        
        folds_data_index.append((train_indices, validation_indices, test_indices))
    return folds_data_index

# 阈值计算函数
def Probs_Switcher(output, labels):
    probs_sigmoid = torch.sigmoid(output)
    probs = probs_sigmoid.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()
    
    precision, recall, thresholds = precision_recall_curve(labels_np, probs)
    precision = precision * 0.6
    recall = recall
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    index = argmax(f1_scores)
    
    if len(thresholds) > index:
        return thresholds[index]
    else:
        return 0.5  # 默认阈值

# 指标计算函数
def f1_score_func(logits, labels, threshold):
    probs_sigmoid = torch.sigmoid(logits)
    probs = probs_sigmoid.detach().cpu().numpy()
    labels = labels.cpu().detach().numpy()

    preds = np.float64(probs >= threshold)

    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    recall = recall_score(labels, preds, average='binary', zero_division=0)

    # 特异性计算
    cm = multilabel_confusion_matrix(labels, preds)
    tn, fp = cm[0][0, 0], cm[0][0, 1]
    if (tn + fp) == 0:
        specificity = 0.0
    else:
        specificity = tn / (tn + fp)

    return f1 * 100, accuracy * 100, precision * 100, recall * 100, preds, specificity * 100

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

# 测试函数
def test(Test_data_loader, threshold, model, device):
    model.eval()
    n = 0

    # 初始化存储概率和标签的变量
    all_probs = []
    all_labels = []

    with torch.no_grad():
        Test_F1 = 0
        Test_accuracy = 0
        Test_precision = 0
        Test_recall = 0
        Test_specificity = 0
        for data, target in Test_data_loader:
            n += 1
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)

            # 收集概率和标签
            probs_sigmoid = torch.sigmoid(output)
            all_probs.append(probs_sigmoid.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(output, target, threshold)

            Test_F1 += test_f1
            Test_accuracy += test_accuracy
            Test_precision += test_precision
            Test_recall += test_recall
            Test_specificity += test_specificity

        # 计算AUC-ROC
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算AUC
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except ValueError:
            print("Class has only one class present in test set.")
            auc_score = float('nan')

        Test_F1 /= n
        Test_accuracy /= n
        Test_precision /= n
        Test_recall /= n
        Test_specificity /= n

        return Test_F1, Test_accuracy, Test_precision, Test_recall, Test_specificity, auc_score, all_probs, all_labels

if __name__ == '__main__':  
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']

    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

    # 测试集的数据不要改定义！！！（一定要原始的数据集）
    features_val = features
    labels_val = labels

    Shap_features = features_val

    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)

    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)
    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)  # Getting minority instance of that datframe
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)  # Applying MLSMOTE to augment the dataframe

    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)

    # 添加方差过滤（阈值设为0.01）
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    features_val = selector.transform(features_val)

    # 更新特征名称
    mask = selector.get_support()
    all_feature_names = all_feature_names[mask]

    # 重新计算标准化所需的均值、标准差
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

    # 重新计算PCA后数据的均值和标准差
    mean_pca, std_pca = np.mean(features, axis=0), np.std(features, axis=0)

    folds_data_index = split_data_5fold(features, original_size=len(features_val))

    num_x = len(Covariates_features)
    num_folds = len(folds_data_index)
    num_classes = len(columns_to_encode)

    # 用于保存不同类别的指标
    all_metrics = {
        'accuracy': [[] for _ in range(num_classes)],
        'precision': [[] for _ in range(num_classes)],
        'recall': [[] for _ in range(num_classes)],
        'f1': [[] for _ in range(num_classes)],
        'specificity': [[] for _ in range(num_classes)],
        'auc': [[] for _ in range(num_classes)]
    }

    # 创建保存模型的目录
    if not os.path.exists('binary_models'):
        os.makedirs('binary_models')
    
    # 创建保存图表的目录
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 让用户输入想要运行的fold编号，用逗号分隔
    selected_folds_input = '1,2,3,4,5'  # 默认运行所有fold
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    # 记录开始时间
    start_time = datetime.now()
    print(f"开始训练时间: {start_time}")

    # 为每个类别训练独立的二分类模型
    for class_idx in range(num_classes):
        print(f"\n{'='*50}")
        print(f"开始训练类别 {columns_to_encode[class_idx]} 的模型")
        print(f"{'='*50}")
        
        class_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': [],
            'auc': []
        }
        
        for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
            if fold not in selected_folds:
                continue  # 跳过未选择的fold
            
            print(f"\n{'-'*40}")
            print(f"训练类别 {columns_to_encode[class_idx]} 的模型 - Fold {fold + 1}")
            print(f"{'-'*40}")
            
            # 设置超参数
            l1_weight = 0.070
            num_epochs = 350
            batch_size = 256
            
            # 创建二分类模型
            model = Model_Binary_VGG.BinaryVGG11(in_channels=features.shape[1], epoch=num_epochs)
            
            # 设置损失函数、优化器和学习率调度器
            # 计算正样本权重
            pos_count = np.sum(labels[:, class_idx])
            neg_count = len(labels) - pos_count
            pos_weight = torch.tensor(neg_count / (pos_count + 1e-8))
            
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.475)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # 打印模型参数数量
            total_num, trainable_num = get_parameter_number(model)
            print(f'类别 {columns_to_encode[class_idx]}, Fold {fold + 1}, 总参数: {total_num}, 可训练参数: {trainable_num}')
            
            # 准备数据
            trainX = features[train_index]
            trainY = labels[train_index, class_idx:class_idx+1]  # 只取当前类别
            valX = features_val[validation_index]
            valY = labels_val[validation_index, class_idx:class_idx+1]
            testX = features_val[test_indices]
            testY = labels_val[test_indices, class_idx:class_idx+1]
            
            # 创建数据加载器
            Train_data = NetDataset(trainX, trainY)
            Validation_data = NetDataset(valX, valY)
            Test_data = NetDataset(testX, testY)
            
            Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
            Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)
            
            # 用于绘制学习曲线的列表
            train_losses = []
            valid_f1_history = []
            valid_accuracy_history = []
            valid_precision_history = []
            valid_recall_history = []
            valid_specificity_history = []  # 添加特异性历史记录
            valid_balanced_acc_history = []  # 添加平衡准确率历史记录
            
            # 早停相关变量 - 修改为使用平衡准确率
            best_valid_balanced_acc = 0
            patience = 15
            counter = 0
            best_epoch = 0
            best_threshold = 0.5
            
            # 训练循环
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(Train_data_loader):
                    data, target = data.float().to(device), target.float().to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # L1正则化
                    l1_criterion = nn.L1Loss()
                    l1_loss = 0
                    for param in model.parameters():
                        l1_loss += l1_criterion(param, torch.zeros_like(param))
                    loss += l1_weight * l1_loss
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)
                
                average_train_loss = train_loss / len(Train_data_loader.dataset)
                scheduler.step()
                
                # 记录训练损失
                train_losses.append(average_train_loss)
                
                # 验证
                model.eval()
                n = 0
                valid_f1 = 0
                valid_accuracy = 0
                valid_precision = 0
                valid_recall = 0
                valid_specificity = 0
                valid_balanced_acc = 0  # 添加平衡准确率变量
                
                with torch.no_grad():
                    for data, target in Validation_data_loader:
                        n += 1
                        data, target = data.float().to(device), target.float().to(device)
                        output = model(data)
                        
                        # 计算最佳阈值
                        if epoch % 5 == 0 and n == 1:  # 每5个epoch更新一次阈值
                            threshold = Probs_Switcher(output, target)
                        else:
                            threshold = best_threshold
                        
                        f1, accuracy, precision, recall, preds, specificity = f1_score_func(output, target, threshold)
                        
                        # 计算平衡准确率 (敏感性和特异性的平均值)
                        balanced_acc = (recall + specificity) / 2

                        valid_f1 += f1
                        valid_accuracy += accuracy
                        valid_precision += precision
                        valid_recall += recall
                        valid_specificity += specificity
                        valid_balanced_acc += balanced_acc  # 累加平衡准确率
                    
                    valid_f1 /= n
                    valid_accuracy /= n
                    valid_precision /= n
                    valid_recall /= n
                    valid_specificity /= n
                    valid_balanced_acc /= n  # 计算平均平衡准确率
                
                # 记录验证集指标
                valid_f1_history.append(valid_f1)
                valid_accuracy_history.append(valid_accuracy)
                valid_precision_history.append(valid_precision)
                valid_recall_history.append(valid_recall)
                valid_specificity_history.append(valid_specificity)  # 记录特异性
                valid_balanced_acc_history.append(valid_balanced_acc)  # 记录平衡准确率
                
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}]: 平均训练损失: {average_train_loss:.4f}, '
                    f'验证F1: {valid_f1:.2f}%, 验证准确率: {valid_accuracy:.2f}%, '
                    f'验证精确率: {valid_precision:.2f}%, 验证召回率: {valid_recall:.2f}%, '
                    f'验证特异性: {valid_specificity:.2f}%'
                )
                
                # 早停机制
                if valid_balanced_acc > best_valid_balanced_acc:
                    best_valid_balanced_acc = valid_balanced_acc
                    counter = 0
                    best_epoch = epoch
                    best_threshold = threshold
                    # 保存最佳模型
                    torch.save(model.state_dict(), f'binary_models/best_model_class{class_idx}_fold{fold}.ckpt')
                else:
                    counter += 1
                    print(f'早停计数器: {counter}/{patience}')
                    if counter >= patience:
                        print(f'早停触发于epoch {epoch+1}. 最佳epoch为 {best_epoch+1}, 平衡准确率: {best_valid_balanced_acc:.2f}%')
                        break
            
            # 绘制学习曲线
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 1, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
            plt.xlabel('Epochs')
            plt.ylabel('loss')
            plt.title(f'train-loss (class {columns_to_encode[class_idx]}, Fold {fold+1})')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, len(valid_f1_history) + 1), valid_f1_history, label='验证F1', color='red')
            plt.plot(range(1, len(valid_accuracy_history) + 1), valid_accuracy_history, label='验证准确率', color='blue')
            plt.plot(range(1, len(valid_precision_history) + 1), valid_precision_history, label='验证精确率', color='green')
            plt.plot(range(1, len(valid_recall_history) + 1), valid_recall_history, label='验证召回率', color='purple')
            plt.plot(range(1, len(valid_specificity_history) + 1), valid_specificity_history, label='验证特异性', color='orange')
            plt.plot(range(1, len(valid_balanced_acc_history) + 1), valid_balanced_acc_history, label='平衡准确率', color='brown', linewidth=2)
            plt.axvline(x=best_epoch+1, color='black', linestyle='--', label=f'最佳Epoch ({best_epoch+1})')
            plt.xlabel('Epochs')
            plt.ylabel('sorce (%)')
            plt.title(f'vaild metrics (class {columns_to_encode[class_idx]}, Fold {fold+1})')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'figure/learning_curve_class{class_idx}_fold{fold+1}.png')
            plt.close()
            
            # 测试最佳模型
            model.load_state_dict(torch.load(f'binary_models/best_model_class{class_idx}_fold{fold}.ckpt'))
            test_f1, test_accuracy, test_precision, test_recall, test_specificity, test_auc, _, _ = test(
                Test_data_loader, best_threshold, model, device
            )
            
            print(f"\n测试结果 - 类别 {columns_to_encode[class_idx]}, Fold {fold+1}:")
            print(f"F1分数: {test_f1:.2f}%")
            print(f"准确率: {test_accuracy:.2f}%")
            print(f"精确率: {test_precision:.2f}%")
            print(f"召回率: {test_recall:.2f}%")
            print(f"特异性: {test_specificity:.2f}%")
            print(f"AUC: {test_auc:.4f}")
            
            # 保存测试指标
            class_metrics['accuracy'].append(test_accuracy)
            class_metrics['precision'].append(test_precision)
            class_metrics['recall'].append(test_recall)
            class_metrics['f1'].append(test_f1)
            class_metrics['specificity'].append(test_specificity)
            class_metrics['auc'].append(test_auc)
        
        # 计算当前类别的平均指标
        for metric_name, values in class_metrics.items():
            avg_value = np.mean(values)
            all_metrics[metric_name][class_idx] = avg_value
            print(f"类别 {columns_to_encode[class_idx]} 平均 {metric_name}: {avg_value:.2f}")
    
    # 打印所有类别的平均指标
    print("\n所有类别的平均指标:")
    for class_idx in range(num_classes):
        print(f"\n类别 {columns_to_encode[class_idx]}:")
        for metric_name, values in all_metrics.items():
            print(f"{metric_name}: {values[class_idx]:.2f}")
    
    # 计算总体平均指标
    print("\n总体平均指标:")
    for metric_name, values in all_metrics.items():
        print(f"平均 {metric_name}: {np.mean(values):.2f}")
    
    # 记录结束时间
    end_time = datetime.now()
    print(f"\n训练结束时间: {end_time}")
    print(f"总训练时间: {end_time - start_time}")
    
    # 绘制所有类别的性能对比图
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc']
    x = np.arange(len(columns_to_encode))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i*width, [all_metrics[metric][j] for j in range(num_classes)], 
                width=width, label=metric.capitalize())
    
    plt.xlabel('class')
    plt.ylabel('sorce')
    plt.title('Performance Comparison Across Classes')
    plt.xticks(x + width*2.5, columns_to_encode)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('figure/class_performance_comparison.png')
    plt.close()