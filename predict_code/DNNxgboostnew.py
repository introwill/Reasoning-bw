import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
from Model import Model_DNN
import torch.optim as optim
from numpy import argmax
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
from temperature_scaling import ModelWithTemperature
from Multilable_temerature_scaling import ModelWithTemperature
import shap
import seaborn as sns
import statsmodels.formula.api as smf
import os
import xgboost as xgb  # 导入XGBoost
matplotlib.use('TKAgg')

seed_value = 3407
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

# ---------- 标签处理函数 ----------
def encode_labels(File, columns_to_encode):
    one_hot_encoded_df = pd.get_dummies(File[columns_to_encode], columns=columns_to_encode, prefix_sep='_')
    selected_columns = [col for col in one_hot_encoded_df.columns if col.endswith('_1')]
    filtered_df = one_hot_encoded_df[selected_columns].to_numpy()
    return filtered_df

def multilabel_to_multiclass(labels):
    """
    将多标签数据（one-hot矩阵）转换为多类别编号（0～31），
    每行视作二进制数，转换为10进制数字
    """
    n_samples = labels.shape[0]
    multiclass_labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        binary_str = ''.join(map(str, labels[i].astype(int)))
        multiclass_labels[i] = int(binary_str, 2)
    return multiclass_labels

def open_excel(filename, columns_to_encode):
    readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
    nplist = readbook.T.to_numpy()
    data = nplist[1:-5].T.astype(np.float64)
    target = encode_labels(readbook, columns_to_encode=columns_to_encode)
    all_feature_names = readbook.columns[1:-5]
    Covariates_features = readbook.columns[1:4]
    print(all_feature_names)
    print(Covariates_features)
    return data, target, all_feature_names, Covariates_features

# ---------- 数据集类 ----------
class NetDataset(Dataset):
    def __init__(self, features, labels, multiclass=True):
        self.features = features
        if multiclass:
            # 转换为单一类别标签
            self.labels = multilabel_to_multiclass(labels)
        else:
            self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        # 返回特征与整数标签
        return self.features[idx], self.labels[idx]

# ---------- 数据划分 ----------
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

# ---------- 绘制混淆矩阵 ----------
def plot_confusion_matrix(cm, class_names, fold, epoch, is_sum=False):
    plt.figure(figsize=(10,8))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if not os.path.exists('figure'):
        os.makedirs('figure')
    if is_sum:
        plt.savefig(f'figure/confusion_matrix_fold{fold+1}_sum.png')
    else:
        plt.savefig(f'figure/confusion_matrix_fold{fold+1}_epoch{epoch+1}.png')
    plt.close()

# ---------- 指标计算函数（多分类）----------
def f1_score_func(logits, labels):
    # logits为模型真实输出（未经过softmax），labels为整数标签
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)
    labels_np = labels.cpu().detach().numpy()
    f1 = f1_score(labels_np, preds, average='weighted', zero_division=0) * 100
    accuracy = accuracy_score(labels_np, preds) * 100
    precision = precision_score(labels_np, preds, average='weighted', zero_division=0) * 100
    recall = recall_score(labels_np, preds, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(labels_np, preds)
    # 特异性通常不适用于多分类，这里取每类TN率再平均
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i,:], i))
        specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
        specificities.append(specificity)
    specificity = np.mean(specificities) * 100
    return f1, accuracy, precision, recall, preds, specificity

# ---------- 获取参数数量 ----------
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

# ---------- 预测函数 ----------
def predict(data):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(data)
        preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
        return preds.cpu().numpy()

# ---------- 绘制ROC曲线（多分类时用One-vs-Rest）----------
def plot_roc_curves(all_labels, all_probs, class_names):
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    all_labels = label_binarize(all_labels, classes=range(n_classes))
    plt.figure(figsize=(10,8))
    for i in range(n_classes):
        if np.sum(all_labels[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    if not os.path.exists('figure'):
        os.makedirs('figure')
    plt.savefig('figure/ROC_Curves.png')
    plt.close()

# ---------- XGBoost评估函数（多分类）----------
def evaluate_xgboost(probs, labels):
    # probs shape: (n_samples, num_classes)
    preds = np.argmax(probs, axis=1)
    f1 = f1_score(labels, preds, average='weighted') * 100
    accuracy = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, average='weighted', zero_division=0) * 100
    recall = recall_score(labels, preds, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(labels, preds)
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i,:], i))
        spec = tn / (tn+fp) if (tn+fp)>0 else 0.0
        specificities.append(spec)
    specificity = np.mean(specificities) * 100
    return f1, accuracy, precision, recall, specificity, preds

# ---------- 主程序 ----------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    columns_to_encode = ['MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']
    # 读取数据（目标为one-hot形式）
    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)
    # 转换目标为32类（假设原one-hot有5列，对应2^5=32类）
    multiclass_labels = multilabel_to_multiclass(labels)
    
    # 标准化、降维：这里使用VarianceThreshold和PCA处理
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    mask = selector.get_support()
    all_feature_names = all_feature_names[mask]
    
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    std_f[std_f==0] = 1e-8
    features = (features - mean_f) / std_f
    print("PCA降维前，训练集形状：", features.shape)
    
    # PCA降维保留95%信息，并再次过零方差过滤
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    features = pca.fit_transform(features)
    pca_selector = VarianceThreshold(threshold=0.01)
    features = pca_selector.fit_transform(features)
    pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1]+1)]
    print("PCA降维后，训练集形状：", features.shape)
    
    # 使用原始验证数据（未经增广）的features和labels
    features_val = features.copy()
    labels_val = labels  # 注意：后续数据集类会转换为32类标签
    
    folds_data_index = split_data_5fold(features)
    num_classes = 32
    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(f.strip())-1 for f in selected_folds_input.split(',')]
    
    # 循环不同fold以及（可选）协变量（本例中固定协变量数量为0，仅训练单一模型）
    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue
            
        l1_weight = 0.07
        num_epochs = 350
        batch_size = 256
        
        # 使用DNN模型，输出维度设置为32
        print(f"Fold {fold+1}：训练DNN模型进行32类多分类预测...")
        model = Model_DNN.DNN(input_dim=features.shape[1],
                              hidden_dims=[512,256,128,64],
                              output_dim=num_classes,
                              dropout_rate=0.3)
        # 温度缩放模型（对多分类同样适用）
        scaled_model = ModelWithTemperature(model, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.475)
        model.to(device)
        
        total_num, trainable_num = get_parameter_number(model)
        print(f'Total params: {total_num}, Trainable params: {trainable_num}')
        
        Validation_size = len(features_val) // 5
        Test_size = Validation_size
        
        indices = np.arange(len(features_val))
        np.random.shuffle(indices)
        validation_index = indices[(fold+1)*Validation_size: (fold+2)*Validation_size] if fold < 4 else indices[4*Validation_size:]
        test_indices = indices[fold*Test_size: (fold+1)*Test_size]
        
        trainX = features[train_index]
        trainY = labels[train_index]
        valX = features_val[validation_index]
        valY = labels_val[validation_index]
        testX = features_val[test_indices]
        testY = labels_val[test_indices]
        
        Train_data = NetDataset(trainX, trainY)  # 返回整数标签
        Validation_data = NetDataset(valX, valY)
        Test_data = NetDataset(testX, testY)
        
        Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
        Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        
        best_valid_f1 = 0
        counter = 0
        best_epoch = 0
        valid_f1_history = []
        valid_accuracy_history = []
        valid_precision_history = []
        valid_recall_history = []
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for data, target in Train_data_loader:
                data, target = data.float().to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                # L1正则化
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.norm(param,1)
                loss += l1_weight * l1_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            average_train_loss = train_loss / len(Train_data_loader.dataset)
            scheduler.step()
            
            model.eval()
            n = 0
            valid_f1 = 0
            valid_accuracy = 0
            valid_precision = 0
            valid_recall = 0
            all_preds = np.array([])
            all_targets = np.array([])
            with torch.no_grad():
                for data, target in Validation_data_loader:
                    n += 1
                    data, target = data.float().to(device), target.to(device)
                    output = model(data)
                    f1, accuracy, precision, recall, preds, _ = f1_score_func(output, target)
                    valid_f1 += f1
                    valid_accuracy += accuracy
                    valid_precision += precision
                    valid_recall += recall
                    all_preds = np.concatenate((all_preds, preds)) if all_preds.size else preds
                    all_targets = np.concatenate((all_targets, target.cpu().numpy()))
                valid_f1 /= n
                valid_accuracy /= n
                valid_precision /= n
                valid_recall /= n
            valid_f1_history.append(valid_f1)
            valid_accuracy_history.append(valid_accuracy)
            valid_precision_history.append(valid_precision)
            valid_recall_history.append(valid_recall)
            print(f'Epoch [{epoch+1}/{num_epochs}]: Train Loss: {average_train_loss:.4f}, Val F1: {valid_f1:.2f}%, Accuracy: {valid_accuracy:.2f}%, Precision: {valid_precision:.2f}%, Recall: {valid_recall:.2f}%')
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                counter = 0
                best_epoch = epoch
                torch.save(model.state_dict(), f'single/best_model_{fold}.ckpt')
                torch.save(scaled_model.state_dict(), f'single/best_scaled_model_{fold}.ckpt')
            else:
                counter += 1
                if counter >= 15:
                    print(f'Early stopping at epoch {epoch+1}, best epoch {best_epoch+1} with Val F1 {best_valid_f1:.2f}%')
                    cm = confusion_matrix(all_targets, all_preds)
                    plot_confusion_matrix(cm, [str(i) for i in range(num_classes)], fold, epoch, is_sum=True)
                    break
            
            if epoch+1 >=300:
                cm = confusion_matrix(all_targets, all_preds)
                plot_confusion_matrix(cm, [str(i) for i in range(num_classes)], fold, epoch)
        
        # 绘制学习曲线图（可选）
        plt.figure(figsize=(12,10))
        plt.subplot(2,1,1)
        plt.plot(range(1,len(valid_f1_history)+1), valid_f1_history, label='Val F1', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score (%)')
        plt.title(f'Validation F1 Curve (Fold {fold+1})')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(range(1,len(valid_accuracy_history)+1), valid_accuracy_history, label='Val Accuracy', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Validation Accuracy Curve (Fold {fold+1})')
        plt.legend()
        plt.tight_layout()
        if not os.path.exists('figure'):
            os.makedirs('figure')
        plt.savefig(f'figure/learning_curve_fold{fold+1}.png')
        plt.close()
        
        # 保存当前fold模型
        torch.save(model.state_dict(), f'single/model_{fold}.ckpt')
        
        # ---------- XGBoost部分（多分类） ----------
        print("提取DNN的特征用于XGBoost训练...")
        def extract_features(data_loader):
            feats = []
            labs = []
            model.eval()
            with torch.no_grad():
                for data, target in data_loader:
                    data = data.float().to(device)
                    # 直接使用模型输出作为特征，也可取某中间层特征
                    output = model(data)
                    feats.append(output.cpu().numpy())
                    labs.append(target.cpu().numpy())
            return np.vstack(feats), np.hstack(labs)
        
        train_features, train_labels = extract_features(Train_data_loader)
        val_features, val_labels = extract_features(Validation_data_loader)
        test_features, test_labels = extract_features(Test_data_loader)
        
        print("训练XGBoost多分类模型...")
        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dval = xgb.DMatrix(val_features, label=val_labels)
        dtest = xgb.DMatrix(test_features)
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'merror',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 3407,
            'num_class': num_classes
        }
        bst = xgb.train(params, dtrain, num_boost_round=100,
                        evals=[(dtrain, 'train'), (dval, 'eval')],
                        early_stopping_rounds=20,
                        verbose_eval=False)
        xgb_probs = bst.predict(dtest)
        XGB_F1, XGB_accuracy, XGB_precision, XGB_recall, XGB_specificity, xgb_preds = evaluate_xgboost(xgb_probs, test_labels)
        print(f'XGBoost Multi-class: F1: {XGB_F1:.2f}%, Accuracy: {XGB_accuracy:.2f}%, Precision: {XGB_precision:.2f}%, Recall: {XGB_recall:.2f}%, Specificity: {XGB_specificity:.2f}%')
        with open('XGBoost_Results.csv', 'a') as f:
            f.write(f'fold [{fold+1}/5], F1: {XGB_F1:.2f}%, Accuracy: {XGB_accuracy:.2f}%, Precision: {XGB_precision:.2f}%, Recall: {XGB_recall:.2f}%, Specificity: {XGB_specificity:.2f}%\n')



            # # 测试集
            # Test_F1, Test_accuracy, Test_precision, Test_recall, Test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt', Scaled_Model=None)
            # Scaled_F1, Scaled_accuracy, Scaled_precision, Scaled_recall, Scaled_specificity, Scaled_macro_auc, Scaled_weighted_auc, _, _, _ = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt', Scaled_Model = scaled_model)
            # # plot_roc_curves(all_labels, all_probs, columns_to_encode)
            # print(
            #     f'fold [{fold + 1}/5]: '
            #     f'Test F1: {Test_F1:.2f}% | '
            #     f'Accuracy: {Test_accuracy:.2f}% | '
            #     f'Precision: {Test_precision:.2f}% | '
            #     f'Recall: {Test_recall:.2f}% | '
            #     f'Specificity: {Test_specificity:.2f}% | '
            #     f'Macro AUC: {macro_auc:.3f} | '
            #     f'Weighted AUC: {weighted_auc:.3f}'
            # )

            # print(
            #     f'fold [{fold + 1}/5]: '
            #     f'Scaled F1: {Scaled_F1:.2f}% | '
            #     f'Scaled Accuracy: {Scaled_accuracy:.2f}% | '
            #     f'Scaled Precision: {Scaled_precision:.2f}% | '
            #     f'Scaled Recall: {Scaled_recall:.2f}% | '
            #     f'Scaled Specificity: {Scaled_specificity:.2f}% | '
            #     f'Scaled Macro AUC: {Scaled_macro_auc:.3f} | '
            #     f'Scaled Weighted AUC: {Scaled_weighted_auc:.3f}'
            # )
            # # 将print的内容保存到result.csv文件,每次运行新起一行
            # with open('Test.csv', 'a') as f:
            #     f.write(
            #         f'fold [{fold + 1}/5], '
            #         f'Test F1: {Test_F1:.2f}%, '
            #         f'Accuracy: {Test_accuracy:.2f}%, '
            #         f'Precision: {Test_precision:.2f}%, '
            #         f'Recall: {Test_recall:.2f}%, '
            #         f'Specificity: {Test_specificity:.2f}%, '
            #         f'Macro AUC: {macro_auc:.3f}, '
            #         f'Weighted AUC: {weighted_auc:.3f}\n'
            #     )

            # with open('Scaled.csv', 'a') as f:
            #     f.write(
            #         f'fold [{fold + 1}/5], '
            #         f'Scaled F1: {Scaled_F1:.2f}%, '
            #         f'Scaled Accuracy: {Scaled_accuracy:.2f}%, '
            #         f'Scaled Precision: {Scaled_precision:.2f}%, '
            #         f'Scaled Recall: {Scaled_recall:.2f}%, '
            #         f'Scaled Specificity: {Scaled_specificity:.2f}%, '
            #         f'Scaled Macro AUC: {Scaled_macro_auc:.3f}, '
            #         f'Scaled Weighted AUC: {Scaled_weighted_auc:.3f}\n'
            #     )

            # # 保存当前 x 和 fold 的指标
            # all_accuracies[x].append(Test_accuracy)
            # all_precisions[x].append(Test_precision)
            # all_recalls[x].append(Test_recall)
            # all_f1_scores[x].append(Test_F1)
            # all_macro_auc[x].append(macro_auc)
            # all_weighted_auc[x].append(weighted_auc)

#     #简单的协变量分析，shap图在后面
#     if len(selected_folds_input) == 5:
#         # 绘制准确率折线图，x是加入的协变量数量
#         plt.figure(figsize=(12, 8))
#         for x in range(num_x):
#             plt.plot(range(1, num_folds + 1), all_accuracies[x], label=f'Accuracy (x={x + 1})', marker='o')
#         plt.xlabel('Fold Number')
#         plt.ylabel('Accuracy (%)')
#         plt.title('Comparison of Accuracy for Different x Values Across Folds')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(range(1, num_folds + 1))
#         plt.tight_layout()
#         # plt.show()
#         # 保存为png图在figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/Comparison of Accuracy for Different x Values Across Folds.png')

#         # 绘制精确率折线图
#         plt.figure(figsize=(12, 8))
#         for x in range(num_x):
#             plt.plot(range(1, num_folds + 1), all_precisions[x], label=f'Precision (x={x + 1})', marker='s')
#         plt.xlabel('Fold Number')
#         plt.ylabel('Precision (%)')
#         plt.title('Comparison of Precision for Different x Values Across Folds')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(range(1, num_folds + 1))
#         plt.tight_layout()
#         # plt.show()
#         # 保存为png图在figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/Comparison of Precision for Different x Values Across Folds.png')

#         # 绘制召回率折线图
#         plt.figure(figsize=(12, 8))
#         for x in range(num_x):
#             plt.plot(range(1, num_folds + 1), all_recalls[x], label=f'Recall (x={x + 1})', marker='^')
#         plt.xlabel('Fold Number')
#         plt.ylabel('Recall (%)')
#         plt.title('Comparison of Recall for Different x Values Across Folds')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(range(1, num_folds + 1))
#         plt.tight_layout()
#         # plt.show()
#         # 保存为png图在figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/Comparison of Recall for Different x Values Across Folds.png')

#         # 绘制 F1 分数折线图
#         plt.figure(figsize=(12, 8))
#         for x in range(num_x):
#             plt.plot(range(1, num_folds + 1), all_f1_scores[x], label=f'F1 Score (x={x + 1})', marker='D')
#         plt.xlabel('Fold Number')
#         plt.ylabel('F1 Score (%)')
#         plt.title('Comparison of F1 Score for Different x Values Across Folds')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(range(1, num_folds + 1))
#         plt.tight_layout()
#         # plt.show()
#         # 保存为png图在figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/Comparison of F1 Score for Different x Values Across Folds.png')

#     # 特征重要性排序，使用SHAP Value说明特征的重要程度，然后基于特征置换的特征重要性评估方法，也可以看作是一种特征消融实验方法，说明特征的重要性。

#     # origin = pd.DataFrame(features_val, columns=all_feature_names)
#     pca_feature_names = [f'PC{i}' for i in range(1, features_val.shape[1] + 1)]
#     origin = pd.DataFrame(features_val, columns=pca_feature_names)  

#     # # 修改前：
#     # COLUMNS = origin.columns

#     # 修改后：
#     COLUMNS = origin.columns  # 使用PCA后生成的特征名称
#     num_classes = labels_val.shape[1]  # 获取类别数

#     # 使用 shap.sample 对背景数据进行降采样
#     K = 32  # 可以根据实际情况调整 K 的值

#     # # 修改前：
#     # background_data = shap.kmeans(Shap_features, K)

#     # 修改后：
#     # 使用相同的方差选择器处理SHAP背景数据
#     Shap_features_filtered = pca_selector.transform(Shap_features)
#     background_data = shap.kmeans(Shap_features_filtered, K)

#     # 为全局SHAP分析选择样本
#     n_samples_for_global_shap = min(500, len(Shap_features_filtered))  # 限制样本数量以提高计算效率
#     global_shap_indices = np.random.choice(len(Shap_features_filtered), n_samples_for_global_shap, replace=False)
#     global_shap_samples = Shap_features_filtered[global_shap_indices]

#     selected_indices = np.random.choice(len(Shap_features), 3501, replace=True)
#     selected_features_val = Shap_features[selected_indices]

#     ALL_shap_exp = {}
#     ALL_top_features = {}

#     # 创建一个适用于DNN模型的SHAP包装类
#     class Trained_DNN(nn.Module):
#         def __init__(self, model, device):
#             super(Trained_DNN, self).__init__()
#             self.model = model
#             self.device = device
            
#         def forward(self, x):
#             if isinstance(x, np.ndarray):
#                 x = torch.tensor(x, dtype=torch.float32).to(self.device)
#             return torch.sigmoid(self.model(x)).cpu().detach().numpy()
    
#     # 使用新的包装类
#     Shap_model = Trained_DNN(model, device).eval()
    
#     # 执行全局SHAP分析
#     print("开始执行全局SHAP分析...")
#     global_shap_values = global_shap_analysis(
#         model=Shap_model,
#         background_data=background_data,
#         test_data=global_shap_samples,
#         feature_names=pca_feature_names,
#         class_names=columns_to_encode,
#         output_dir='figure/global_shap'
#     )
#     print("全局SHAP分析完成！")

#     # 将PCA特征的SHAP值映射回原始特征空间
#     print("开始将SHAP值映射回原始特征空间...")
#     original_feature_importance = map_pca_shap_to_original_features(
#         shap_values=global_shap_values,
#         pca_model=pca,  # 使用之前训练的PCA模型
#         feature_names=all_feature_names,  # 原始特征名称
#         class_names=columns_to_encode,
#         output_dir='figure/original_feature_shap'
#     )
#     print("SHAP值映射回原始特征空间完成！")

#     for class_idx in range(num_classes):

#         # 使用 SHAP 评估特征重要性
#         print(f"正在使用 SHAP 评估类别 {columns_to_encode[class_idx]} 的特征重要性...")

#         explainer = shap.KernelExplainer(model_wrapper, background_data)
#         shap_values = explainer.shap_values(selected_features_val, nsamples=256, main_effects=False, interaction_index=None)

#         base_values = explainer.expected_value

#         if isinstance(base_values, (int, float)):
#             base_values = np.full(len(selected_features_val), base_values)

#         shap_exp = shap.Explanation(shap_values, data=selected_features_val, feature_names=all_feature_names, base_values=base_values)

#         #保存shap_exp
#         ALL_shap_exp[columns_to_encode[class_idx]] = shap_exp

#         # 绘制 SHAP 汇总图
#         shap.plots.bar(shap_exp, max_display=16)
#         # 保存到figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/SHAP Summary Plot for category{columns_to_encode[class_idx]}.png')

#         # 绘制 SHAP 摘要图
#         shap.plots.beeswarm(shap_exp, max_display=16)
#         # 保存到figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/SHAP Beeswarm Plot for category{columns_to_encode[class_idx]}.png')

#         # 特征消融实验
#         print(f"正在计算类别 {columns_to_encode[class_idx]} 的特征重要性...")
#         out = pd.DataFrame({'tag': predict(features_val, probs_Switcher)[:, class_idx]})
#         importance_dict = {}  # 用于存储每个特征的重要性计算结果

#         for key in COLUMNS:
#             copy = origin.copy()
#             copy[key] = copy[key].sample(frac=1, random_state=1).reset_index()[key]
#             cp_out = predict(copy.values, probs_Switcher)[:, class_idx]
#             # 将 out['tag'] 转换为 numpy.ndarray 后再进行减法运算
#             diff = (out['tag'].values - cp_out).flatten()
#             importance_dict[key] = diff ** 2
#             print('key = ', key, ' affect = ', importance_dict[key].sum() ** 0.5)

#         # 一次性将所有列合并到 out 中
#         importance_df = pd.DataFrame(importance_dict)
#         out = pd.concat([out, importance_df], axis=1)

#         importance_result = (pd.DataFrame(out.sum(axis=0)) ** 0.5).sort_values(by=0, ascending=False)
#         print(f"类别 {class_idx} 的特征重要性排序结果：")
#         print(importance_result)

#         # 绘制柱状图
#         plt.figure(figsize=(15, 6))
#         top_features = importance_result.iloc[1:].head(64)  # 取前 64 个特征
#         plt.bar(top_features.index, top_features[0])
#         plt.xlabel('Features')
#         plt.ylabel('Importance of Features')
#         plt.title(f'Bar chart of the top 64 feature importances for category{columns_to_encode[class_idx]}')
#         plt.xticks(rotation=45, fontsize=6)
#         plt.tight_layout()
#         #plt.show()
#         # 保存为png图在figure文件夹中，命名为这个图的title
#         plt.savefig(f'figure/Bar chart of the top 64 feature importances for category{columns_to_encode[class_idx]}.png')



#         # 选取 SHAP 高重要性特征（这里以绝对值的均值排序取前 209 个为例）
#         shap_importance = np.abs(shap_values).mean(axis=0)
#         sorted_indices = np.argsort(shap_importance)[::-1]
#         top_209_features = np.array(all_feature_names)[sorted_indices[:209]]
#         ALL_top_features[columns_to_encode[class_idx]] = top_209_features

#     #分层分析，对患病程度，性别，种族进行分层分析。目前用多因素线性回归模型，说明不同层次人群特征的线性相关程度，但部分特征无法推导线性相关程度。
#     # 使用SHAP图可以补充说明非线性相关程度特征如何影响判断结果（非线性分析？），目前有瀑布图（一类人群的shap value对模型结果影响的具象化），force图（个例shap value对结果的影响）
#     # 参考图片都在文件夹里

#     stratify_variable = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']#分别是患病程度，性别，种族

#     for i in stratify_variable:
#         # 获取分层变量的不同取值
#         strata = Multilay_origin[i].unique()

#         for class_idx in range(num_classes):
#             top_209_features = ALL_top_features[columns_to_encode[class_idx]]

#             for stratum in strata:
#                 # 筛选出当前层的数据
#                 stratum_mask = Multilay_origin[i] == stratum
#                 stratum_indices = np.where(stratum_mask)[0]
#                 # 确保只选择与 selected_features_val 相同索引的数据
#                 common_indices = np.intersect1d(stratum_indices, selected_indices)
#                 stratum_data = Multilay_origin[stratum_mask].copy()
#                 labels_stratum = labels_val[stratum_indices]

#                 # 构建用于线性回归的数据集
#                 regression_data = stratum_data[top_209_features].copy()

#                 regression_data['disease_label'] = labels_stratum[:, class_idx].astype(int)

#                 # 生成与 shap_values.data 长度一致的布尔数组
#                 bool_tf = np.isin(selected_indices, common_indices)

#                 # 这里算一类的平均shap value,用于分层分析的
#                 try:
#                     # 获取当前类别对应的 shap.Explanation 对象
#                     shap_exp = ALL_shap_exp[columns_to_encode[class_idx]]

#                     # 筛选 shap.Explanation 对象
#                     filtered_shap_values = shap_exp.values[bool_tf]
#                     filtered_data = shap_exp.data[bool_tf]
#                     filtered_base_values = shap_exp.base_values[bool_tf]

#                     # 创建筛选后的 shap.Explanation 对象
#                     filtered_shap_exp = shap.Explanation(
#                         values=filtered_shap_values,
#                         data=filtered_data,
#                         feature_names=shap_exp.feature_names,
#                         base_values=filtered_base_values
#                     )

#                     new_shap = new_shap_values(ALL_shap_exp[columns_to_encode[class_idx]], bool_tf=bool_tf, method='mean')
#                     # 确保传递给 shap.plots.waterfall 的是 shap.Explanation 对象
#                     shap.plots.waterfall(new_shap.get_explanation())

#                     shap.plots.beeswarm(filtered_shap_exp, max_display=64)
#                 except Exception as e:
#                     print(f"绘制 SHAP 瀑布图时出错，类别 {columns_to_encode[class_idx]}，{i} 为 {stratum} 层，错误信息：{e}")

#                 # 构建线性回归模型公式
#                 if np.var(regression_data['disease_label']) == 0:
#                     print(f"因变量 'disease_label' 没有变异性，跳过类别 {columns_to_encode[class_idx]} 在 {i} 为 {stratum} 层的线性回归分析。")
#                 else:
#                     formula = 'disease_label ~ ' + ' + '.join(top_209_features)
#                     try:
#                         Ols_model = smf.ols(formula, data=regression_data).fit(cov_type='HC3')

#                         # 输出回归结果
#                         print(f"\n类别 {columns_to_encode[class_idx]} 在 {i} 为 {stratum} 层的多因素线性回归分析结果：")
#                         print(Ols_model.summary())

#                         # 输出相关程度描述
#                         print(f"\n在类别 {columns_to_encode[class_idx]} 中，当 {i} 为 {stratum} 时，协变量调整模型显示：")
#                         for feature in top_209_features:
#                             coef = Ols_model.params[feature]
#                             p_value = Ols_model.pvalues[feature]
#                             conf_int = Ols_model.conf_int().loc[feature]
#                             if p_value < 0.05:
#                                 if coef > 0:
#                                     print(
#                                         f"{feature} 与疾病标签呈正相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率增加 {coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。")
#                                 else:
#                                     print(
#                                         f"{feature} 与疾病标签呈负相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率降低 {-coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。")
#                             else:
#                                 print(f"{feature} 与疾病标签之间未发现显著的线性关系 (p = {p_value:.4f})。")
#                     except Exception as e:
#                         print(f"在类别 {columns_to_encode[class_idx]}，{i} 为 {stratum} 层时，线性回归模型拟合失败，错误信息：{e}")


#                 # # 检查多重共线性
#                 # X = regression_data[top_209_features]
#                 # vif = pd.DataFrame()
#                 # for j in range(X.shape[1]):
#                 #     try:
#                 #         vif.loc[j, "VIF Factor"] = variance_inflation_factor(X.values, j)
#                 #     except ZeroDivisionError:
#                 #         print(f"自变量 {X.columns[j]} 存在完全多重共线性，VIF 无法计算。考虑删除该自变量。")
#                 #         vif.loc[j, "VIF Factor"] = np.nan
#                 # vif["features"] = X.columns
#                 # print(vif)

# # 在这里添加 analyze_shap_performance_correlation 函数
#     def analyze_shap_performance_correlation(global_shap_values, test_metrics, class_names, output_dir='figure/shap_performance'):
#         """
#         分析SHAP值与模型性能指标的关系
        
#         参数:
#         global_shap_values -- 全局SHAP值列表
#         test_metrics -- 测试指标字典，包含'f1', 'accuracy', 'precision', 'recall'等
#         class_names -- 类别名称列表
#         output_dir -- 输出目录
#         """
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         # 计算每个类别的平均绝对SHAP值
#         mean_abs_shap = []
#         for i, shap_values in enumerate(global_shap_values):
#             mean_abs_shap.append(np.mean(np.abs(shap_values)))
            
#         # 创建性能与SHAP值的对比图
#         metrics = ['f1', 'accuracy', 'precision', 'recall']
#         for metric in metrics:
#             if metric in test_metrics:
#                 plt.figure(figsize=(10, 6))
#                 plt.scatter(mean_abs_shap, test_metrics[metric], c=range(len(class_names)), cmap='viridis')
                
#                 # 添加类别标签
#                 for i, class_name in enumerate(class_names):
#                     plt.annotate(class_name, 
#                                 (mean_abs_shap[i], test_metrics[metric][i]),
#                                 textcoords="offset points", 
#                                 xytext=(0,10), 
#                                 ha='center')
                
#                 # 添加趋势线
#                 z = np.polyfit(mean_abs_shap, test_metrics[metric], 1)
#                 p = np.poly1d(z)
#                 plt.plot(mean_abs_shap, p(mean_abs_shap), "r--", alpha=0.8)
                
#                 plt.xlabel('平均绝对SHAP值')
#                 plt.ylabel(f'{metric.capitalize()} (%)')
#                 plt.title(f'SHAP值与{metric.capitalize()}的关系')
#                 plt.colorbar(label='类别索引')
#                 plt.grid(True, linestyle='--', alpha=0.7)
#                 plt.tight_layout()
#                 plt.savefig(f'{output_dir}/shap_vs_{metric}.png')
#                 plt.close()
    
#     # 如果已经计算了全局SHAP值，分析其与模型性能的关系
#     if 'global_shap_values' in locals():
#         # 收集测试指标
#         test_metrics = {
#             'f1': [all_f1_scores[0][i] for i in range(len(columns_to_encode))],
#             'accuracy': [all_accuracies[0][i] for i in range(len(columns_to_encode))],
#             'precision': [all_precisions[0][i] for i in range(len(columns_to_encode))],
#             'recall': [all_recalls[0][i] for i in range(len(columns_to_encode))]
#         }
        
#         # 分析SHAP值与性能的关系
#         analyze_shap_performance_correlation(
#             global_shap_values=global_shap_values,
#             test_metrics=test_metrics,
#             class_names=columns_to_encode,
#             output_dir='figure/shap_performance'
#         )
