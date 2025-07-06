import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
from Model import Model_VGG
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
import xgboost as xgb  # 新增XGBoost导入

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

# 添加绘制混淆矩阵的函数
def plot_confusion_matrix(cm, class_names, fold, epoch, is_sum=False):
    """
    绘制归一化的混淆矩阵图片
    
    参数:
    cm -- 混淆矩阵，形状为 (n_classes, 2, 2)
    class_names -- 类别名称列表
    fold -- 当前fold编号
    epoch -- 当前epoch编号
    is_sum -- 是否为累积混淆矩阵
    """
    n_classes = len(class_names)
    
    # 从原始预测和标签中提取数据
    if is_sum:
        # 使用all_targets和all_preds
        y_true = all_targets
        y_pred = all_preds
    else:
        # 使用当前batch的target和preds
        y_true = target.cpu().numpy()
        y_pred = preds
    
    # 将多标签转换为多类别编码
    # 例如[0,1,0,0,1]转换为二进制"01001"，再转为十进制9
    y_true_classes = np.zeros(len(y_true), dtype=int)
    y_pred_classes = np.zeros(len(y_pred), dtype=int)
    
    for i in range(len(y_true)):
        # 检查是否为"无类别"情况
        if np.sum(y_true[i]) == 0:
            y_true_classes[i] = 2**n_classes  # 使用一个额外的编码表示"无类别"
        else:
            true_str = ''.join(map(str, y_true[i].astype(int)))
            y_true_classes[i] = int(true_str, 2)
        
        if np.sum(y_pred[i]) == 0:
            y_pred_classes[i] = 2**n_classes  # 使用一个额外的编码表示"无类别"
        else:
            pred_str = ''.join(map(str, y_pred[i].astype(int)))
            y_pred_classes[i] = int(pred_str, 2)
    
    # 找出实际出现的类别
    unique_classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
    n_unique = len(unique_classes)
    
    # 创建混淆矩阵
    conf_matrix = np.zeros((n_unique, n_unique))
    for i in range(len(y_true_classes)):
        true_idx = np.where(unique_classes == y_true_classes[i])[0][0]
        pred_idx = np.where(unique_classes == y_pred_classes[i])[0][0]
        conf_matrix[true_idx, pred_idx] += 1
    
    # 归一化混淆矩阵，处理除零问题
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # 避免除以零，将零值替换为1
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm_conf_matrix = conf_matrix / row_sums
    
    # 确保figure目录存在
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    
    # 准备标签，为"无类别"情况添加特殊标签
    xticklabels = []
    yticklabels = []
    for c in unique_classes:
        if c == 2**n_classes:
            xticklabels.append("None")
            yticklabels.append("None")
        else:
            xticklabels.append(bin(c)[2:].zfill(n_classes))
            yticklabels.append(bin(c)[2:].zfill(n_classes))
    
    # 使用seaborn绘制热图
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=xticklabels,
               yticklabels=yticklabels)
    
    plt.title(f'Normalized Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 添加类别名称标注
    plt.figtext(0.5, 0.01, f'Classes: {", ".join(class_names)} + None', ha='center')
    
    plt.tight_layout()
    
    # 保存图片
    if is_sum:
        plt.savefig(f'figure/confusion_matrix_fold{fold+1}_sum.png')
    else:
        plt.savefig(f'figure/confusion_matrix_fold{fold+1}_epoch{epoch+1}.png')
    
    plt.close()
# 测试函数
def test(Test_data_loader, probs_Switcher, Saved_Model, Scaled_Model):
    model.load_state_dict(torch.load(Saved_Model, map_location=device, weights_only=False))
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
            if Scaled_Model != None:
                output = Scaled_Model(data)

            # 收集概率和标签
            probs_sigmoid = torch.sigmoid(output)
            all_probs.append(probs_sigmoid.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(output, target, probs_Switcher)

            Test_F1 += test_f1
            Test_accuracy += test_accuracy
            Test_precision += test_precision
            Test_recall += test_recall
            Test_specificity += test_specificity

        # 计算AUC-ROC
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算每个类别的AUC
        auc_scores = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_scores.append(auc)
            except ValueError:
                print(f"Class {i} has only one class present in test set.")
                auc_scores.append(float('nan'))

        # 计算宏平均和加权平均AUC
        macro_auc = np.nanmean(auc_scores)
        weighted_auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')

        Test_F1 /= n
        Test_accuracy /= n
        Test_precision /= n
        Test_recall /= n
        Test_specificity /= n

        return Test_F1, Test_accuracy, Test_precision, Test_recall, Test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels

# 阈值计算函数
def Probs_Switcher(output, labels):
    probs_Switcher = np.array([])
    probs_sigmoid = torch.sigmoid(output)
    probs = probs_sigmoid.cpu().detach().numpy()
    probs_tensor, labels_tensor = torch.from_numpy(probs), labels
    Split_probs = torch.unbind(probs_tensor, dim=1)
    Split_labels = torch.unbind(labels_tensor, dim=1)
    # print(labels.size(1))
    for i in range(labels.size(1)):
        split_labels_np = Split_labels[i].cpu().numpy()
        split_probs_np = Split_probs[i].cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(split_labels_np, split_probs_np)
        precision = precision * 0.6
        recall = recall
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        index = argmax(f1_scores)
        probs_Switcher = np.append(probs_Switcher, thresholds[index])

    return probs_Switcher

# 指标计算函数
def f1_score_func(logits, labels, probs_Switcher):
    probs_sigmoid = torch.sigmoid(logits)
    probs = probs_sigmoid.detach().cpu().numpy()
    labels = labels.cpu().detach().numpy()

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

# 自定义预测函数
def predict(data, probs_Switcher):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(data)
        probs_sigmoid = torch.sigmoid(output)
        probs = probs_sigmoid.cpu().detach().numpy()
        preds = np.float64(probs >= probs_Switcher)
        return preds

# SHAP分析
def Shap_predict(X):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X)
        probs_sigmoid = torch.sigmoid(output)
        probs = probs_sigmoid.cpu().detach()
    return probs[:, class_idx]

def model_wrapper(X):   # 这个函数用于包装模型，以便SHAP库可以调用
    output = Shap_model.forward(X)
    if isinstance(output, torch.Tensor):
        output = output.cpu().detach().numpy()
    return output[:, class_idx]

def global_shap_analysis(model, background_data, test_data, feature_names, class_names, output_dir='figure/global_shap'):
    """
    执行全局SHAP分析并生成可视化图表
    
    参数:
    model -- 训练好的模型
    background_data -- 用于SHAP解释器的背景数据
    test_data -- 用于生成SHAP值的测试数据
    feature_names -- 特征名称列表
    class_names -- 类别名称列表
    output_dir -- 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取模型所在设备
    device = next(model.parameters()).device

    # 检查特征名称长度
    if len(feature_names) != test_data.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]
        print(f"警告：自动生成特征名称，长度为 {len(feature_names)}")
    
    # 创建解释器，确保数据和模型在同一设备上
    explainer = shap.KernelExplainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().detach().numpy(),
        background_data
    )

    # 计算SHAP值
    shap_values = explainer.shap_values(test_data, nsamples=500)
    # 打印形状和类型信息
    print("shap_values 类型:", type(shap_values))
    
    # 检查shap_values是列表还是数组
    if isinstance(shap_values, list):
        print("shap_values 是列表，长度:", len(shap_values))
        if len(shap_values) > 0:
            print("第一个元素形状:", np.array(shap_values[0]).shape)
    else:
        print("shap_values 形状:", shap_values.shape)
    
    # 存储所有类别的SHAP值
    all_shap_values = []
    
    # 为每个类别处理SHAP值
    for i, class_name in enumerate(class_names):
        print(f"计算类别 {class_name} 的全局SHAP值...")
        
        # 根据shap_values的类型获取对应类别的值
        if isinstance(shap_values, list):
            # 如果是列表，每个元素对应一个类别
            if i < len(shap_values):
                class_shap_values = shap_values[i]
            else:
                print(f"警告：类别索引 {i} 超出shap_values列表长度 {len(shap_values)}")
                continue
        else:
            # 如果是数组，假设形状为 (samples, features, classes)
            class_shap_values = shap_values[:, :, i]
            
        print(f"类别 {class_name} 的SHAP值形状:", np.array(class_shap_values).shape)
        all_shap_values.append(class_shap_values)
        
        # 创建SHAP解释对象
        shap_exp = shap.Explanation(
            values=class_shap_values,
            data=test_data,
            feature_names=feature_names
        )

        # 绘制条形图 - 使用英文命名
        plt.figure(figsize=(10, 8))
        try:
            # 计算特征重要性（平均绝对SHAP值）
            feature_importance = np.abs(np.array(class_shap_values)).mean(0)
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
            try:
                shap.summary_plot(class_shap_values, test_data, feature_names=feature_names,
                                  plot_type="bar", show=False, max_display=20)
            except Exception as e2:
                print(f"使用shap.summary_plot绘制条形图也失败: {e2}")
        
        plt.tight_layout()
        # 使用英文命名保存文件
        plt.savefig(f'{output_dir}/global_bar_plot_category_{class_name}.png')
        plt.close()
        
        # 绘制摘要图 - 使用英文命名
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(class_shap_values, test_data, feature_names=feature_names, show=False, max_display=20)
            plt.title(f'SHAP Summary Plot for Category {class_name}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/global_summary_plot_category_{class_name}.png')
        except Exception as e:
            print(f"绘制摘要图时出错: {e}")
        plt.close()

    # 绘制多项式图 - Polynomial-SHAP plot of the data.
    # 整合 SHAP 值为三维数组，处理列表和数组两种情况
    if isinstance(shap_values, list):
        # 确保所有元素都是数组并且形状一致
        all_arrays = [np.array(sv) for sv in all_shap_values]
        if all(arr.shape == all_arrays[0].shape for arr in all_arrays):
            shap_3d = np.stack(all_arrays, axis=2)
        else:
            print("警告：不同类别的SHAP值形状不一致，跳过多项式图绘制")
            return all_shap_values
    else:
        shap_3d = shap_values
    
    # 计算每个特征在各分类下的平均绝对 SHAP 值
    mean_abs_shap = np.abs(shap_3d).mean(axis=0)  # 形状：(特征数, 类别数)
    agg_shap_df = pd.DataFrame(mean_abs_shap, columns=class_names, index=feature_names)

    # 按特征重要性总和排序（模拟示例图特征顺序）
    feature_order = agg_shap_df.sum(axis=1).sort_values(ascending=False).index
    agg_shap_df = agg_shap_df.loc[feature_order]

    plt.figure(figsize=(18, 8))
    bottom = np.zeros(len(agg_shap_df))
    colors = sns.color_palette("tab10", len(class_names))  # 生成类别对应颜色

    for i, disease in enumerate(class_names):
        plt.bar(
            agg_shap_df.index,
            agg_shap_df[disease],
            bottom=bottom,
            label=disease,
            color=colors[i],
            edgecolor="black",  # 显示条形边界
            linewidth=0.5
        )
        bottom += agg_shap_df[disease]

    plt.xlabel("Top Most Important Features in Predicting Liver Disease", fontsize=12)
    plt.ylabel("mean(|SHAP value|) / average impact on model output magnitude", fontsize=12)
    plt.title("Polynomial-SHAP plot of the data", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)  # 旋转并右对齐特征标签
    plt.legend(
        title="",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
        frameon=False  # 不显示图例边框
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/polynomial_shap_plot.png")
    plt.show()

    return all_shap_values

# 添加全局SHAP解释函数
def map_pca_shap_to_original_features(shap_values, pca_model, feature_names, class_names, output_dir='figure/original_feature_shap'):
    """
    将PCA特征的SHAP值映射回原始特征空间
    
    参数:
    shap_values -- PCA特征的SHAP值列表
    pca_model -- 训练好的PCA模型
    feature_names -- 原始特征名称列表
    class_names -- 类别名称列表
    output_dir -- 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取PCA组件
    components = pca_model.components_
    
    # 对每个类别进行处理
    for i, class_name in enumerate(class_names):
        print(f"将类别 {class_name} 的SHAP值映射回原始特征空间...")
        
        # 获取当前类别的SHAP值
        if isinstance(shap_values, list):
            class_shap_values = shap_values[i]
        else:
            class_shap_values = shap_values
        
        # 计算原始特征的重要性
        # SHAP值与PCA组件的点积
        original_importance = np.zeros(len(feature_names))
        
        # 对每个样本的SHAP值
        for sample_idx in range(class_shap_values.shape[0]):
            # 对每个PCA特征
            for pca_idx in range(class_shap_values.shape[1]):
                # 将SHAP值分配给原始特征
                for feat_idx in range(len(feature_names)):
                    # 权重是PCA组件中原始特征的贡献
                    weight = abs(components[pca_idx, feat_idx])
                    # 将SHAP值按权重分配给原始特征
                    original_importance[feat_idx] += abs(class_shap_values[sample_idx, pca_idx]) * weight
        
        # 归一化重要性分数
        original_importance = original_importance / original_importance.sum() * 100
        
        # 创建原始特征重要性的DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': original_importance
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 保存到CSV
        importance_df.to_csv(f'{output_dir}/original_feature_importance_{class_name}.csv', index=False)
        
        # 绘制前20个最重要的原始特征
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['Importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Relative Importance (%)')
        plt.title(f'Top 20 Original Features Importance for {class_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/original_feature_importance_{class_name}.png')
        plt.close()
        
        print(f"类别 {class_name} 的前10个最重要原始特征:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.2f}%")
    
    return importance_df

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
    #plt.show()
    # 保存为png图在figure文件夹中，命名为这个图的title
    plt.savefig(f'figure/Receiver Operating Characteristic (ROC) Curves.png')

if __name__ == '__main__':  
    # columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    # columns_to_encode = ['MCQ160C', 'MCQ160F']
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    # columns_to_encode = ['MCQ160B', 'MCQ160D']

    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

    # 特征数量选择
    # num_columns_to_select = 202
    # selected_column_indices = np.random.choice(features.shape[1], size=num_columns_to_select, replace=False)
    # features = features[:, selected_column_indices]
    # selected_feature_names = all_feature_names[selected_column_indices]

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

    mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    mean_val, std_val = np.mean(features_val, axis=0), np.std(features_val, axis=0)
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1e-8  # 将标准差为零的值设置为一个很小的数，避免除以零

    # # 修改前：
    # features = (features - mean) / std
    # features_val = (features_val - mean_val) / (std_val + 1e-8)
    # features = features.reshape(features.shape[0], -1)

    # 修改后：
    from sklearn.feature_selection import VarianceThreshold

    # # 添加方差过滤（阈值设为0.01）
    # selector = VarianceThreshold(threshold=0.01)
    # features = selector.fit_transform(features)
    # features_val = selector.transform(features_val)

    # # 更新特征名称
    # mask = selector.get_support()
    # all_feature_names = all_feature_names[mask]

    # 重新计算标准化所需的均值、标准差（基于筛选后的 features）
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
    # # 修改前：
    # features = pca.fit_transform(features)
    # features_val = pca.transform(features_val)

    # 修改后：
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
    # pause = input("Press Enter to continue...")

    # 重新计算PCA后数据的均值和标准差（例如，基于训练集features）
    mean_pca, std_pca = np.mean(features, axis=0), np.std(features, axis=0)

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

    # 让用户输入想要运行的fold编号，用逗号分隔
    # selected_folds_input = input("请输入想要运行的fold编号（用逗号分隔，例如：1,3,5）：")  #这里输入之后，变量的值是字符串，需要转换为列表
    # 为了调试方便，这里写死selected_folds_input为1,2,3,4,5
    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  # 跳过未选择的fold

        l1_weight = 0.070
        num_classes = len(columns_to_encode)  # Adjust the number of classes
        num_epochs = 350
        batch_size = 256
        input_length = batch_size
        for x in range(len(Covariates_features)):

            #model = Model_VGG.VGG11(num_classes=num_classes, in_channels=len(all_feature_names)-len(Covariates_features), Covariates_features_length=x)
            #model = Model_VGG.VGG11(num_classes=num_classes, in_channels=features.shape[1], Covariates_features_length=x)
            # 将 Covariates_features_length 固定为0，确保输入通道与PCA后的特征一致。
            # 1. 训练VGG模型提取特征
            print("第1步：训练VGG模型用于特征提取...")
            model = Model_VGG.VGG11(num_classes=num_classes, in_channels=features.shape[1], epoch=num_epochs)

            # 温度缩放模型
            scaled_model = ModelWithTemperature(model, len(columns_to_encode))

            pos_weight = torch.tensor(1.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            # optimizer = optim.Adam(model.parameters(), lr=0.00275)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.475) 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            # scaled_model.to(device)

            total_num, trainable_num = get_parameter_number(model)
            print(f'Fold {fold + 1}, Total params: {total_num}, Trainable params: {trainable_num}')

            Validation_size = len(features_val) // 5
            Test_size = Validation_size

            indices = np.arange(len(features_val))
            np.random.shuffle(indices)
            validation_index = indices[(fold + 1) * Validation_size: (fold + 2) * Validation_size] if fold < 4 else indices[4 * Validation_size:]
            test_indices = indices[fold * Test_size: (fold + 1) * Test_size]

            # print(validation_index)

            trainX = features[train_index]
            trainY = labels[train_index]
            valX = features_val[validation_index]
            valY = labels_val[validation_index]
            testX = features_val[test_indices]
            testY = labels_val[test_indices]

            Train_data = NetDataset(trainX, trainY)
            Validation_data = NetDataset(valX, valY)
            Test_data = NetDataset(testX, testY)

            Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
            Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)

            # 初始化存储变量
            all_preds = np.empty((0, len(columns_to_encode)))
            all_targets = np.empty((0, len(columns_to_encode)))

            valid_f1_scores = []
            valid_precision_scores = []

            # 添加用于绘制学习曲线的列表
            train_losses = []
            valid_f1_history = []
            valid_accuracy_history = []
            valid_precision_history = []
            valid_recall_history = []

            # 添加早停相关变量  
            best_valid_f1 = 0
            patience = 15  # 设置耐心值，连续多少个epoch没有提升则停止
            counter = 0
            best_epoch = 0

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
                    loss += l1_weight * l1_loss  # l1_weight是L1正则化的权重，可以根据需要调整

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)

                average_train_loss = train_loss / len(Train_data_loader.dataset)
                scheduler.step()

                # 记录训练损失
                train_losses.append(average_train_loss)

                model.eval()
                n = 0
                valid_f1 = 0
                valid_accuracy = 0
                valid_precision = 0
                valid_recall = 0

                Scaled_f1 = 0
                Scaled_accuracy = 0
                Scaled_precision = 0
                Scaled_recall = 0
                with torch.no_grad():
                    for data, target in Validation_data_loader:
                        n += 1
                        data, target = data.float().to(device), target.float().to(device)
                        output = model(data)
                        probs_Switcher = Probs_Switcher(output, target)
                        f1, accuracy, precision, recall, preds, _= f1_score_func(output, target, probs_Switcher)

                        #温度缩放
                        ##温度缩放需要在经过一定训练阶段后进行，这里我们使用早停的计数，当小于patience的一定倍数时，我们开始温度缩放
                        if counter >= patience/3 :
                            if n == 1:
                                scaled_model.set_temperature(Train_data_loader, probs_Switcher)
                                # print(f'Before scaling:\n:{output}')
                                output = scaled_model(data)
                                # print(f'After scaling:\n:{output}')
                                scaled_f1, scaled_accuracy, scaled_precision, scaled_recall, _, _ = f1_score_func(output, target,
                                                                                                            probs_Switcher)
                                Scaled_f1 += scaled_f1
                                Scaled_accuracy += scaled_accuracy
                                Scaled_precision += scaled_precision
                                Scaled_recall += scaled_recall

                        valid_f1 += f1
                        valid_accuracy += accuracy
                        valid_precision += precision
                        valid_recall += recall

                        # 保存预测值和真实值
                        all_preds = np.concatenate((all_preds, preds), axis=0)
                        all_targets = np.concatenate((all_targets, target.cpu().numpy()), axis=0)

                    valid_f1 /= n
                    valid_accuracy /= n
                    valid_precision /= n
                    valid_recall /= n

                    Scaled_f1 /= n
                    Scaled_accuracy /= n
                    Scaled_precision /= n
                    Scaled_recall /= n

                # 记录验证集指标
                valid_f1_history.append(valid_f1)
                valid_accuracy_history.append(valid_accuracy)
                valid_precision_history.append(valid_precision)
                valid_recall_history.append(valid_recall)

                valid_f1_scores.append(valid_f1)
                valid_precision_scores.append(valid_precision)
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}]: Average Train Loss: {average_train_loss:.4f} '
                    f'Validation F1: {valid_f1:.2f}%, Validation Accuracy: {valid_accuracy:.2f}%, Validation Precision: {valid_precision:.2f}%, Validation Recall: {valid_recall:.2f}%')
                if Scaled_f1 != 0:
                    print(
                    f'Epoch [{epoch + 1}/{num_epochs}]: Scaled F1: {Scaled_f1:.2f}%, '
                    f'Scaled Accuracy: {Scaled_accuracy:.2f}%, Scaled Precision: {Scaled_precision:.2f}%, Scaled Recall: {Scaled_recall:.2f}%')

                # 在第300次迭代后绘制混淆矩阵
                if epoch + 1 >= 300:
                    cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
                    cm = multilabel_confusion_matrix(target.cpu().numpy(), preds)
                    print(cm_sum)
                    print(cm)

                    # 添加绘制混淆矩阵图片的代码
                    plot_confusion_matrix(cm, columns_to_encode, fold, epoch)
                    plot_confusion_matrix(cm_sum, columns_to_encode, fold, epoch, is_sum=True)

                # 早停机制：检查验证集F1分数是否提高
                if valid_f1 > best_valid_f1:
                    best_valid_f1 = valid_f1
                    counter = 0
                    best_epoch = epoch
                    # 保存最佳模型
                    torch.save(model.state_dict(), f'single/best_model_{fold}_{x}.ckpt')
                    # 同时保存温度缩放模型
                    torch.save(scaled_model.state_dict(), f'single/best_scaled_model_{fold}_{x}.ckpt')
                else:
                    counter += 1
                    print(f'EarlyStopping counter: {counter} out of {patience}')
                    if counter >= patience:
                        print(f'Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with F1: {best_valid_f1:.2f}%')
                        
                        # 在早停时绘制sum版本的当前折的混淆矩阵
                        cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
                        print(f"早停时的混淆矩阵 (Fold {fold+1}):")
                        print(cm_sum)
                        # 绘制混淆矩阵图片
                        plot_confusion_matrix(cm_sum, columns_to_encode, fold, epoch, is_sum=True)
                        
                        break

            # 绘制学习曲线
            plt.figure(figsize=(12, 10))
            
            # 创建两个子图
            plt.subplot(2, 1, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training Loss Curve (Fold {fold+1}, x={x+1})')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, len(valid_f1_history) + 1), valid_f1_history, label='Validation F1', color='red')
            plt.plot(range(1, len(valid_accuracy_history) + 1), valid_accuracy_history, label='Validation Accuracy', color='blue')
            plt.plot(range(1, len(valid_precision_history) + 1), valid_precision_history, label='Validation Precision', color='green')
            plt.plot(range(1, len(valid_recall_history) + 1), valid_recall_history, label='Validation Recall', color='purple')
            plt.axvline(x=best_epoch+1, color='black', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
            plt.xlabel('Epochs')
            plt.ylabel('Score (%)')
            plt.title(f'Validation Metrics Curve (Fold {fold+1}, x={x+1})')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
            # 确保figure目录存在
            if not os.path.exists('figure'):
                os.makedirs('figure')
                
            plt.savefig(f'figure/learning_curve_fold{fold+1}_x{x+1}.png')
            plt.close()       
            # 保存模型
            # torch.save(model.state_dict(), f'single/best_model_{fold}_{x}.ckpt')
            torch.save(model.state_dict(), f'single/model_{fold}_{x}.ckpt')  

            # 2. 提取特征
            print("第2步：使用VGG模型提取特征...")
            def extract_features(data_loader):
                features = []
                labels = []
                
                model.eval()
                with torch.no_grad():
                    for data, target in data_loader:
                        data = data.float().to(device)
                        # 直接使用模型的前向传播获取输出
                        output = model(data)
                        # 获取模型的最后一个全连接层之前的特征
                        batch_features = model.get_features(data)  # 假设模型有get_features方法
                        features.append(batch_features.cpu().numpy())
                        labels.append(target.numpy())
                return np.vstack(features), np.vstack(labels)
            
            # 在Model_VGG.py中添加get_features方法
            # 如果没有get_features方法，可以临时添加一个简单的特征提取函数
            def get_features(model, data):
                # 使用模型的前向传播，但只到倒数第二层
                # 这里假设模型的最后一层是分类器
                x = data
                # 对于2D输入，需要将其转换为4D
                if len(x.shape) == 2:
                    # 将[batch_size, features]转换为[batch_size, channels, height, width]
                    # 这里我们将特征视为单通道，并设置高度为1
                    x = x.unsqueeze(1).unsqueeze(-1)  # 变为[batch_size, 1, features, 1]
                
                # 获取特征
                features = model.features(x)  # 假设model.features是特征提取部分
                features = features.view(features.size(0), -1)  # 展平特征
                return features
            
            # 使用临时函数替代
            train_features, train_labels = [], []
            val_features, val_labels = [], []
            test_features, test_labels = [], []
            
            model.eval()
            with torch.no_grad():
                # 处理训练数据
                for data, target in Train_data_loader:
                    data = data.float().to(device)
                    # 我们需要将其转换为[batch_size, 93, 1, 1]
                    data_4d = data.unsqueeze(-1).unsqueeze(-1)  # 变为[batch_size, features, 1, 1]
                    # 获取特征
                    batch_features = model.features(data_4d).view(data_4d.size(0), -1)
                    train_features.append(batch_features.cpu().numpy())
                    train_labels.append(target.numpy())
                
                # 处理验证数据
                for data, target in Validation_data_loader:
                    data = data.float().to(device)
                    data_4d = data.unsqueeze(-1).unsqueeze(-1)  # 变为[batch_size, features, 1, 1]
                    batch_features = model.features(data_4d).view(data_4d.size(0), -1)
                    val_features.append(batch_features.cpu().numpy())
                    val_labels.append(target.numpy())
                
                # 处理测试数据
                for data, target in Test_data_loader:
                    data = data.float().to(device)
                    data_4d = data.unsqueeze(-1).unsqueeze(-1)  # 变为[batch_size, features, 1, 1]
                    batch_features = model.features(data_4d).view(data_4d.size(0), -1)
                    test_features.append(batch_features.cpu().numpy())
                    test_labels.append(target.numpy())
            
            train_features = np.vstack(train_features)
            train_labels = np.vstack(train_labels)
            val_features = np.vstack(val_features)
            val_labels = np.vstack(val_labels)
            test_features = np.vstack(test_features)
            test_labels = np.vstack(test_labels)
            
            # 3. 训练XGBoost模型
            print("第3步：训练XGBoost模型...")
            xgb_models = []
            for i in range(num_classes):
                print(f"训练类别 {columns_to_encode[i]} 的XGBoost模型...")
                dtrain = xgb.DMatrix(train_features, label=train_labels[:, i])
                dval = xgb.DMatrix(val_features, label=val_labels[:, i])
                
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'eta': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'seed': 3407
                }
                
                bst = xgb.train(params, dtrain, num_boost_round=100,
                            evals=[(dtrain, 'train'), (dval, 'eval')],
                            early_stopping_rounds=20,
                            verbose_eval=50)
                xgb_models.append(bst)
            
            # 4. 测试XGBoost模型
            print("第4步：测试XGBoost模型...")
            dtest = xgb.DMatrix(test_features)
            all_probs = np.zeros((len(test_labels), num_classes))
            for i, model in enumerate(xgb_models):
                all_probs[:, i] = model.predict(dtest)

            # 定义XGBoost模型的评估函数
            def evaluate_xgboost(probs, labels, threshold=0.5):
                preds = (probs >= threshold).astype(int)
                
                # 计算各项指标
                f1 = f1_score(labels, preds, average='macro') * 100
                accuracy = accuracy_score(labels, preds) * 100
                precision = precision_score(labels, preds, average='macro', zero_division=0) * 100
                recall = recall_score(labels, preds, average='macro', zero_division=0) * 100
                
                # 计算特异性
                cm = multilabel_confusion_matrix(labels, preds)
                specificity = np.mean([cm[i, 0, 0] / (cm[i, 0, 0] + cm[i, 0, 1]) for i in range(len(cm))]) * 100
                
                # 计算AUC
                macro_auc = roc_auc_score(labels, probs, average='macro')
                weighted_auc = roc_auc_score(labels, probs, average='weighted')
                auc_scores = [roc_auc_score(labels[:, i], probs[:, i]) for i in range(labels.shape[1])]
                
                return f1, accuracy, precision, recall, specificity, macro_auc, weighted_auc, auc_scores, probs, labels
            
            # 评估XGBoost模型
            XGB_F1, XGB_accuracy, XGB_precision, XGB_recall, XGB_specificity, XGB_macro_auc, XGB_weighted_auc, XGB_auc_scores, _, _ = evaluate_xgboost(all_probs, test_labels)
            
            # 打印XGBoost模型评估结果
            print(
                f'fold [{fold + 1}/5] XGBoost: '
                f'F1: {XGB_F1:.2f}% | '
                f'Accuracy: {XGB_accuracy:.2f}% | '
                f'Precision: {XGB_precision:.2f}% | '
                f'Recall: {XGB_recall:.2f}% | '
                f'Specificity: {XGB_specificity:.2f}% | '
                f'Macro AUC: {XGB_macro_auc:.3f} | '
                f'Weighted AUC: {XGB_weighted_auc:.3f}'
            )
            
            # 将XGBoost结果保存到文件
            with open('XGBoost_Results.csv', 'a') as f:
                f.write(
                    f'fold [{fold + 1}/5], '
                    f'F1: {XGB_F1:.2f}%, '
                    f'Accuracy: {XGB_accuracy:.2f}%, '
                    f'Precision: {XGB_precision:.2f}%, '
                    f'Recall: {XGB_recall:.2f}%, '
                    f'Specificity: {XGB_specificity:.2f}%, '
                    f'Macro AUC: {XGB_macro_auc:.3f}, '
                    f'Weighted AUC: {XGB_weighted_auc:.3f}\n'
                )


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

    #简单的协变量分析，shap图在后面
    if len(selected_folds_input) == 5:
        # 绘制准确率折线图，x是加入的协变量数量
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_accuracies[x], label=f'Accuracy (x={x + 1})', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of Accuracy for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        # plt.show()
        # 保存为png图在figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/Comparison of Accuracy for Different x Values Across Folds.png')

        # 绘制精确率折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_precisions[x], label=f'Precision (x={x + 1})', marker='s')
        plt.xlabel('Fold Number')
        plt.ylabel('Precision (%)')
        plt.title('Comparison of Precision for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        # plt.show()
        # 保存为png图在figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/Comparison of Precision for Different x Values Across Folds.png')

        # 绘制召回率折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_recalls[x], label=f'Recall (x={x + 1})', marker='^')
        plt.xlabel('Fold Number')
        plt.ylabel('Recall (%)')
        plt.title('Comparison of Recall for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        # plt.show()
        # 保存为png图在figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/Comparison of Recall for Different x Values Across Folds.png')

        # 绘制 F1 分数折线图
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_f1_scores[x], label=f'F1 Score (x={x + 1})', marker='D')
        plt.xlabel('Fold Number')
        plt.ylabel('F1 Score (%)')
        plt.title('Comparison of F1 Score for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        # plt.show()
        # 保存为png图在figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/Comparison of F1 Score for Different x Values Across Folds.png')

    # 特征重要性排序，使用SHAP Value说明特征的重要程度，然后基于特征置换的特征重要性评估方法，也可以看作是一种特征消融实验方法，说明特征的重要性。

    # origin = pd.DataFrame(features_val, columns=all_feature_names)
    pca_feature_names = [f'PC{i}' for i in range(1, features_val.shape[1] + 1)]
    origin = pd.DataFrame(features_val, columns=pca_feature_names)  

    # # 修改前：
    # COLUMNS = origin.columns

    # 修改后：
    COLUMNS = origin.columns  # 使用PCA后生成的特征名称
    num_classes = labels_val.shape[1]  # 获取类别数

    # 使用 shap.sample 对背景数据进行降采样
    K = 32  # 可以根据实际情况调整 K 的值

    # # 修改前：
    # background_data = shap.kmeans(Shap_features, K)

    # 修改后：
    # 使用相同的方差选择器处理SHAP背景数据
    Shap_features_filtered = pca_selector.transform(Shap_features)
    background_data = shap.kmeans(Shap_features_filtered, K)

    # 为全局SHAP分析选择样本
    n_samples_for_global_shap = min(500, len(Shap_features_filtered))  # 限制样本数量以提高计算效率
    global_shap_indices = np.random.choice(len(Shap_features_filtered), n_samples_for_global_shap, replace=False)
    global_shap_samples = Shap_features_filtered[global_shap_indices]

    selected_indices = np.random.choice(len(Shap_features), 3501, replace=True)
    selected_features_val = Shap_features[selected_indices]

    ALL_shap_exp = {}
    ALL_top_features = {}

    
    # 创建SHAP模型
    Shap_model = Model_VGG.Trained_VGG11(model, num_epochs, mean_pca, std_pca, device).eval()
    
    # 执行全局SHAP分析
    print("开始执行全局SHAP分析...")
    global_shap_values = global_shap_analysis(
        model=Shap_model,
        background_data=background_data,
        test_data=global_shap_samples,
        feature_names=pca_feature_names,
        class_names=columns_to_encode,
        output_dir='figure/global_shap'
    )
    print("全局SHAP分析完成！")

    # 将PCA特征的SHAP值映射回原始特征空间
    print("开始将SHAP值映射回原始特征空间...")
    original_feature_importance = map_pca_shap_to_original_features(
        shap_values=global_shap_values,
        pca_model=pca,  # 使用之前训练的PCA模型
        feature_names=all_feature_names,  # 原始特征名称
        class_names=columns_to_encode,
        output_dir='figure/original_feature_shap'
    )
    print("SHAP值映射回原始特征空间完成！")

    for class_idx in range(num_classes):

        # 使用 SHAP 评估特征重要性
        print(f"正在使用 SHAP 评估类别 {columns_to_encode[class_idx]} 的特征重要性...")

        explainer = shap.KernelExplainer(model_wrapper, background_data)
        shap_values = explainer.shap_values(selected_features_val, nsamples=256, main_effects=False, interaction_index=None)

        base_values = explainer.expected_value

        if isinstance(base_values, (int, float)):
            base_values = np.full(len(selected_features_val), base_values)

        shap_exp = shap.Explanation(shap_values, data=selected_features_val, feature_names=all_feature_names, base_values=base_values)

        #保存shap_exp
        ALL_shap_exp[columns_to_encode[class_idx]] = shap_exp

        # 绘制 SHAP 汇总图
        shap.plots.bar(shap_exp, max_display=16)
        # 保存到figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/SHAP Summary Plot for category{columns_to_encode[class_idx]}.png')

        # 绘制 SHAP 摘要图
        shap.plots.beeswarm(shap_exp, max_display=16)
        # 保存到figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/SHAP Beeswarm Plot for category{columns_to_encode[class_idx]}.png')

        # 特征消融实验
        print(f"正在计算类别 {columns_to_encode[class_idx]} 的特征重要性...")
        out = pd.DataFrame({'tag': predict(features_val, probs_Switcher)[:, class_idx]})
        importance_dict = {}  # 用于存储每个特征的重要性计算结果

        for key in COLUMNS:
            copy = origin.copy()
            copy[key] = copy[key].sample(frac=1, random_state=1).reset_index()[key]
            cp_out = predict(copy.values, probs_Switcher)[:, class_idx]
            # 将 out['tag'] 转换为 numpy.ndarray 后再进行减法运算
            diff = (out['tag'].values - cp_out).flatten()
            importance_dict[key] = diff ** 2
            print('key = ', key, ' affect = ', importance_dict[key].sum() ** 0.5)

        # 一次性将所有列合并到 out 中
        importance_df = pd.DataFrame(importance_dict)
        out = pd.concat([out, importance_df], axis=1)

        importance_result = (pd.DataFrame(out.sum(axis=0)) ** 0.5).sort_values(by=0, ascending=False)
        print(f"类别 {class_idx} 的特征重要性排序结果：")
        print(importance_result)

        # 绘制柱状图
        plt.figure(figsize=(15, 6))
        top_features = importance_result.iloc[1:].head(64)  # 取前 64 个特征
        plt.bar(top_features.index, top_features[0])
        plt.xlabel('Features')
        plt.ylabel('Importance of Features')
        plt.title(f'Bar chart of the top 64 feature importances for category{columns_to_encode[class_idx]}')
        plt.xticks(rotation=45, fontsize=6)
        plt.tight_layout()
        #plt.show()
        # 保存为png图在figure文件夹中，命名为这个图的title
        plt.savefig(f'figure/Bar chart of the top 64 feature importances for category{columns_to_encode[class_idx]}.png')



        # 选取 SHAP 高重要性特征（这里以绝对值的均值排序取前 209 个为例）
        shap_importance = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(shap_importance)[::-1]
        top_209_features = np.array(all_feature_names)[sorted_indices[:209]]
        ALL_top_features[columns_to_encode[class_idx]] = top_209_features

    #分层分析，对患病程度，性别，种族进行分层分析。目前用多因素线性回归模型，说明不同层次人群特征的线性相关程度，但部分特征无法推导线性相关程度。
    # 使用SHAP图可以补充说明非线性相关程度特征如何影响判断结果（非线性分析？），目前有瀑布图（一类人群的shap value对模型结果影响的具象化），force图（个例shap value对结果的影响）
    # 参考图片都在文件夹里

    stratify_variable = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']#分别是患病程度，性别，种族

    for i in stratify_variable:
        # 获取分层变量的不同取值
        strata = Multilay_origin[i].unique()

        for class_idx in range(num_classes):
            top_209_features = ALL_top_features[columns_to_encode[class_idx]]

            for stratum in strata:
                # 筛选出当前层的数据
                stratum_mask = Multilay_origin[i] == stratum
                stratum_indices = np.where(stratum_mask)[0]
                # 确保只选择与 selected_features_val 相同索引的数据
                common_indices = np.intersect1d(stratum_indices, selected_indices)
                stratum_data = Multilay_origin[stratum_mask].copy()
                labels_stratum = labels_val[stratum_indices]

                # 构建用于线性回归的数据集
                regression_data = stratum_data[top_209_features].copy()

                regression_data['disease_label'] = labels_stratum[:, class_idx].astype(int)

                # 生成与 shap_values.data 长度一致的布尔数组
                bool_tf = np.isin(selected_indices, common_indices)

                # 这里算一类的平均shap value,用于分层分析的
                try:
                    # 获取当前类别对应的 shap.Explanation 对象
                    shap_exp = ALL_shap_exp[columns_to_encode[class_idx]]

                    # 筛选 shap.Explanation 对象
                    filtered_shap_values = shap_exp.values[bool_tf]
                    filtered_data = shap_exp.data[bool_tf]
                    filtered_base_values = shap_exp.base_values[bool_tf]

                    # 创建筛选后的 shap.Explanation 对象
                    filtered_shap_exp = shap.Explanation(
                        values=filtered_shap_values,
                        data=filtered_data,
                        feature_names=shap_exp.feature_names,
                        base_values=filtered_base_values
                    )

                    new_shap = new_shap_values(ALL_shap_exp[columns_to_encode[class_idx]], bool_tf=bool_tf, method='mean')
                    # 确保传递给 shap.plots.waterfall 的是 shap.Explanation 对象
                    shap.plots.waterfall(new_shap.get_explanation())

                    shap.plots.beeswarm(filtered_shap_exp, max_display=64)
                except Exception as e:
                    print(f"绘制 SHAP 瀑布图时出错，类别 {columns_to_encode[class_idx]}，{i} 为 {stratum} 层，错误信息：{e}")

                # 构建线性回归模型公式
                if np.var(regression_data['disease_label']) == 0:
                    print(f"因变量 'disease_label' 没有变异性，跳过类别 {columns_to_encode[class_idx]} 在 {i} 为 {stratum} 层的线性回归分析。")
                else:
                    formula = 'disease_label ~ ' + ' + '.join(top_209_features)
                    try:
                        Ols_model = smf.ols(formula, data=regression_data).fit(cov_type='HC3')

                        # 输出回归结果
                        print(f"\n类别 {columns_to_encode[class_idx]} 在 {i} 为 {stratum} 层的多因素线性回归分析结果：")
                        print(Ols_model.summary())

                        # 输出相关程度描述
                        print(f"\n在类别 {columns_to_encode[class_idx]} 中，当 {i} 为 {stratum} 时，协变量调整模型显示：")
                        for feature in top_209_features:
                            coef = Ols_model.params[feature]
                            p_value = Ols_model.pvalues[feature]
                            conf_int = Ols_model.conf_int().loc[feature]
                            if p_value < 0.05:
                                if coef > 0:
                                    print(
                                        f"{feature} 与疾病标签呈正相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率增加 {coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。")
                                else:
                                    print(
                                        f"{feature} 与疾病标签呈负相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率降低 {-coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。")
                            else:
                                print(f"{feature} 与疾病标签之间未发现显著的线性关系 (p = {p_value:.4f})。")
                    except Exception as e:
                        print(f"在类别 {columns_to_encode[class_idx]}，{i} 为 {stratum} 层时，线性回归模型拟合失败，错误信息：{e}")


                # # 检查多重共线性
                # X = regression_data[top_209_features]
                # vif = pd.DataFrame()
                # for j in range(X.shape[1]):
                #     try:
                #         vif.loc[j, "VIF Factor"] = variance_inflation_factor(X.values, j)
                #     except ZeroDivisionError:
                #         print(f"自变量 {X.columns[j]} 存在完全多重共线性，VIF 无法计算。考虑删除该自变量。")
                #         vif.loc[j, "VIF Factor"] = np.nan
                # vif["features"] = X.columns
                # print(vif)

# 在这里添加 analyze_shap_performance_correlation 函数
    def analyze_shap_performance_correlation(global_shap_values, test_metrics, class_names, output_dir='figure/shap_performance'):
        """
        分析SHAP值与模型性能指标的关系
        
        参数:
        global_shap_values -- 全局SHAP值列表
        test_metrics -- 测试指标字典，包含'f1', 'accuracy', 'precision', 'recall'等
        class_names -- 类别名称列表
        output_dir -- 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 计算每个类别的平均绝对SHAP值
        mean_abs_shap = []
        for i, shap_values in enumerate(global_shap_values):
            mean_abs_shap.append(np.mean(np.abs(shap_values)))
            
        # 创建性能与SHAP值的对比图
        metrics = ['f1', 'accuracy', 'precision', 'recall']
        for metric in metrics:
            if metric in test_metrics:
                plt.figure(figsize=(10, 6))
                plt.scatter(mean_abs_shap, test_metrics[metric], c=range(len(class_names)), cmap='viridis')
                
                # 添加类别标签
                for i, class_name in enumerate(class_names):
                    plt.annotate(class_name, 
                                (mean_abs_shap[i], test_metrics[metric][i]),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
                
                # 添加趋势线
                z = np.polyfit(mean_abs_shap, test_metrics[metric], 1)
                p = np.poly1d(z)
                plt.plot(mean_abs_shap, p(mean_abs_shap), "r--", alpha=0.8)
                
                plt.xlabel('平均绝对SHAP值')
                plt.ylabel(f'{metric.capitalize()} (%)')
                plt.title(f'SHAP值与{metric.capitalize()}的关系')
                plt.colorbar(label='类别索引')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/shap_vs_{metric}.png')
                plt.close()
    
    # 如果已经计算了全局SHAP值，分析其与模型性能的关系
    if 'global_shap_values' in locals():
        # 收集测试指标
        test_metrics = {
            'f1': [all_f1_scores[0][i] for i in range(len(columns_to_encode))],
            'accuracy': [all_accuracies[0][i] for i in range(len(columns_to_encode))],
            'precision': [all_precisions[0][i] for i in range(len(columns_to_encode))],
            'recall': [all_recalls[0][i] for i in range(len(columns_to_encode))]
        }
        
        # 分析SHAP值与性能的关系
        analyze_shap_performance_correlation(
            global_shap_values=global_shap_values,
            test_metrics=test_metrics,
            class_names=columns_to_encode,
            output_dir='figure/shap_performance'
        )
