# import torch
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import f1_score, multilabel_confusion_matrix
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import torch.nn as nn
# from Model import Model_VGG32
# import torch.nn.functional as F
# import torch.optim as optim
# from numpy import argmax
# from sklearn.metrics import precision_recall_curve
# from Model import mlsmote
# import matplotlib
# import matplotlib.pyplot as plt
# from temperature_scaling import ModelWithTemperature
# from Multilable_temerature_scaling32  import ModelWithTemperature
# import shap
# import seaborn as sns  # 添加seaborn库用于更美观的混淆矩阵绘制
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# import statsmodels.formula.api as smf
# import os

# matplotlib.use('TKAgg')  # 用于解决绘图时的报错问题python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
# seed_value = 3407
# np.random.seed(seed_value)
# torch.manual_seed(seed_value)
# torch.backends.cudnn.deterministic = True

# # 数据集处理
# def encode_labels(File, columns_to_encode):
#     """
#     将多标签编码为单一的32类别标签
#     """
#     # 获取原始的多标签编码
#     one_hot_encoded_df = pd.get_dummies(File[columns_to_encode], columns=columns_to_encode, prefix_sep='_')
#     selected_columns = [col for col in one_hot_encoded_df.columns if col.endswith('_1')]
#     filtered_df = one_hot_encoded_df[selected_columns].to_numpy()
    
#     # 将多标签转换为32类的独热编码
#     class_indices = np.zeros((len(filtered_df), 32), dtype=np.float32)
#     for i in range(len(filtered_df)):
#         # 将二进制标签转换为十进制类别索引
#         binary_str = ''.join([str(int(bit)) for bit in filtered_df[i]])
#         class_idx = int(binary_str, 2)
#         class_indices[i, class_idx] = 1.0
    
#     return class_indices


# def open_excel(filename, columns_to_encode):
#     readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
#     nplist = readbook.T.to_numpy()
#     data = nplist[1:-5].T
#     data = np.float64(data)
#     target = encode_labels(readbook, columns_to_encode=columns_to_encode)
#     all_feature_names = readbook.columns[1:-5]
#     Covariates_features = readbook.columns[1:4]
#     print(all_feature_names)
#     print(Covariates_features)
#     return data, target, all_feature_names, Covariates_features

# # 自定义数据集类
# class NetDataset(Dataset):
#     def __init__(self, features, labels):
#         self.Data = features
#         self.label = labels

#     def __getitem__(self, index):
#         return self.Data[index], self.label[index]

#     def __len__(self):
#         return len(self.Data)


# # 数据标准化与交叉验证划分
# def split_data_5fold(input_data):
#     np.random.seed(3407)
#     indices = np.arange(len(input_data))
#     np.random.shuffle(indices)
#     fold_size = len(input_data) // 5
#     folds_data_index = []
#     for i in range(5):
#         test_indices = indices[i * fold_size: (i + 1) * fold_size]
#         validation_indices = indices[(i + 1) * fold_size: (i + 2) * fold_size] if i < 4 else indices[4 * fold_size:]
#         train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 2) * fold_size:]]) if i < 4 else indices[:4 * fold_size]
#         folds_data_index.append((train_indices, validation_indices, test_indices))
#     return folds_data_index

# # 添加绘制混淆矩阵的函数
# # 修改plot_confusion_matrix函数
# def plot_confusion_matrix(cm, class_names, fold, epoch, is_sum=False):
#     """
#     绘制混淆矩阵
#     """
#     # 对于32类分类，我们需要创建一个32x32的混淆矩阵
#     if is_sum:
#         # 如果是汇总的混淆矩阵，我们需要将多标签混淆矩阵转换为多类别混淆矩阵
#         # 这里简化处理，只显示对角线元素（正确分类）和非对角线元素（错误分类）的总和
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm[:, 1, 1].reshape(-1, 1), annot=True, fmt='d', cmap='Blues', 
#                     xticklabels=['Correct'], yticklabels=class_names)
#         plt.title(f'Confusion Matrix (Sum) - Fold {fold+1}, Epoch {epoch+1}')
#         plt.ylabel('True Label')
#         plt.xlabel('Predicted Label')
#         plt.tight_layout()
#         plt.savefig(f'figure/confusion_matrix_sum_fold{fold+1}_epoch{epoch+1}.png')
#         plt.close()
#     else:
#         # 对于单个批次的混淆矩阵，我们可以显示每个类别的TP, FP, TN, FN
#         fig, axes = plt.subplots(4, 8, figsize=(20, 10))  # 5x7网格用于显示32个类别
#         axes = axes.flatten()
        
#         for i, (ax, name) in enumerate(zip(axes, class_names)):
#             if i < len(cm):
#                 sns.heatmap([[cm[i, 0, 0], cm[i, 0, 1]], [cm[i, 1, 0], cm[i, 1, 1]]], 
#                             annot=True, fmt='d', cmap='Blues', ax=ax,
#                             xticklabels=['Negative', 'Positive'],
#                             yticklabels=['Negative', 'Positive'])
#                 ax.set_title(f'Class: {name}')
#             ax.set_ylabel('True')
#             ax.set_xlabel('Predicted')
        
#         plt.tight_layout()
#         plt.savefig(f'figure/confusion_matrix_fold{fold+1}_epoch{epoch+1}.png')
#         plt.close()

# # 测试函数
# def test(test_loader, probs_Switcher, model_path, Scaled_Model=None):
#     """
#     在测试集上评估模型
#     """
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     all_preds = []
#     all_targets = []
#     all_probs = []
    
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.float().to(device), target.float().to(device)
            
#             if Scaled_Model is not None:
#                 output = Scaled_Model(data)
#             else:
#                 output = model(data)
            
#             # 使用softmax获取概率
#             probs = F.softmax(output, dim=1).cpu().numpy()
#             all_probs.append(probs)
            
#             # 获取预测的类别（独热编码形式）
#             preds = np.zeros_like(probs)
#             preds[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1
#             all_preds.append(preds)
            
#             all_targets.append(target.cpu().numpy())
    
#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)
#     all_probs = np.vstack(all_probs)
    
#     # 计算各项指标
#     f1 = f1_score(all_targets, all_preds, average='macro') * 100
#     accuracy = accuracy_score(np.argmax(all_targets, axis=1), np.argmax(all_preds, axis=1)) * 100
#     precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
#     recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    
#     # 计算特异度
#     cm = multilabel_confusion_matrix(all_targets, all_preds)
#     specificity = np.mean([cm[i, 0, 0] / (cm[i, 0, 0] + cm[i, 0, 1]) for i in range(len(cm))]) * 100
    
#     # 计算AUC
#     macro_auc = 0
#     weighted_auc = 0
#     auc_scores = []
    
#     try:
#         # 计算每个类别的ROC AUC
#         for i in range(all_targets.shape[1]):
#             if len(np.unique(all_targets[:, i])) > 1:  # 确保有正负样本
#                 auc_score = roc_auc_score(all_targets[:, i], all_probs[:, i])
#                 auc_scores.append(auc_score)
#             else:
#                 auc_scores.append(0)
        
#         # 计算宏平均和加权平均AUC
#         macro_auc = np.mean(auc_scores)
#         weighted_auc = roc_auc_score(all_targets, all_probs, average='weighted', multi_class='ovr')
#     except Exception as e:
#         print(f"计算AUC时出错: {e}")
    
#     return f1, accuracy, precision, recall, specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_targets

# # 阈值计算函数
# def Probs_Switcher(outputs, targets=None):
#     """
#     根据输出和目标确定最佳阈值
#     """
#     # 对于32类分类，直接返回0.5作为阈值
#     return 0.5

# # 指标计算函数
# def f1_score_func(outputs, targets, probs_Switcher=None):
#     """
#     计算F1分数和其他指标
#     """
#     outputs = outputs.cpu().detach()
#     targets = targets.cpu().detach()
    
#     # 使用softmax获取概率
#     probs = F.softmax(outputs, dim=1).numpy()
    
#     # 获取预测的类别（独热编码形式）
#     preds = np.zeros_like(probs)
#     preds[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1
    
#     # 计算各项指标
#     f1 = f1_score(targets, preds, average='macro') * 100
#     accuracy = accuracy_score(np.argmax(targets, axis=1), np.argmax(preds, axis=1)) * 100
#     precision = precision_score(targets, preds, average='macro', zero_division=0) * 100
#     recall = recall_score(targets, preds, average='macro', zero_division=0) * 100
    
#     return f1, accuracy, precision, recall, preds, probs

# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_num, trainable_num

# # 自定义预测函数
# def predict(features, probs_Switcher=None):
#     """
#     使用模型进行预测
#     """
#     model.eval()
#     with torch.no_grad():
#         features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
#         outputs = model(features_tensor)
#         # 对于32类分类，使用softmax获取概率
#         probs = F.softmax(outputs, dim=1).cpu().numpy()
#         # 获取最高概率的类别
#         predictions = np.zeros_like(probs)
#         predictions[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1
#         return predictions


# # SHAP分析
# def Shap_predict(X):
#     X = torch.tensor(X, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         output = model(X)
#         probs_sigmoid = torch.sigmoid(output)
#         probs = probs_sigmoid.cpu().detach()
#     return probs[:, class_idx]

# def model_wrapper(x):
#     """
#     SHAP解释器的模型包装函数
#     """
#     model.eval()
#     with torch.no_grad():
#         x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
#         output = model(x_tensor)
#         # 使用softmax获取概率
#         probs = F.softmax(output, dim=1).cpu().numpy()
#         return probs

# def global_shap_analysis(model, background_data, test_data, feature_names, class_names, output_dir='figure/global_shap'):
#     """
#     执行全局SHAP分析并生成可视化图表
    
#     参数:
#     model -- 训练好的模型
#     background_data -- 用于SHAP解释器的背景数据
#     test_data -- 用于生成SHAP值的测试数据
#     feature_names -- 特征名称列表
#     class_names -- 类别名称列表
#     output_dir -- 输出目录
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 获取模型所在设备
#     device = next(model.parameters()).device

#     # 检查特征名称长度
#     if len(feature_names) != test_data.shape[1]:
#         feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]
#         print(f"警告：自动生成特征名称，长度为 {len(feature_names)}")
    
#     # 创建解释器，确保数据和模型在同一设备上
#     explainer = shap.KernelExplainer(
#         lambda x: model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().detach().numpy(),
#         background_data
#     )

#     # 计算SHAP值
#     shap_values = explainer.shap_values(test_data, nsamples=500)
#     # 打印形状和类型信息
#     print("shap_values 类型:", type(shap_values))
    
#     # 检查shap_values是列表还是数组
#     if isinstance(shap_values, list):
#         print("shap_values 是列表，长度:", len(shap_values))
#         if len(shap_values) > 0:
#             print("第一个元素形状:", np.array(shap_values[0]).shape)
#     else:
#         print("shap_values 形状:", shap_values.shape)
    
#     # 存储所有类别的SHAP值
#     all_shap_values = []
    
#     # 为每个类别处理SHAP值
#     for i, class_name in enumerate(class_names):
#         print(f"计算类别 {class_name} 的全局SHAP值...")
        
#         # 根据shap_values的类型获取对应类别的值
#         if isinstance(shap_values, list):
#             # 如果是列表，每个元素对应一个类别
#             if i < len(shap_values):
#                 class_shap_values = shap_values[i]
#             else:
#                 print(f"警告：类别索引 {i} 超出shap_values列表长度 {len(shap_values)}")
#                 continue
#         else:
#             # 如果是数组，假设形状为 (samples, features, classes)
#             class_shap_values = shap_values[:, :, i]
            
#         print(f"类别 {class_name} 的SHAP值形状:", np.array(class_shap_values).shape)
#         all_shap_values.append(class_shap_values)
        
#         # 创建SHAP解释对象
#         shap_exp = shap.Explanation(
#             values=class_shap_values,
#             data=test_data,
#             feature_names=feature_names
#         )

#         # 绘制条形图 - 使用英文命名
#         plt.figure(figsize=(10, 8))
#         try:
#             # 计算特征重要性（平均绝对SHAP值）
#             feature_importance = np.abs(np.array(class_shap_values)).mean(0)
#             # 获取排序索引
#             sorted_idx = np.argsort(feature_importance)
#             # 选择最重要的特征
#             top_features = min(20, len(feature_names))
#             plt.barh(range(top_features), feature_importance[sorted_idx[-top_features:]])
#             plt.yticks(range(top_features), [feature_names[i] for i in sorted_idx[-top_features:]])
#             plt.xlabel('Mean |SHAP value|')
#             plt.title(f'Feature Importance for Category {class_name}')
#         except Exception as e:
#             print(f"绘制条形图时出错: {e}")
#             try:
#                 shap.summary_plot(class_shap_values, test_data, feature_names=feature_names,
#                                   plot_type="bar", show=False, max_display=20)
#             except Exception as e2:
#                 print(f"使用shap.summary_plot绘制条形图也失败: {e2}")
        
#         plt.tight_layout()
#         # 使用英文命名保存文件
#         plt.savefig(f'{output_dir}/global_bar_plot_category_{class_name}.png')
#         plt.close()
        
#         # 绘制摘要图 - 使用英文命名
#         plt.figure(figsize=(12, 8))
#         try:
#             shap.summary_plot(class_shap_values, test_data, feature_names=feature_names, show=False, max_display=20)
#             plt.title(f'SHAP Summary Plot for Category {class_name}')
#             plt.tight_layout()
#             plt.savefig(f'{output_dir}/global_summary_plot_category_{class_name}.png')
#         except Exception as e:
#             print(f"绘制摘要图时出错: {e}")
#         plt.close()

#     # 绘制多项式图 - Polynomial-SHAP plot of the data.
#     # 整合 SHAP 值为三维数组，处理列表和数组两种情况
#     if isinstance(shap_values, list):
#         # 确保所有元素都是数组并且形状一致
#         all_arrays = [np.array(sv) for sv in all_shap_values]
#         if all(arr.shape == all_arrays[0].shape for arr in all_arrays):
#             shap_3d = np.stack(all_arrays, axis=2)
#         else:
#             print("警告：不同类别的SHAP值形状不一致，跳过多项式图绘制")
#             return all_shap_values
#     else:
#         shap_3d = shap_values
    
#     # 计算每个特征在各分类下的平均绝对 SHAP 值
#     mean_abs_shap = np.abs(shap_3d).mean(axis=0)  # 形状：(特征数, 类别数)
#     agg_shap_df = pd.DataFrame(mean_abs_shap, columns=class_names, index=feature_names)

#     # 按特征重要性总和排序（模拟示例图特征顺序）
#     feature_order = agg_shap_df.sum(axis=1).sort_values(ascending=False).index
#     agg_shap_df = agg_shap_df.loc[feature_order]

#     plt.figure(figsize=(18, 8))
#     bottom = np.zeros(len(agg_shap_df))
#     colors = sns.color_palette("tab10", len(class_names))  # 生成类别对应颜色

#     for i, disease in enumerate(class_names):
#         plt.bar(
#             agg_shap_df.index,
#             agg_shap_df[disease],
#             bottom=bottom,
#             label=disease,
#             color=colors[i],
#             edgecolor="black",  # 显示条形边界
#             linewidth=0.5
#         )
#         bottom += agg_shap_df[disease]

#     plt.xlabel("Top Most Important Features in Predicting Liver Disease", fontsize=12)
#     plt.ylabel("mean(|SHAP value|) / average impact on model output magnitude", fontsize=12)
#     plt.title("Polynomial-SHAP plot of the data", fontsize=14)
#     plt.xticks(rotation=45, ha="right", fontsize=10)  # 旋转并右对齐特征标签
#     plt.legend(
#         title="",
#         bbox_to_anchor=(1.02, 1),
#         loc="upper left",
#         fontsize=10,
#         frameon=False  # 不显示图例边框
#     )
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/polynomial_shap_plot.png")
#     plt.show()

#     return all_shap_values

# # 添加全局SHAP解释函数
# def map_pca_shap_to_original_features(shap_values, pca_model, feature_names, class_names, output_dir='figure/original_feature_shap'):
#     """
#     将PCA特征的SHAP值映射回原始特征空间
    
#     参数:
#     shap_values -- PCA特征的SHAP值列表
#     pca_model -- 训练好的PCA模型
#     feature_names -- 原始特征名称列表
#     class_names -- 类别名称列表
#     output_dir -- 输出目录
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 获取PCA组件
#     components = pca_model.components_
    
#     # 对每个类别进行处理
#     for i, class_name in enumerate(class_names):
#         print(f"将类别 {class_name} 的SHAP值映射回原始特征空间...")
        
#         # 获取当前类别的SHAP值
#         if isinstance(shap_values, list):
#             class_shap_values = shap_values[i]
#         else:
#             class_shap_values = shap_values
        
#         # 计算原始特征的重要性
#         # SHAP值与PCA组件的点积
#         original_importance = np.zeros(len(feature_names))
        
#         # 对每个样本的SHAP值
#         for sample_idx in range(class_shap_values.shape[0]):
#             # 对每个PCA特征
#             for pca_idx in range(class_shap_values.shape[1]):
#                 # 将SHAP值分配给原始特征
#                 for feat_idx in range(len(feature_names)):
#                     # 权重是PCA组件中原始特征的贡献
#                     weight = abs(components[pca_idx, feat_idx])
#                     # 将SHAP值按权重分配给原始特征
#                     original_importance[feat_idx] += abs(class_shap_values[sample_idx, pca_idx]) * weight
        
#         # 归一化重要性分数
#         original_importance = original_importance / original_importance.sum() * 100
        
#         # 创建原始特征重要性的DataFrame
#         importance_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': original_importance
#         })
        
#         # 按重要性排序
#         importance_df = importance_df.sort_values('Importance', ascending=False)
        
#         # 保存到CSV
#         importance_df.to_csv(f'{output_dir}/original_feature_importance_{class_name}.csv', index=False)
        
#         # 绘制前20个最重要的原始特征
#         plt.figure(figsize=(12, 8))
#         top_features = importance_df.head(20)
#         plt.barh(range(len(top_features)), top_features['Importance'], align='center')
#         plt.yticks(range(len(top_features)), top_features['Feature'])
#         plt.xlabel('Relative Importance (%)')
#         plt.title(f'Top 20 Original Features Importance for {class_name}')
#         plt.tight_layout()
#         plt.savefig(f'{output_dir}/original_feature_importance_{class_name}.png')
#         plt.close()
        
#         print(f"类别 {class_name} 的前10个最重要原始特征:")
#         for idx, row in importance_df.head(10).iterrows():
#             print(f"{row['Feature']}: {row['Importance']:.2f}%")
    
#     return importance_df

# class new_shap_values():
#     def __init__(self, shap_values, bool_tf=None, method='sum'):
#         self.feature_names = shap_values.feature_names
#         if method == 'sum':
#             self.data = np.nansum(shap_values.data[bool_tf], axis=0)
#             self.values = np.nansum(shap_values.values[bool_tf], axis=0)
#             self.base_values = np.nansum(shap_values.base_values[bool_tf], axis=0)
#         elif method == 'mean':
#             self.data = np.nanmean(shap_values.data[bool_tf], axis=0)
#             self.values = np.nanmean(shap_values.values[bool_tf], axis=0)
#             self.base_values = np.nanmean(shap_values.base_values[bool_tf], axis=0)
#         else:
#             print('sry,not right method.')
#             return
#         self.explanation = shap.Explanation(values=self.values, data=self.data, feature_names=self.feature_names,
#                                         base_values=self.base_values)

#     def get_explanation(self):
#         return self.explanation


# def plot_roc_curves(all_labels, all_probs, class_names):
#     plt.figure(figsize=(10, 8))

#     # 为每个类别绘制ROC曲线
#     for i in range(len(class_names)):
#         fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curves')
#     plt.legend(loc="lower right")
#     #plt.show()
#     # 保存为png图在figure文件夹中，命名为这个图的title
#     plt.savefig(f'figure/Receiver Operating Characteristic (ROC) Curves.png')

# if __name__ == '__main__':  
#     # columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
#     # columns_to_encode = ['MCQ160C', 'MCQ160F']
#     columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
#     # columns_to_encode = ['MCQ160B', 'MCQ160D']

#     features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

#     class_names = []
#     for i in range(32):
#         binary = format(i, '05b')  # 将数字转换为5位二进制
#         class_names.append(f"Class_{binary}")


#     # 特征数量选择
#     # num_columns_to_select = 202
#     # selected_column_indices = np.random.choice(features.shape[1], size=num_columns_to_select, replace=False)
#     # features = features[:, selected_column_indices]
#     # selected_feature_names = all_feature_names[selected_column_indices]

#     # 测试集的数据不要改定义！！！（一定要原始的数据集）
#     features_val = features
#     labels_val = labels

#     Shap_features = features_val

#     Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)

#     # 修改这里：使用class_names作为列名，而不是columns_to_encode
#     labels_DF = pd.DataFrame(labels, columns=class_names)
#     data_DF = pd.DataFrame(features, columns=all_feature_names)
#     X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)  # Getting minority instance of that datframe
#     X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)  # Applying MLSMOTE to augment the dataframe

#     features = np.concatenate((features, np.float64(X_res)), axis=0)
#     labels = np.concatenate((labels, np.float64(y_res)), axis=0)

#     mean, std = np.mean(features, axis=0), np.std(features, axis=0)
#     mean_val, std_val = np.mean(features_val, axis=0), np.std(features_val, axis=0)
#     for i in range(len(std)):
#         if std[i] == 0:
#             std[i] = 1e-8  # 将标准差为零的值设置为一个很小的数，避免除以零

#     # # 修改前：
#     # features = (features - mean) / std
#     # features_val = (features_val - mean_val) / (std_val + 1e-8)
#     # features = features.reshape(features.shape[0], -1)

#     # 修改后：
#     from sklearn.feature_selection import VarianceThreshold

#     # # 添加方差过滤（阈值设为0.01）
#     # selector = VarianceThreshold(threshold=0.01)
#     # features = selector.fit_transform(features)
#     # features_val = selector.transform(features_val)

#     # # 更新特征名称
#     # mask = selector.get_support()
#     # all_feature_names = all_feature_names[mask]

#     # 重新计算标准化所需的均值、标准差（基于筛选后的 features）
#     mean_f = np.mean(features, axis=0)
#     std_f = np.std(features, axis=0)
#     for i in range(len(std_f)):
#         if std_f[i] == 0:
#             std_f[i] = 1e-8

#     features = (features - mean_f) / std_f
#     features_val = (features_val - mean_f) / (std_f + 1e-8)
#     features = features.reshape(features.shape[0], -1)
#     # 降维前的形状
#     print("PCA降维前，训练集形状：", features.shape)
#     print("PCA降维前，验证集形状：", features_val.shape)

#     # 加入PCA降维，保留95%的信息
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=0.95)
#     # # 修改前：
#     # features = pca.fit_transform(features)
#     # features_val = pca.transform(features_val)

#     # 修改后：
#     # PCA后再次过滤零方差
#     pca_selector = VarianceThreshold(threshold=0.01)
#     features = pca_selector.fit_transform(pca.fit_transform(features))
#     features_val = pca_selector.transform(pca.transform(features_val))

#     # 更新PCA特征名称
#     pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1] + 1)]
#     print("PCA降维后，训练集形状：", features.shape)
#     print("PCA降维后，验证集形状：", features_val.shape)
#     # 更新 Shap_features 为 PCA 后的特征
    
#     Shap_features = features_val.copy()
#     # pause = input("Press Enter to continue...")

#     # 重新计算PCA后数据的均值和标准差（例如，基于训练集features）
#     mean_pca, std_pca = np.mean(features, axis=0), np.std(features, axis=0)

#     folds_data_index = split_data_5fold(features)

#     num_x = len(Covariates_features)
#     num_folds = len(folds_data_index)

#     # 用于保存不同x的指标
#     all_accuracies = [[] for _ in range(num_x)]
#     all_precisions = [[] for _ in range(num_x)]
#     all_recalls = [[] for _ in range(num_x)]
#     all_f1_scores = [[] for _ in range(num_x)]
#     all_macro_auc = [[] for _ in range(num_x)]
#     all_weighted_auc = [[] for _ in range(num_x)]

#     # 让用户输入想要运行的fold编号，用逗号分隔
#     # selected_folds_input = input("请输入想要运行的fold编号（用逗号分隔，例如：1,3,5）：")  #这里输入之后，变量的值是字符串，需要转换为列表
#     # 为了调试方便，这里写死selected_folds_input为1,2,3,4,5
#     selected_folds_input = '1,2,3,4,5'
#     selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

#     for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
#         if fold not in selected_folds:
#             continue  # 跳过未选择的fold

#         l1_weight = 0.070
#         num_classes = 32  # Adjust the number of classes
#         num_epochs = 350
#         batch_size = 256
#         input_length = batch_size
#         for x in range(len(Covariates_features)):

#             #model = Model_VGG32.VGG11(num_classes=num_classes, in_channels=len(all_feature_names)-len(Covariates_features), Covariates_features_length=x)
#             #model = Model_VGG32.VGG11(num_classes=num_classes, in_channels=features.shape[1], Covariates_features_length=x)
#             # 将 Covariates_features_length 固定为0，确保输入通道与PCA后的特征一致。
#             model = Model_VGG32.VGG11(num_classes=num_classes, in_channels=features.shape[1], epoch=num_epochs)

#             # 温度缩放模型
#             scaled_model = ModelWithTemperature(model, len(columns_to_encode))

#             pos_weight = torch.tensor(1.0)
#             criterion = nn.CrossEntropyLoss()
#             # optimizer = optim.Adam(model.parameters(), lr=0.00275)
#             optimizer = optim.Adam(model.parameters(), lr=0.001)
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.475) 
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             model.to(device)

#             scaled_model.to(device)

#             total_num, trainable_num = get_parameter_number(model)
#             print(f'Fold {fold + 1}, Total params: {total_num}, Trainable params: {trainable_num}')

#             Validation_size = len(features_val) // 5
#             Test_size = Validation_size

#             indices = np.arange(len(features_val))
#             np.random.shuffle(indices)
#             validation_index = indices[(fold + 1) * Validation_size: (fold + 2) * Validation_size] if fold < 4 else indices[4 * Validation_size:]
#             test_indices = indices[fold * Test_size: (fold + 1) * Test_size]

#             # print(validation_index)

#             trainX = features[train_index]
#             trainY = labels[train_index]
#             valX = features_val[validation_index]
#             valY = labels_val[validation_index]
#             testX = features_val[test_indices]
#             testY = labels_val[test_indices]

#             Train_data = NetDataset(trainX, trainY)
#             Validation_data = NetDataset(valX, valY)
#             Test_data = NetDataset(testX, testY)

#             Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=True)
#             Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
#             Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)

#             # 初始化存储变量
#             all_preds = np.empty((0, num_classes))
#             all_targets = np.empty((0, num_classes))

#             valid_f1_scores = []
#             valid_precision_scores = []

#             # 添加用于绘制学习曲线的列表
#             train_losses = []
#             valid_f1_history = []
#             valid_accuracy_history = []
#             valid_precision_history = []
#             valid_recall_history = []

#             # 添加早停相关变量  
#             best_valid_f1 = 0
#             patience = 15  # 设置耐心值，连续多少个epoch没有提升则停止
#             counter = 0
#             best_epoch = 0

#             for epoch in range(num_epochs):
#                 model.train()
#                 train_loss = 0.0
#                 for batch_idx, (data, target) in enumerate(Train_data_loader):
#                     data, target = data.float().to(device), target.float().to(device)
#                     optimizer.zero_grad()
#                     output = model(data)
#                     loss = criterion(output, target)

#                     # L1正则化
#                     l1_criterion = nn.L1Loss()
#                     l1_loss = 0
#                     for param in model.parameters():
#                         l1_loss += l1_criterion(param, torch.zeros_like(param))
#                     loss += l1_weight * l1_loss  # l1_weight是L1正则化的权重，可以根据需要调整

#                     loss.backward()
#                     optimizer.step()
#                     train_loss += loss.item() * data.size(0)

#                 average_train_loss = train_loss / len(Train_data_loader.dataset)
#                 scheduler.step()

#                 # 记录训练损失
#                 train_losses.append(average_train_loss)

#                 model.eval()
#                 n = 0
#                 valid_f1 = 0
#                 valid_accuracy = 0
#                 valid_precision = 0
#                 valid_recall = 0

#                 Scaled_f1 = 0
#                 Scaled_accuracy = 0
#                 Scaled_precision = 0
#                 Scaled_recall = 0
#                 with torch.no_grad():
#                     for data, target in Validation_data_loader:
#                         n += 1
#                         data, target = data.float().to(device), target.float().to(device)
#                         output = model(data)
#                         probs_Switcher = Probs_Switcher(output, target)
#                         f1, accuracy, precision, recall, preds, _= f1_score_func(output, target, probs_Switcher)

#                         #温度缩放
#                         ##温度缩放需要在经过一定训练阶段后进行，这里我们使用早停的计数，当小于patience的一定倍数时，我们开始温度缩放
#                         if counter >= patience/3 :
#                             if n == 1:
#                                 scaled_model.set_temperature(Train_data_loader, probs_Switcher)
#                                 scaled_model.to(device)
#                                 # print(f'Before scaling:\n:{output}')
#                                 output = scaled_model(data)
#                                 # print(f'After scaling:\n:{output}')
#                                 scaled_f1, scaled_accuracy, scaled_precision, scaled_recall, _, _ = f1_score_func(output, target,
#                                                                                                             probs_Switcher)
#                                 Scaled_f1 += scaled_f1
#                                 Scaled_accuracy += scaled_accuracy
#                                 Scaled_precision += scaled_precision
#                                 Scaled_recall += scaled_recall

#                         valid_f1 += f1
#                         valid_accuracy += accuracy
#                         valid_precision += precision
#                         valid_recall += recall

#                         # 保存预测值和真实值
#                         all_preds = np.concatenate((all_preds, preds), axis=0)
#                         all_targets = np.concatenate((all_targets, target.cpu().numpy()), axis=0)

#                     valid_f1 /= n
#                     valid_accuracy /= n
#                     valid_precision /= n
#                     valid_recall /= n

#                     Scaled_f1 /= n
#                     Scaled_accuracy /= n
#                     Scaled_precision /= n
#                     Scaled_recall /= n

#                 # 记录验证集指标
#                 valid_f1_history.append(valid_f1)
#                 valid_accuracy_history.append(valid_accuracy)
#                 valid_precision_history.append(valid_precision)
#                 valid_recall_history.append(valid_recall)

#                 valid_f1_scores.append(valid_f1)
#                 valid_precision_scores.append(valid_precision)
#                 print(
#                     f'Epoch [{epoch + 1}/{num_epochs}]: Average Train Loss: {average_train_loss:.4f} '
#                     f'Validation F1: {valid_f1:.2f}%, Validation Accuracy: {valid_accuracy:.2f}%, Validation Precision: {valid_precision:.2f}%, Validation Recall: {valid_recall:.2f}%')
#                 if Scaled_f1 != 0:
#                     print(
#                     f'Epoch [{epoch + 1}/{num_epochs}]: Scaled F1: {Scaled_f1:.2f}%, '
#                     f'Scaled Accuracy: {Scaled_accuracy:.2f}%, Scaled Precision: {Scaled_precision:.2f}%, Scaled Recall: {Scaled_recall:.2f}%')

#                 # 在第300次迭代后绘制混淆矩阵
#                 if epoch + 1 >= 300:
#                     cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
#                     cm = multilabel_confusion_matrix(target.cpu().numpy(), preds)
#                     print(cm_sum)
#                     print(cm)

#                     # 添加绘制混淆矩阵图片的代码
#                     plot_confusion_matrix(cm, class_names, fold, epoch)
#                     plot_confusion_matrix(cm_sum, class_names, fold, epoch, is_sum=True)

#                 # 早停机制：检查验证集F1分数是否提高
#                 if valid_f1 > best_valid_f1:
#                     best_valid_f1 = valid_f1
#                     counter = 0
#                     best_epoch = epoch
#                     # 保存最佳模型
#                     torch.save(model.state_dict(), f'single/best_model_{fold}_{x}.ckpt')
#                     # 同时保存温度缩放模型
#                     torch.save(scaled_model.state_dict(), f'single/best_scaled_model_{fold}_{x}.ckpt')
#                 else:
#                     counter += 1
#                     print(f'EarlyStopping counter: {counter} out of {patience}')
#                     if counter >= patience:
#                         print(f'Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with F1: {best_valid_f1:.2f}%')
#                         # 在早停时绘制sum版本的当前折的混淆矩阵
#                         cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
#                         print(f"早停时的混淆矩阵 (Fold {fold+1}):")
#                         print(cm_sum)
#                         # 绘制混淆矩阵图片
#                         plot_confusion_matrix(cm_sum, columns_to_encode, fold, epoch, is_sum=True)
                        
#                         break

#             # 绘制学习曲线
#             plt.figure(figsize=(12, 10))
            
#             # 创建两个子图
#             plt.subplot(2, 1, 1)
#             plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.title(f'Training Loss Curve (Fold {fold+1}, x={x+1})')
#             plt.grid(True)
#             plt.legend()
            
#             plt.subplot(2, 1, 2)
#             plt.plot(range(1, len(valid_f1_history) + 1), valid_f1_history, label='Validation F1', color='red')
#             plt.plot(range(1, len(valid_accuracy_history) + 1), valid_accuracy_history, label='Validation Accuracy', color='blue')
#             plt.plot(range(1, len(valid_precision_history) + 1), valid_precision_history, label='Validation Precision', color='green')
#             plt.plot(range(1, len(valid_recall_history) + 1), valid_recall_history, label='Validation Recall', color='purple')
#             plt.axvline(x=best_epoch+1, color='black', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
#             plt.xlabel('Epochs')
#             plt.ylabel('Score (%)')
#             plt.title(f'Validation Metrics Curve (Fold {fold+1}, x={x+1})')
#             plt.grid(True)
#             plt.legend()
            
#             plt.tight_layout()
            
#             # 确保figure目录存在
#             if not os.path.exists('figure'):
#                 os.makedirs('figure')
                
#             plt.savefig(f'figure/learning_curve_fold{fold+1}_x{x+1}.png')
#             plt.close()       
#             # 保存模型
#             # torch.save(model.state_dict(), f'single/best_model_{fold}_{x}.ckpt')
#             torch.save(model.state_dict(), f'single/model_{fold}_{x}.ckpt')  

#             # 测试集
#             Test_F1, Test_accuracy, Test_precision, Test_recall, Test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt', Scaled_Model=None)
#             Scaled_F1, Scaled_accuracy, Scaled_precision, Scaled_recall, Scaled_specificity, Scaled_macro_auc, Scaled_weighted_auc, _, _, _ = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt', Scaled_Model = scaled_model)
#             # plot_roc_curves(all_labels, all_probs, columns_to_encode)
#             print(
#                 f'fold [{fold + 1}/5]: '
#                 f'Test F1: {Test_F1:.2f}% | '
#                 f'Accuracy: {Test_accuracy:.2f}% | '
#                 f'Precision: {Test_precision:.2f}% | '
#                 f'Recall: {Test_recall:.2f}% | '
#                 f'Specificity: {Test_specificity:.2f}% | '
#                 f'Macro AUC: {macro_auc:.3f} | '
#                 f'Weighted AUC: {weighted_auc:.3f}'
#             )

#             print(
#                 f'fold [{fold + 1}/5]: '
#                 f'Scaled F1: {Scaled_F1:.2f}% | '
#                 f'Scaled Accuracy: {Scaled_accuracy:.2f}% | '
#                 f'Scaled Precision: {Scaled_precision:.2f}% | '
#                 f'Scaled Recall: {Scaled_recall:.2f}% | '
#                 f'Scaled Specificity: {Scaled_specificity:.2f}% | '
#                 f'Scaled Macro AUC: {Scaled_macro_auc:.3f} | '
#                 f'Scaled Weighted AUC: {Scaled_weighted_auc:.3f}'
#             )
#             # 将print的内容保存到result.csv文件,每次运行新起一行
#             with open('Test.csv', 'a') as f:
#                 f.write(
#                     f'fold [{fold + 1}/5], '
#                     f'Test F1: {Test_F1:.2f}%, '
#                     f'Accuracy: {Test_accuracy:.2f}%, '
#                     f'Precision: {Test_precision:.2f}%, '
#                     f'Recall: {Test_recall:.2f}%, '
#                     f'Specificity: {Test_specificity:.2f}%, '
#                     f'Macro AUC: {macro_auc:.3f}, '
#                     f'Weighted AUC: {weighted_auc:.3f}\n'
#                 )

#             with open('Scaled.csv', 'a') as f:
#                 f.write(
#                     f'fold [{fold + 1}/5], '
#                     f'Scaled F1: {Scaled_F1:.2f}%, '
#                     f'Scaled Accuracy: {Scaled_accuracy:.2f}%, '
#                     f'Scaled Precision: {Scaled_precision:.2f}%, '
#                     f'Scaled Recall: {Scaled_recall:.2f}%, '
#                     f'Scaled Specificity: {Scaled_specificity:.2f}%, '
#                     f'Scaled Macro AUC: {Scaled_macro_auc:.3f}, '
#                     f'Scaled Weighted AUC: {Scaled_weighted_auc:.3f}\n'
#                 )

#             # 保存当前 x 和 fold 的指标
#             all_accuracies[x].append(Test_accuracy)
#             all_precisions[x].append(Test_precision)
#             all_recalls[x].append(Test_recall)
#             all_f1_scores[x].append(Test_F1)
#             all_macro_auc[x].append(macro_auc)
#             all_weighted_auc[x].append(weighted_auc)

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

    
#     # 创建SHAP模型
#     Shap_model = Model_VGG32.Trained_VGG11(model, num_epochs, mean_pca, std_pca, device).eval()
    
#     # 执行全局SHAP分析
#     print("开始执行全局SHAP分析...")
#     global_shap_values = global_shap_analysis(
#         model=Shap_model,
#         background_data=background_data,
#         test_data=global_shap_samples,
#         feature_names=pca_feature_names,
#         class_names=class_names,
#         output_dir='figure/global_shap'
#     )
#     print("全局SHAP分析完成！")

#     # 将PCA特征的SHAP值映射回原始特征空间
#     print("开始将SHAP值映射回原始特征空间...")
#     original_feature_importance = map_pca_shap_to_original_features(
#         shap_values=global_shap_values,
#         pca_model=pca,  # 使用之前训练的PCA模型
#         feature_names=all_feature_names,  # 原始特征名称
#         class_names=class_names,
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



import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
from sklearn.metrics import confusion_matrix
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
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from datetime import datetime
import math

matplotlib.use('TKAgg')  # 用于解决绘图时的报错问题
seed_value = 3407
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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
        binary_str = ''.join(map(str, labels[i].astype(int)))
        multiclass_labels[i] = int(binary_str, 2)
    return multiclass_labels

# 数据标准化与交叉验证划分
def split_data_5fold(input_data):
    """
    将输入数据划分为5个fold，每个fold包含：
    - 训练集：前3个fold (60%)
    - 验证集：第4个fold (20%)
    - 测试集：第5个fold (20%)
    比例调整为 3:1:1（训练:验证:测试）
    """
    np.random.seed(3407)
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    n_samples = len(input_data)
    fold_size = n_samples // 5
    folds_data_index = []
    for i in range(5):
        test_start = i * fold_size
        test_end = (i+1) * fold_size
        test_indices = indices[test_start:test_end]
        val_start = test_end
        val_end = (i+2)*fold_size if i < 3 else n_samples
        validation_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:test_start], indices[val_end:]])
        folds_data_index.append((train_indices, validation_indices, test_indices))
    return folds_data_index

# 自定义数据集类（单标签分类）
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义网络（VGG32Class）及相关模块
class ECANet(nn.Module):
    def __init__(self, in_channels, b=1, gamma=2):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.b = b
        self.gamma = gamma
        self.kernel_size = int(abs((math.log(self.in_channels, 2) + self.b) / self.gamma))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(self.in_channels, in_channels, kernel_size=self.kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = y.reshape(batch_size, channels, -1)
        y = self.conv1d(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

class VGG32Class(nn.Module):
    def __init__(self, input_dim, num_classes=32):
        super(VGG32Class, self).__init__()
        self.epoch = 0 
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Classifier_eca = ECANet(1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, 125),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(125, num_classes),
        )
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.features(x)
        x = self.Classifier_eca(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x * (1 + math.log(self.epoch+1))
        return x

# 温度缩放模型
class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
    def set_temperature(self, valid_loader):
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward()
                return loss
            optimizer.step(eval)
        print(f"温度参数设置为: {self.temperature.item():.3f}")

# 绘制混淆矩阵函数
def plot_confusion_matrix(cm, class_names, fold, epoch):
    try:
        n_classes_actual = cm.shape[0]
        if n_classes_actual < len(class_names):
            new_cm = np.zeros((len(class_names), len(class_names)))
            new_cm[:n_classes_actual, :n_classes_actual] = cm
            cm = new_cm
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        cm_normalized = np.nan_to_num(cm_normalized)
        os.makedirs('figure', exist_ok=True)
        plt.switch_backend('Agg')
        plt.figure(figsize=(15, 12))
        plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(min(len(class_names), cm_normalized.shape[0]))
        plt.xticks(tick_marks, range(len(tick_marks)), rotation=45)
        plt.yticks(tick_marks, range(len(tick_marks)))
        plt.title(f'Normalized Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'figure/vgg_32class_confusion_matrix_fold{fold+1}_epoch{epoch+1}.png', dpi=100)
        plt.close()
        if len(class_names) > 10:
            top_errors = []
            for i in range(min(cm_normalized.shape[0], len(class_names))):
                if i < cm_normalized.shape[0]:
                    row = cm_normalized[i].copy()
                    if i < row.size:
                        row[i] = 0
                    if np.max(row) > 0.1:
                        top_errors.append((i, np.argmax(row), np.max(row)))
            if top_errors:
                plt.figure(figsize=(10, 8))
                error_classes = [f"{a}->{p}" for a, p, _ in top_errors]
                error_rates = [e[2] for e in top_errors]
                plt.bar(range(len(error_rates)), error_rates)
                plt.xticks(range(len(error_rates)), error_classes, rotation=45)
                plt.title(f'Top Misclassifications (Fold {fold+1}, Epoch {epoch+1})')
                plt.xlabel('Actual -> Predicted')
                plt.ylabel('Error Rate')
                plt.tight_layout()
                plt.savefig(f'figure/vgg_32class_top_errors_fold{fold+1}_epoch{epoch+1}.png', dpi=100)
                plt.close()
                print(f"成功保存主要错误图: figure/vgg_32class_top_errors_fold{fold+1}_epoch{epoch+1}.png")
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")
        import traceback
        traceback.print_exc()

# 绘制ROC曲线函数
def plot_roc_curves(y_test, probs, n_classes):
    plt.figure(figsize=(10, 8))
    y_test_onehot = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        y_test_onehot[i, y_test[i]] = 1
    for i in range(n_classes):
        if np.sum(y_test_onehot[:, i]) > 0:
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
    plt.savefig(f'figure/VGG_32class_ROC_Curves.png')
    plt.close()

# 测试函数
def test_model(test_loader, model_path, scaled_model=None):
    model = VGG32Class(input_dim=features.shape[1], num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if scaled_model is not None:
        scaled_model.model.load_state_dict(torch.load(model_path))
        scaled_model.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            if scaled_model is not None:
                outputs = scaled_model(inputs)
            else:
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(all_labels, all_preds)
    n_classes_actual = len(np.unique(np.concatenate([all_labels, all_preds])))
    specificities = []
    for i in range(n_classes_actual):
        y_true_binary = (all_labels == i).astype(int)
        y_pred_binary = (all_preds == i).astype(int)
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
        specificities.append(specificity)
    specificity = np.mean(specificities) * 100
    macro_auc = 0.0
    weighted_auc = 0.0
    try:
        unique_classes = np.unique(all_labels)
        y_test_onehot = np.zeros((len(all_labels), n_classes))
        for i in range(len(all_labels)):
            if all_labels[i] < n_classes:
                y_test_onehot[i, all_labels[i]] = 1
        aucs = []
        for cls in unique_classes:
            if cls < n_classes:
                fpr, tpr, _ = roc_curve(y_test_onehot[:, cls], all_probs[:, cls])
                aucs.append(auc(fpr, tpr))
        if aucs:
            macro_auc = np.mean(aucs)
            class_weights = np.array([np.sum(all_labels == cls) for cls in unique_classes])
            weighted_auc = np.average(aucs, weights=class_weights)
    except Exception as e:
        print(f"计算AUC时出错: {e}")
    return f1, accuracy, precision, recall, specificity, macro_auc, weighted_auc, all_probs, all_labels

# 主函数
if __name__ == '__main__':
    if not os.path.exists('figure'):
        os.makedirs('figure')
    if not os.path.exists('vgg_models'):
        os.makedirs('vgg_models')
    start_time = time.time()
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)
    multiclass_labels = multilabel_to_multiclass(labels)
    unique_classes, counts = np.unique(multiclass_labels, return_counts=True)
    print("类别分布:")
    for cls, count in zip(unique_classes, counts):
        binary = format(cls, '05b')
        print(f"类别 {cls} (二进制: {binary}): {count}个样本")
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    mask = selector.get_support()
    all_feature_names = all_feature_names[mask]
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8
    features = (features - mean_f) / std_f
    print("PCA降维前，特征形状：", features.shape)
    pca = PCA(n_components=0.95)
    pca_selector = VarianceThreshold(threshold=0.01)
    features = pca_selector.fit_transform(pca.fit_transform(features))
    pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1] + 1)]
    print("PCA降维后，特征形状：", features.shape)
    folds_data_index = split_data_5fold(features)
    n_classes = 2**len(columns_to_encode)
    selected_folds_input = '1,2,3,4,5'
    if not selected_folds_input.strip():
        selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]
    all_fold_results = []
    log_filename = f'vgg_32class_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(log_filename, 'w') as f:
        f.write("fold,test_f1,test_accuracy,test_precision,test_recall,test_specificity,macro_auc,weighted_auc\n")
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_macro_auc = []
    all_weighted_auc = []
    for fold, (train_indices, val_indices, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue
        print(f"\n开始训练 Fold {fold+1}")
        X_train = features[train_indices]
        y_train = multiclass_labels[train_indices]
        X_val = features[val_indices]
        y_val = multiclass_labels[val_indices]
        X_test = features[test_indices]
        y_test = multiclass_labels[test_indices]
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        test_dataset = CustomDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = VGG32Class(input_dim=features.shape[1], num_classes=n_classes).to(device)
        scaled_model = TemperatureScaling(model).to(device)
        class_counts = np.bincount(y_train, minlength=n_classes)
        class_weights = 1.0 / np.sqrt(class_counts + 1)
        class_weights = class_weights / np.sum(class_weights) * n_classes
        class_weights = torch.FloatTensor(class_weights).to(device)
        print("类别权重:")
        for cls, weight in enumerate(class_weights.cpu().numpy()):
            if cls in unique_classes:
                binary = format(cls, '05b')
                count = class_counts[cls]
                print(f"类别 {cls} (二进制: {binary}): 样本数 {count}, 权重 {weight:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        num_epochs = 350
        patience = 20
        best_val_f1 = 0
        counter = 0
        best_epoch = 0
        train_losses = []
        val_losses = []
        val_f1_history = []
        val_accuracy_history = []
        val_precision_history = []
        val_recall_history = []
        for epoch in range(num_epochs):
            model.epoch = epoch
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                loss += 1e-5 * l1_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            val_accuracy = accuracy_score(all_labels, all_preds) * 100
            val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
            val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
            val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
            val_f1_history.append(val_f1)
            val_accuracy_history.append(val_accuracy)
            val_precision_history.append(val_precision)
            val_recall_history.append(val_recall)
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Val Precision: {val_precision:.2f}%, Val Recall: {val_recall:.2f}%')
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'当前学习率: {current_lr:.6f}')
            if epoch + 1 >= 300:
                cm = confusion_matrix(all_labels, all_preds)
                plot_confusion_matrix(cm, [f'Class_{i}' for i in range(n_classes)], fold, epoch)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                counter = 0
                best_epoch = epoch
                torch.save(model.state_dict(), f'vgg_models/best_model_fold{fold+1}.pt')
                torch.save(scaled_model.state_dict(), f'vgg_models/best_scaled_model_fold{fold+1}.pt')
            else:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {patience}')
                if counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with F1: {best_val_f1:.2f}%')
                    cm = confusion_matrix(all_labels, all_preds)
                    plot_confusion_matrix(cm, [f'Class_{i}' for i in range(n_classes)], fold, epoch)
                    break
            if epoch + 1 == 300:
                scaled_model.set_temperature(val_loader)
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curve (Fold {fold+1})')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(val_f1_history) + 1), val_f1_history, label='Validation F1', color='red')
        plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', color='blue')
        plt.plot(range(1, len(val_precision_history) + 1), val_precision_history, label='Validation Precision', color='green')
        plt.plot(range(1, len(val_recall_history) + 1), val_recall_history, label='Validation Recall', color='purple')
        plt.axvline(x=best_epoch+1, color='black', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
        plt.xlabel('Epochs')
        plt.ylabel('Score (%)')
        plt.title(f'Validation Metrics Curve (Fold {fold+1})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figure/vgg_32class_learning_curve_fold{fold+1}.png')
        plt.close()
        print(f"\n开始在测试集上评估 Fold {fold+1} 的模型")
        model.load_state_dict(torch.load(f'vgg_models/best_model_fold{fold+1}.pt'))
        model.eval()
        scaled_model.model.load_state_dict(torch.load(f'vgg_models/best_model_fold{fold+1}.pt'))
        scaled_model.model.eval()
        test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, all_probs, all_labels = test_model(test_loader, f'vgg_models/best_model_fold{fold+1}.pt')
        scaled_f1, scaled_accuracy, scaled_precision, scaled_recall, scaled_specificity, scaled_macro_auc, scaled_weighted_auc, _, _ = test_model(test_loader, f'vgg_models/best_model_fold{fold+1}.pt', scaled_model)
        plot_roc_curves(all_labels, all_probs, n_classes)
        fold_result = {
            'fold': fold + 1,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_specificity': test_specificity,
            'macro_auc': macro_auc,
            'weighted_auc': weighted_auc,
            'scaled_f1': scaled_f1,
            'scaled_accuracy': scaled_accuracy,
            'scaled_precision': scaled_precision,
            'scaled_recall': scaled_recall,
            'scaled_specificity': scaled_specificity,
            'scaled_macro_auc': scaled_macro_auc,
            'scaled_weighted_auc': scaled_weighted_auc
        }
        all_fold_results.append(fold_result)
        with open(log_filename, 'a') as f:
            f.write(f"{fold+1},{test_f1:.2f},{test_accuracy:.2f},{test_precision:.2f},{test_recall:.2f},{test_specificity:.2f},{macro_auc:.3f},{weighted_auc:.3f}\n")
            f.write(f"{fold+1}(scaled),{scaled_f1:.2f},{scaled_accuracy:.2f},{scaled_precision:.2f},{scaled_recall:.2f},{scaled_specificity:.2f},{scaled_macro_auc:.3f},{scaled_weighted_auc:.3f}\n")
        print(
            f'Fold [{fold + 1}/5]: '
            f'Test F1: {test_f1:.2f}% | '
            f'Accuracy: {test_accuracy:.2f}% | '
            f'Precision: {test_precision:.2f}% | '
            f'Recall: {test_recall:.2f}% | '
            f'Specificity: {test_specificity:.2f}% | '
            f'Macro AUC: {macro_auc:.3f} | '
            f'Weighted AUC: {weighted_auc:.3f}'
        )
        print(
            f'Fold [{fold + 1}/5] (温度缩放后): '
            f'Test F1: {scaled_f1:.2f}% | '
            f'Accuracy: {scaled_accuracy:.2f}% | '
            f'Precision: {scaled_precision:.2f}% | '
            f'Recall: {scaled_recall:.2f}% | '
            f'Specificity: {scaled_specificity:.2f}% | '
            f'Macro AUC: {scaled_macro_auc:.3f} | '
            f'Weighted AUC: {scaled_weighted_auc:.3f}'
        )
        with open('VGG_32class_Test.csv', 'a') as f:
            f.write(
                f'Fold [{fold + 1}/5]:, '
                f'Test F1: {test_f1:.2f}% , '
                f'Accuracy: {test_accuracy:.2f}% , '
                f'Precision: {test_precision:.2f}% , '
                f'Recall: {test_recall:.2f}% , '
                f'Specificity: {test_specificity:.2f}% , '
                f'Macro AUC: {macro_auc:.3f} , '
                f'Weighted AUC: {weighted_auc:.3f}\n'
            )
        with open('VGG_32class_scale_Test.csv', 'a') as f:
            f.write(
                f'Fold [{fold + 1}/5] (温度缩放后): ,'
                f'Test F1: {scaled_f1:.2f}% , '
                f'Accuracy: {scaled_accuracy:.2f}% , '
                f'Precision: {scaled_precision:.2f}% , '
                f'Recall: {scaled_recall:.2f}% , '
                f'Specificity: {scaled_specificity:.2f}% , '
                f'Macro AUC: {scaled_macro_auc:.3f} , '
                f'Weighted AUC: {scaled_weighted_auc:.3f}\n'
            )
        all_accuracies.append(test_accuracy)
        all_precisions.append(test_precision)
        all_recalls.append(test_recall)
        all_f1_scores.append(test_f1)
        all_macro_auc.append(macro_auc)
        all_weighted_auc.append(weighted_auc)
    if len(selected_folds) == 5:
        avg_accuracy = np.mean(all_accuracies)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1_scores)
        avg_macro_auc = np.mean(all_macro_auc)
        avg_weighted_auc = np.mean(all_weighted_auc)
        print("\n所有Fold的平均指标:")
        print(f"平均准确率: {avg_accuracy:.2f}%")
        print(f"平均精确率: {avg_precision:.2f}%")
        print(f"平均召回率: {avg_recall:.2f}%")
        print(f"平均F1分数: {avg_f1:.2f}%")
        print(f"平均宏平均AUC: {avg_macro_auc:.3f}")
        print(f"平均加权AUC: {avg_weighted_auc:.3f}")
        with open('VGG_32class_Test.csv', 'a') as f:
            f.write("\n所有Fold的平均指标:\n")
            f.write(f"平均准确率: {avg_accuracy:.2f}%\n")
            f.write(f"平均精确率: {avg_precision:.2f}%\n")
            f.write(f"平均召回率: {avg_recall:.2f}%\n")
            f.write(f"平均F1分数: {avg_f1:.2f}%\n")
            f.write(f"平均宏平均AUC: {avg_macro_auc:.3f}\n")
            f.write(f"平均加权AUC: {avg_weighted_auc:.3f}\n")
        plt.figure(figsize=(12, 8))
        x = np.arange(len(selected_folds))
        width = 0.15
        plt.bar(x - 2*width, all_accuracies, width, label='Accuracy')
        plt.bar(x - width, all_precisions, width, label='Precision')
        plt.bar(x, all_recalls, width, label='Recall')
        plt.bar(x + width, all_f1_scores, width, label='F1')
        plt.bar(x + 2*width, [auc*100 for auc in all_macro_auc], width, label='Macro AUC*100')
        plt.xlabel('Fold')
        plt.ylabel('Score (%)')
        plt.title('VGG 32-Class: Performance Metrics Across Folds')
        plt.xticks(x, [f'Fold {fold+1}' for fold in selected_folds])
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('figure/vgg_32class_performance_comparison.png')
        plt.close()
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(selected_folds) + 1), all_accuracies, label='Accuracy', marker='o')
        plt.plot(range(1, len(selected_folds) + 1), all_precisions, label='Precision', marker='s')
        plt.plot(range(1, len(selected_folds) + 1), all_recalls, label='Recall', marker='^')
        plt.plot(range(1, len(selected_folds) + 1), all_f1_scores, label='F1', marker='D')
        plt.plot(range(1, len(selected_folds) + 1), [auc*100 for auc in all_macro_auc], label='Macro AUC*100', marker='*')
        plt.xlabel('Fold')
        plt.ylabel('Score (%)')
        plt.title('VGG 32-Class: Performance Metrics Across Folds')
        plt.xticks(range(1, len(selected_folds) + 1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('figure/vgg_32class_performance_line.png')
        plt.close()
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nVGG 32类模型训练和评估完成！")
    print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    # 分层分析部分
    print("\n开始进行分层分析...")
    original_data = pd.read_excel('DR-CVD DataSet v1.2.xlsx', engine='openpyxl')
    stratify_variables = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']
    for fold in selected_folds:
        print(f"\n对Fold {fold+1}的模型进行分层分析")
        _, _, test_indices = folds_data_index[fold]
        model = VGG32Class(input_dim=features.shape[1], num_classes=n_classes).to(device)
        model.load_state_dict(torch.load(f'vgg_models/best_model_fold{fold+1}.pt'))
        model.eval()
        for stratify_var in stratify_variables:
            print(f"\n按 {stratify_var} 进行分层分析:")
            strata = original_data[stratify_var].unique()
            for stratum in strata:
                stratum_mask = original_data[stratify_var] == stratum
                stratum_indices = np.where(stratum_mask)[0]
                common_indices = np.intersect1d(stratum_indices, test_indices)
                if len(common_indices) > 0:
                    stratum_X = features[common_indices]
                    stratum_y = multiclass_labels[common_indices]
                    stratum_dataset = CustomDataset(stratum_X, stratum_y)
                    stratum_loader = DataLoader(stratum_dataset, batch_size=64, shuffle=False)
                    model.eval()
                    all_preds_stratum = []
                    all_labels_stratum = []
                    all_probs_stratum = []
                    with torch.no_grad():
                        for inputs, labels in stratum_loader:
                            inputs = inputs.to(device)
                            outputs = model(inputs)
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()
                            all_probs_stratum.append(probs)
                            _, preds = torch.max(outputs, 1)
                            all_preds_stratum.extend(preds.cpu().numpy())
                            all_labels_stratum.extend(labels.numpy())
                    all_preds_stratum = np.array(all_preds_stratum)
                    all_labels_stratum = np.array(all_labels_stratum)
                    all_probs_stratum = np.vstack(all_probs_stratum) if all_probs_stratum else np.array([])
                    if len(np.unique(all_labels_stratum)) > 1:
                        accuracy_stratum = accuracy_score(all_labels_stratum, all_preds_stratum) * 100
                        f1_stratum = f1_score(all_labels_stratum, all_preds_stratum, average='weighted', zero_division=0) * 100
                        precision_stratum = precision_score(all_labels_stratum, all_preds_stratum, average='weighted', zero_division=0) * 100
                        recall_stratum = recall_score(all_labels_stratum, all_preds_stratum, average='weighted', zero_division=0) * 100
                        n_classes_actual = len(np.unique(np.concatenate([all_labels_stratum, all_preds_stratum])))
                        specificities = []
                        for i in range(n_classes_actual):
                            y_true_binary = (all_labels_stratum == i).astype(int)
                            y_pred_binary = (all_preds_stratum == i).astype(int)
                            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                            spec = 0.0 if tn+fp==0 else tn/(tn+fp)
                            specificities.append(spec)
                        specificity_stratum = np.mean(specificities) * 100
                        try:
                            unique_cls = np.unique(all_labels_stratum)
                            y_test_onehot = np.zeros((len(all_labels_stratum), n_classes))
                            for i in range(len(all_labels_stratum)):
                                if all_labels_stratum[i] < n_classes:
                                    y_test_onehot[i, all_labels_stratum[i]] = 1
                            aucs = []
                            for cls in unique_cls:
                                if cls < n_classes and cls < all_probs_stratum.shape[1]:
                                    fpr, tpr, _ = roc_curve(y_test_onehot[:, cls], all_probs_stratum[:, cls])
                                    aucs.append(auc(fpr, tpr))
                            if aucs:
                                macro_auc_stratum = np.mean(aucs)
                                class_weights = np.array([np.sum(all_labels_stratum == cls) for cls in unique_cls])
                                weighted_auc_stratum = np.average(aucs, weights=class_weights)
                            else:
                                macro_auc_stratum = 0.0
                                weighted_auc_stratum = 0.0
                        except Exception as e:
                            print(f"计算AUC时出错: {e}")
                            macro_auc_stratum = 0.0
                            weighted_auc_stratum = 0.0
                        print(f"{stratify_var}={stratum} (样本数: {len(common_indices)}): F1: {f1_stratum:.2f}%, Accuracy: {accuracy_stratum:.2f}%, Precision: {precision_stratum:.2f}%, Recall: {recall_stratum:.2f}%, Specificity: {specificity_stratum:.2f}%, Macro AUC: {macro_auc_stratum:.3f}, Weighted AUC: {weighted_auc_stratum:.3f}")
                        cm_stratum = confusion_matrix(all_labels_stratum, all_preds_stratum)
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cm_stratum, interpolation='nearest', cmap='Blues')
                        plt.title(f'Confusion Matrix - {stratify_var}={stratum} (Fold {fold+1})')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(f'figure/vgg_32class_cm_{stratify_var}_{stratum}_fold{fold+1}.png')
                        plt.close()
                    else:
                        print(f"{stratify_var}={stratum} (样本数: {len(common_indices)}): 该层只有一个类别，无法计算指标")
                else:
                    print(f"{stratify_var}={stratum}: 测试集中没有该层的样本")
            # 绘制分层结果比较图
            # 此部分可根据需要进一步完善图表展示
        if fold in selected_folds:
            print(f"分层分析Fold {fold+1}完成。")
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('VGG_32class_all_results.csv', index=False)
    print("\n所有结果已保存到 VGG_32class_all_results.csv")
    plt.figure(figsize=(15, 8))
    # 可视化类别分布（这里展示前20个类别）
    sorted_indices = np.argsort(-counts)[:20]
    display_classes = unique_classes[sorted_indices]
    display_counts = counts[sorted_indices]
    plt.bar(range(len(display_classes)), display_counts)
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('32类分类问题的类别分布 (前20个类别)')
    labels_list = []
    for cls, cnt in zip(display_classes, display_counts):
        binary = format(cls, '05b')
        labels_list.append(f"{cls}\n({binary})\n{cnt}")
    plt.xticks(range(len(display_classes)), labels_list, rotation=45)
    plt.tight_layout()
    plt.savefig('figure/vgg_32class_class_distribution.png')
    plt.close()
    print("\n类别分布图已保存到 figure/vgg_32class_class_distribution.png")