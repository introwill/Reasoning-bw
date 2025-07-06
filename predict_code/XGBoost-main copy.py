import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import argmax
from sklearn.metrics import precision_recall_curve
from Model import mlsmote
import matplotlib
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, roc_curve, auc
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

matplotlib.use('TKAgg')  # 用于解决绘图时的报错问题
seed_value = 3407
np.random.seed(seed_value)

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


# 阈值计算函数
def Probs_Switcher(probs, labels):
    probs_Switcher = np.array([])
    
    # 修复：MultiOutputClassifier的predict_proba返回的是列表，每个元素对应一个输出
    # 原来的代码假设probs是一个形状为(n_samples, n_classes)的数组
    for i in range(labels.shape[1]):
        split_labels_np = labels[:, i]
        # 修复：从列表中获取对应类别的概率
        split_probs_np = probs[i][:, 1]  # 取正类的概率（索引1）
        precision, recall, thresholds = precision_recall_curve(split_labels_np, split_probs_np)
        precision = precision * 0.6
        recall = recall
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        index = argmax(f1_scores)
        if len(thresholds) > index:
            probs_Switcher = np.append(probs_Switcher, thresholds[index])
        else:
            # 如果索引超出范围，使用默认阈值0.5
            probs_Switcher = np.append(probs_Switcher, 0.5)

    return probs_Switcher


# 指标计算函数
def f1_score_func(probs, labels, probs_Switcher):
    # 修复：将MultiOutputClassifier的predict_proba输出转换为适合的格式
    # 创建一个与labels形状相同的预测数组
    preds = np.zeros_like(labels, dtype=np.float64)
    
    # 对每个输出应用阈值
    for i in range(labels.shape[1]):
        # 获取第i个分类器的正类概率
        class_probs = probs[i][:, 1]
        # 应用阈值
        preds[:, i] = (class_probs >= probs_Switcher[i]).astype(np.float64)

    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    
    # 计算特异性
    cm = multilabel_confusion_matrix(labels, preds)
    specificity = np.mean([cm[i, 0, 0] / (cm[i, 0, 0] + cm[i, 0, 1]) if (cm[i, 0, 0] + cm[i, 0, 1]) > 0 else 0 for i in range(len(cm))])

    return f1 * 100, accuracy * 100, precision * 100, recall * 100, specificity * 100, preds


# 测试函数
def test(X_test, y_test, model, probs_Switcher=None):
    # 预测概率
    probs = model.predict_proba(X_test)
    
    # 如果没有提供阈值，计算最优阈值
    if probs_Switcher is None:
        probs_Switcher = Probs_Switcher(probs, y_test)
    
    # 计算评估指标
    test_f1, test_accuracy, test_precision, test_recall, test_specificity, preds = f1_score_func(probs, y_test, probs_Switcher)
    
    # 计算AUC-ROC
    auc_scores = []
    for i in range(y_test.shape[1]):
        try:
            # 获取第i个分类器的正类概率
            class_probs = probs[i][:, 1]
            auc_score = roc_auc_score(y_test[:, i], class_probs)
            auc_scores.append(auc_score)
        except ValueError:
            print(f"Class {i} has only one class present in test set.")
            auc_scores.append(float('nan'))
    
    # 计算宏平均和加权平均AUC
    macro_auc = np.nanmean(auc_scores)
    
    # 修复：为weighted_auc创建适当格式的概率数组
    # 创建一个与y_test形状相同的概率数组
    probs_array = np.zeros_like(y_test, dtype=np.float64)
    for i in range(y_test.shape[1]):
        probs_array[:, i] = probs[i][:, 1]
    
    weighted_auc = roc_auc_score(y_test, probs_array, average='weighted', multi_class='ovr')
    
    return test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, probs, y_test


def plot_roc_curves(all_labels, all_probs, class_names):
    plt.figure(figsize=(10, 8))

    # 为每个类别绘制ROC曲线
    for i in range(len(class_names)):
        # 修复：从列表中获取对应类别的概率
        class_probs = all_probs[i][:, 1]
        fpr, tpr, _ = roc_curve(all_labels[:, i], class_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'figure/XGBoost_ROC_Curves.png')


# XGBoost模型训练函数
# XGBoost模型训练函数
def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    # 创建XGBoost模型
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1,
        scale_pos_weight=2.0,
        random_state=seed_value
    )
    
    # 使用MultiOutputClassifier包装XGBoost模型以处理多标签分类
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    
    # 训练模型
    if X_val is not None and y_val is not None:
        # 由于MultiOutputClassifier不支持eval_set等参数，我们需要直接训练模型
        # 不使用早停和验证集评估
        model.fit(X_train, y_train)
        
        # 如果需要早停和验证集评估，需要单独为每个分类器设置
        # 但这需要更复杂的实现，这里简化处理
    else:
        model.fit(X_train, y_train)
    
    return model


# SHAP分析函数
def analyze_feature_importance(model, X_test, feature_names, class_names, class_idx=0):
    # 获取特定类别的XGBoost模型
    xgb_model = model.estimators_[class_idx]
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    # 绘制SHAP汇总图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {class_names[class_idx]}')
    plt.tight_layout()
    plt.savefig(f'figure/XGBoost_SHAP_Summary_{class_names[class_idx]}.png')
    
    # 绘制SHAP条形图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance for {class_names[class_idx]}')
    plt.tight_layout()
    plt.savefig(f'figure/XGBoost_SHAP_Importance_{class_names[class_idx]}.png')
    
    return shap_values, explainer


if __name__ == '__main__':
    # columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    columns_to_encode = ['MCQ160C', 'MCQ160F']
    # columns_to_encode = ['MCQ160B', 'MCQ160D']

    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

    # 测试集的数据不要改定义！！！（一定要原始的数据集）
    features_val = features.copy()
    labels_val = labels.copy()

    Shap_features = features_val.copy()

    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)

    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)
    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)  # Getting minority instance of that datframe
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)  # Applying MLSMOTE to augment the dataframe

    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)

    # 添加方差过滤（阈值设为0.01）
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    features_val = selector.transform(features_val)

    # 更新特征名称
    mask = selector.get_support()
    all_feature_names = all_feature_names[mask]

    # 标准化数据
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

    # 加入PCA降维，保留95%的信息
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

    folds_data_index = split_data_5fold(features)

    num_x = len(Covariates_features)
    num_folds = len(folds_data_index)

    # 用于保存不同x的指标
    all_accuracies = [[] for _ in range(num_x)]
    all_precisions = [[] for _ in range(num_x)]
    all_recalls = [[] for _ in range(num_x)]
    all_specificities = [[] for _ in range(num_x)]
    all_f1_scores = [[] for _ in range(num_x)]
    all_macro_auc = [[] for _ in range(num_x)]
    all_weighted_auc = [[] for _ in range(num_x)]

    # 让用户输入想要运行的fold编号，用逗号分隔
    # selected_folds_input = input("请输入想要运行的fold编号（用逗号分隔，例如：1,3,5）：")
    # 为了调试方便，这里写死selected_folds_input为1,2,3,4,5
    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    # 用于存储所有折叠的结果
    all_test_results = []
    all_probs_switchers = []
    all_models = []

    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  # 跳过未选择的fold

        print(f"\n开始处理第 {fold + 1} 折...")
        
        # 准备数据
        trainX = features[train_index]
        trainY = labels[train_index]
        
        # 修复：使用与features相同的索引来划分features_val
        # 原来的代码：valX = features_val[validation_index]
        # 问题：validation_index可能超出features_val的范围
        
        # 解决方案：创建新的交叉验证索引，专门用于features_val
        folds_val_index = split_data_5fold(features_val)
        _, val_index, test_index = folds_val_index[fold]
        
        valX = features_val[val_index]
        valY = labels_val[val_index]
        testX = features_val[test_index]
        testY = labels_val[test_index]
        
        # 训练XGBoost模型
        print("训练XGBoost模型...")
        model = train_xgboost(trainX, trainY, valX, valY)
        
        # 保存模型
        model_path = f'single/xgboost_model_fold_{fold}.pkl'
        import pickle
        import os
        
        # 确保目录存在
        os.makedirs('single', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到 {model_path}")
        
        # 在验证集上计算最优阈值
        print("在验证集上计算最优阈值...")
        val_probs = model.predict_proba(valX)
        probs_switcher = Probs_Switcher(val_probs, valY)
        all_probs_switchers.append(probs_switcher)
        
        # 在测试集上评估模型
        print("在测试集上评估模型...")
        test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels = test(testX, testY, model, probs_switcher)
        
        # 保存测试结果
        test_result = {
            'fold': fold + 1,
            'f1': test_f1,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'specificity': test_specificity,
            'macro_auc': macro_auc,
            'weighted_auc': weighted_auc,
            'auc_scores': auc_scores,
            'probs': all_probs,
            'labels': all_labels
        }
        all_test_results.append(test_result)
        all_models.append(model)
        
        # 打印测试结果
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
        
        # 将结果保存到CSV文件
        with open('xgboost_result.csv', 'a') as f:
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
        
        # 绘制ROC曲线
        plot_roc_curves(all_labels, all_probs, columns_to_encode)
        
        # 保存当前fold的指标
        all_accuracies[0].append(test_accuracy)
        all_precisions[0].append(test_precision)
        all_recalls[0].append(test_recall)
        all_f1_scores[0].append(test_f1)
        all_macro_auc[0].append(macro_auc)
        all_weighted_auc[0].append(weighted_auc)

    # 计算平均指标
    if len(all_test_results) > 0:
        avg_f1 = np.mean([result['f1'] for result in all_test_results])
        avg_accuracy = np.mean([result['accuracy'] for result in all_test_results])
        avg_precision = np.mean([result['precision'] for result in all_test_results])
        avg_recall = np.mean([result['recall'] for result in all_test_results])
        avg_specificity = np.mean([result['specificity'] for result in all_test_results])
        avg_macro_auc = np.mean([result['macro_auc'] for result in all_test_results])
        avg_weighted_auc = np.mean([result['weighted_auc'] for result in all_test_results])
        
        print("\n所有折叠的平均指标:")
        print(f"平均 F1: {avg_f1:.2f}%")
        print(f"平均准确率: {avg_accuracy:.2f}%")
        print(f"平均精确率: {avg_precision:.2f}%")
        print(f"平均召回率: {avg_recall:.2f}%")
        print(f"平均特异性: {avg_specificity:.2f}%")
        print(f"平均宏观AUC: {avg_macro_auc:.3f}")
        print(f"平均加权AUC: {avg_weighted_auc:.3f}")
        
        # 将平均指标保存到CSV文件
        with open('xgboost_result.csv', 'a') as f:
            f.write("\n所有折叠的平均指标:\n")
            f.write(f"平均 F1: {avg_f1:.2f}%\n")
            f.write(f"平均准确率: {avg_accuracy:.2f}%\n")
            f.write(f"平均精确率: {avg_precision:.2f}%\n")
            f.write(f"平均召回率: {avg_recall:.2f}%\n")
            f.write(f"平均特异性: {avg_specificity:.2f}%\n")
            f.write(f"平均宏观AUC: {avg_macro_auc:.3f}\n")
            f.write(f"平均加权AUC: {avg_weighted_auc:.3f}\n")
    
    # 如果运行了所有折叠，进行特征重要性分析
    if len(selected_folds) == 5:
        # 选择最佳模型进行特征重要性分析
        best_model_idx = np.argmax([result['f1'] for result in all_test_results])
        best_model = all_models[best_model_idx]
        best_fold = selected_folds[best_model_idx] + 1
        
        print(f"\n使用第 {best_fold} 折的最佳模型进行特征重要性分析...")
        
        # 对每个类别进行特征重要性分析
        for class_idx in range(len(columns_to_encode)):
            print(f"\n分析类别 {columns_to_encode[class_idx]} 的特征重要性...")
            
            # 获取该类别的XGBoost模型
            xgb_model = best_model.estimators_[class_idx]
            
            # 获取特征重要性
            importance_type = 'gain'  # 可以是 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
            feature_importance = xgb_model.get_booster().get_score(importance_type=importance_type)
            
            # 转换为DataFrame并排序
            importance_df = pd.DataFrame({
                'feature': list(feature_importance.keys()),
                'importance': list(feature_importance.values())
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 打印前20个重要特征
            print(f"类别 {columns_to_encode[class_idx]} 的前20个重要特征:")
            print(importance_df.head(20))
            
            # 绘制特征重要性图
            plt.figure(figsize=(12, 8))
            top_n = min(20, len(importance_df))
            plt.barh(importance_df['feature'].head(top_n), importance_df['importance'].head(top_n))
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Feature Importance for {columns_to_encode[class_idx]}')
            plt.tight_layout()
            plt.savefig(f'figure/XGBoost_Feature_Importance_{columns_to_encode[class_idx]}.png')
            
            # 使用SHAP分析特征重要性
            analyze_feature_importance(best_model, testX, pca_feature_names, columns_to_encode, class_idx)
    
    # 分层分析
    print("\n开始分层分析...")
    stratify_variable = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']  # 分别是患病程度，性别，种族
    
    # 选择最佳模型进行分层分析
    if len(all_models) > 0:
        best_model_idx = np.argmax([result['f1'] for result in all_test_results])
        best_model = all_models[best_model_idx]
        best_probs_switcher = all_probs_switchers[best_model_idx]
        
        for strat_var in stratify_variable:
            print(f"\n按 {strat_var} 进行分层分析...")
            
            # 获取分层变量的不同取值
            strata = Multilay_origin[strat_var].unique()
            
            for stratum in strata:
                # 筛选出当前层的数据
                stratum_mask = Multilay_origin[strat_var] == stratum
                stratum_indices = np.where(stratum_mask)[0]
                
                # 确保有足够的样本
                if len(stratum_indices) < 10:
                    print(f"  {strat_var} = {stratum} 的样本数量不足，跳过")
                    continue
                
                # 获取该层的特征和标签
                stratum_features = features_val[stratum_indices]
                stratum_labels = labels_val[stratum_indices]
                
                # 在该层上评估模型
                print(f"  评估 {strat_var} = {stratum} 的模型性能...")
                strat_f1, strat_accuracy, strat_precision, strat_recall, strat_specificity, strat_macro_auc, strat_weighted_auc, strat_auc_scores, strat_probs, strat_labels = test(
                    stratum_features, stratum_labels, best_model, best_probs_switcher
                )
                
                # 打印该层的评估结果
                print(
                    f'  {strat_var} = {stratum}: '
                    f'F1: {strat_f1:.2f}% | '
                    f'Accuracy: {strat_accuracy:.2f}% | '
                    f'Precision: {strat_precision:.2f}% | '
                    f'Recall: {strat_recall:.2f}% | '
                    f'Specificity: {strat_specificity:.2f}% | '
                    f'Macro AUC: {strat_macro_auc:.3f} | '
                    f'Weighted AUC: {strat_weighted_auc:.3f}'
                )
                
                # 将结果保存到CSV文件
                with open('xgboost_stratified_result.csv', 'a') as f:
                    f.write(
                        f'{strat_var} = {stratum}: '
                        f'F1: {strat_f1:.2f}% | '
                        f'Accuracy: {strat_accuracy:.2f}% | '
                        f'Precision: {strat_precision:.2f}% | '
                        f'Recall: {strat_recall:.2f}% | '
                        f'Specificity: {strat_specificity:.2f}% | '
                        f'Macro AUC: {strat_macro_auc:.3f} | '
                        f'Weighted AUC: {strat_weighted_auc:.3f}\n'
                    )
    
    print("\nXGBoost模型训练和评估完成！")