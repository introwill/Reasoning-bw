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
import shap
from sklearn.metrics import roc_auc_score, roc_curve, auc
import statsmodels.formula.api as smf

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

class NetDataset(Dataset):
    def __init__(self, features, labels):
        self.Data = features
        self.label = labels

    def __getitem__(self, index):
        return self.Data[index], self.label[index]

    def __len__(self):
        return len(self.Data)

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


def test(Test_data_loader, probs_Switcher, Saved_Model):
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
        for data, target in Test_data_loader:
            n += 1
            data, target = data.float().to(device), target.float().to(device)
            output = model(data, epoch)

            # 收集概率和标签
            probs_sigmoid = torch.sigmoid(output)
            all_probs.append(probs_sigmoid.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            test_f1, test_accuracy, test_precision, test_recall, preds = f1_score_func(output, target, probs_Switcher)

            Test_F1 += test_f1
            Test_accuracy += test_accuracy
            Test_precision += test_precision
            Test_recall += test_recall

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

        return Test_F1, Test_accuracy, Test_precision, Test_recall, macro_auc, weighted_auc, auc_scores, all_probs, all_labels

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

def f1_score_func(logits, labels, probs_Switcher):
    probs_sigmoid = torch.sigmoid(logits)
    probs = probs_sigmoid.detach().cpu().numpy()
    labels = labels.cpu().detach().numpy()

    preds = np.float64(probs >= probs_Switcher)

    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    return f1 * 100, accuracy * 100, precision * 100, recall * 100, preds

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

# 自定义预测函数
def predict(data, probs_Switcher):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(data, epoch)
        probs_sigmoid = torch.sigmoid(output)
        probs = probs_sigmoid.cpu().detach().numpy()
        preds = np.float64(probs >= probs_Switcher)
        return preds

def Shap_predict(X):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X, epoch)
        probs_sigmoid = torch.sigmoid(output)
        probs = probs_sigmoid.cpu().detach()
    return probs[:, class_idx]

def model_wrapper(X):
    output = Shap_model.forward(X)
    if isinstance(output, torch.Tensor):
        output = output.cpu().detach().numpy()
    return output[:, class_idx]

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
    plt.show()

def Filtered_Data(df):
    youth_df = df[df['RIDAGEYR'] < 59]
    No_youth_df = df[df['RIDAGEYR'] >= 59]

    return youth_df, No_youth_df

def Calcluate_mean_std(df, columns):
    Data = df[columns]
    mean, std = np.mean(Data, axis=0), np.std(Data, axis=0)
    # mean_std = ', '.join([f'{m:.2f}±{s:.2f}' for m, s in zip(mean, std)])
    output = {}
    for col, m, s in zip(columns, mean, std):
        output[col] = f'{m:.2f}({s:.2f})'
    return output

def Calcluate_percentage(df, columns):
    output = {}
    for i in columns:
        output[i] = {}
        # 获取分层变量的不同取值
        Strata = df[i].unique()
        total_count = len(df)
        for x in Strata:
            x = int(x)
            stratum_mask = df[i] == x
            stratum_data = df[stratum_mask].copy()
            number = len(stratum_data)
            percentage = (number / total_count) * 100
            output[i][x] = f'{number}({percentage:.2f}%)'
    return output


if __name__ == '__main__':
    # columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    columns_to_encode = ['MCQ160C', 'MCQ160F']
    # columns_to_encode = ['MCQ160B', 'MCQ160D']

    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)

    #基线特征
    Columns_mean = ['RIDAGEYR', 'BMXWT', 'BMXHT']
    Columns_percentage = ['RIAGENDR', 'RIDRETH1', 'OPDURL4']

    Youth_DF, No_Youth_DF = Filtered_Data(data_DF)
    Total_DF = [Youth_DF, No_Youth_DF]
    Output_DF = {}
    Index = ['Youth', 'No_Youth']

    for index, df in enumerate(Total_DF):
        Output = Calcluate_mean_std(df, Columns_mean)
        percentage = Calcluate_percentage(df, Columns_percentage)

        Output.update(percentage)

        Output_DF[Index[index]] = Output
    Output_DataFrame = pd.DataFrame(Output_DF)
    print(Output_DataFrame)

    # 测试集的数据不要改定义！！！（一定要原始的数据集）
    features_val = features
    labels_val = labels

    Shap_features = features_val

    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)

    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)  # Getting minority instance of that datframe
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)  # Applying MLSMOTE to augment the dataframe

    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)

    mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    mean_val, std_val = np.mean(features_val, axis=0), np.std(features_val, axis=0)
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1e-8  # 将标准差为零的值设置为一个很小的数，避免除以零
    features = (features - mean) / std
    features_val = (features_val - mean_val) / (std_val + 1e-8)
    features = features.reshape(features.shape[0], -1)

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
    selected_folds_input = input("请输入想要运行的fold编号（用逗号分隔，例如：1,3,5）：")
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  # 跳过未选择的fold

        l1_weight = 0.070
        num_classes = len(columns_to_encode)  # Adjust the number of classes
        num_epochs = 305
        batch_size = 256
        input_length = batch_size
        for x in range(len(Covariates_features)):

            model = Model_VGG.VGG11(num_classes=num_classes, in_channels=len(all_feature_names)-len(Covariates_features), Covariates_features_length=x)

            # 温度缩放模型(没用)
            scaled_model = ModelWithTemperature(model)

            pos_weight = torch.tensor(2.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            optimizer = optim.Adam(scaled_model.parameters(), lr=0.00425)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.475)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            scaled_model.to(device)

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
            all_preds = np.empty((0, 2))
            all_targets = np.empty((0, 2))

            valid_f1_scores = []
            valid_precision_scores = []
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(Train_data_loader):
                    data, target = data.float().to(device), target.float().to(device)
                    optimizer.zero_grad()
                    output = model(data, epoch)
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

                model.eval()
                n = 0
                valid_f1 = 0
                valid_accuracy = 0
                valid_precision = 0
                valid_recall = 0

                with torch.no_grad():
                    for data, target in Validation_data_loader:
                        n += 1
                        data, target = data.float().to(device), target.float().to(device)
                        output = model(data, epoch)
                        probs_Switcher = Probs_Switcher(output, target)
                        f1, accuracy, precision, recall, preds = f1_score_func(output, target, probs_Switcher)

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

                valid_f1_scores.append(valid_f1)
                valid_precision_scores.append(valid_precision)
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}]: Average Train Loss: {average_train_loss:.4f} '
                    f'Validation F1: {valid_f1:.2f}%, Validation Accuracy: {valid_accuracy:.2f}%, Validation Precision: {valid_precision:.2f}%, Validation Recall: {valid_recall:.2f}%')

                # 在第300次迭代后绘制混淆矩阵
                if epoch + 1 >= 300:
                    cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
                    cm = multilabel_confusion_matrix(target.cpu().numpy(), preds)
                    print(cm_sum)
                    print(cm)

            torch.save(model.state_dict(), f'single/model_{fold}_{x}.ckpt')

            # 测试集
            Test_F1, Test_accuracy, Test_precision, Test_recall, macro_auc, weighted_auc, auc_scores, all_probs, all_labels = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt')
            # plot_roc_curves(all_labels, all_probs, columns_to_encode)
            print(
                f'fold [{fold + 1}/5]: '
                f'Test F1: {Test_F1:.2f}% | '
                f'Accuracy: {Test_accuracy:.2f}% | '
                f'Precision: {Test_precision:.2f}% | '
                f'Recall: {Test_recall:.2f}% | '
                f'Macro AUC: {macro_auc:.3f} | '
                f'Weighted AUC: {weighted_auc:.3f}'
            )

            # 保存当前 x 和 fold 的指标
            all_accuracies[x].append(Test_accuracy)
            all_precisions[x].append(Test_precision)
            all_recalls[x].append(Test_recall)
            all_f1_scores[x].append(Test_F1)
            all_macro_auc[x].append(macro_auc)
            all_weighted_auc[x].append(weighted_auc)

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
        plt.show()

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
        plt.show()

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
        plt.show()

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
        plt.show()

    # 特征重要性排序，使用SHAP Value说明特征的重要程度，然后基于特征置换的特征重要性评估方法，也可以看作是一种特征消融实验方法，说明特征的重要性。

    origin = pd.DataFrame(features_val, columns=all_feature_names)
    COLUMNS = origin.columns
    num_classes = labels_val.shape[1]  # 获取类别数

    # 使用 shap.sample 对背景数据进行降采样
    K = 16  # 可以根据实际情况调整 K 的值
    background_data = shap.kmeans(Shap_features, K)

    selected_indices = np.random.choice(len(Shap_features), 3501, replace=True)
    selected_features_val = Shap_features[selected_indices]

    ALL_shap_exp = {}
    ALL_top_features = {}

    Shap_model = Model_VGG.Trained_VGG11(model, epoch, mean_val, std_val, device).eval()

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

        # 绘制 SHAP 摘要图
        shap.plots.beeswarm(shap_exp, max_display=16)

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
        plt.show()

        # 选取 SHAP 高重要性特征（这里以绝对值的均值排序取前 209 个为例）
        shap_importance = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(shap_importance)[::-1]
        top_209_features = np.array(all_feature_names)[sorted_indices[:209]]
        ALL_top_features[columns_to_encode[class_idx]] = top_209_features

    #分层分析，对患病程度，性别，种族进行分层分析。目前用多因素线性回归模型，说明不同层次人群特征的线性相关程度，但部分特征无法推导线性相关程度。
    # 使用SHAP图可以补充说明非线性相关程度特征如何影响判断结果（非线性分析？），目前有瀑布图（一类人群的shap value对模型结果影响的具象化），force图（个例shap value对结果的影响）
    # 参考图片都在文件夹里

    stratify_variable = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']#分别是患病程度，性别，种族

    # 创建一个空的 DataFrame 用于保存结果
    results_df = pd.DataFrame(columns=['类别', '分层变量', '分层值', '特征', '系数', 'p 值', '置信区间下限', '置信区间上限', '相关性描述'])

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

                # 对特征数据进行对数变换
                for feature in top_209_features:
                    if np.all(regression_data[feature] > 0):  # 确保数据都为正，才能进行对数变换
                        regression_data[feature] = np.log(regression_data[feature])

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
                                    description = f"{feature} 与疾病标签呈正相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率增加 {coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。"
                                else:
                                    print(
                                        f"{feature} 与疾病标签呈负相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率降低 {-coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。")
                                    description = f"{feature} 与疾病标签呈负相关。具体而言，{feature} 每增加一个单位，{columns_to_encode[class_idx]}患病概率降低 {-coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% 至 {conf_int[1] * 100:.2f}%)。"
                            else:
                                print(f"{feature} 与疾病标签之间未发现显著的线性关系 (p = {p_value:.4f})。")
                                description = f"{feature} 与疾病标签之间未发现显著的线性关系 (p = {p_value:.4f})。"

                                # 将结果添加到 DataFrame 中
                                new_row = {
                                    '类别': columns_to_encode[class_idx],
                                    '分层变量': i,
                                    '分层值': stratum,
                                    '特征': feature,
                                    '系数': coef,
                                    'p 值': p_value,
                                    '置信区间下限': conf_int[0],
                                    '置信区间上限': conf_int[1],
                                    '相关性描述': description
                                }
                                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

                    except Exception as e:
                        print(f"在类别 {columns_to_encode[class_idx]}，{i} 为 {stratum} 层时，线性回归模型拟合失败，错误信息：{e}")

    results_df.to_csv('stratified_analysis_results.csv', index=False)


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
