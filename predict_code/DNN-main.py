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
import seaborn as sns  # Add seaborn library for better-looking confusion matrix plots
from sklearn.metrics import roc_auc_score, roc_curve, auc
import statsmodels.formula.api as smf
import os

matplotlib.use('TKAgg')  # Used to solve plotting errors, e.g. python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
seed_value = 3407
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

 # Dataset processing
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

 # Custom dataset class
class NetDataset(Dataset):
    def __init__(self, features, labels):
        self.Data = features
        self.label = labels

    def __getitem__(self, index):
        return self.Data[index], self.label[index]

    def __len__(self):
        return len(self.Data)


 # Replace VGG model with DNN model
class DNNModel(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(DNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

 # Data normalization and cross-validation split
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

 # Add function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, fold, epoch, is_sum=False):
    """
    Plot normalized confusion matrix image
    
    Args:
        cm -- confusion matrix, shape (n_classes, 2, 2)
        class_names -- list of class names
        fold -- current fold number
        epoch -- current epoch number
        is_sum -- whether it is a cumulative confusion matrix
    """
    n_classes = len(class_names)
    
    # Extract data from original predictions and labels
    if is_sum:
        # Use all_targets and all_preds
        y_true = all_targets
        y_pred = all_preds
    else:
        # Use current batch's target and preds
        y_true = target.cpu().numpy()
        y_pred = preds
    
    # Convert multilabel to multiclass encoding
    # For example, [0,1,0,0,1] to binary "01001", then to decimal 9
    y_true_classes = np.zeros(len(y_true), dtype=int)
    y_pred_classes = np.zeros(len(y_pred), dtype=int)
    
    for i in range(len(y_true)):
        # Check for "no class" case
        if np.sum(y_true[i]) == 0:
            y_true_classes[i] = 2**n_classes  # Use an extra code to represent "no class"
        else:
            true_str = ''.join(map(str, y_true[i].astype(int)))
            y_true_classes[i] = int(true_str, 2)
        
        if np.sum(y_pred[i]) == 0:
            y_pred_classes[i] = 2**n_classes  # Use an extra code to represent "no class"
        else:
            pred_str = ''.join(map(str, y_pred[i].astype(int)))
            y_pred_classes[i] = int(pred_str, 2)
    
    # Find actually occurring classes
    unique_classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
    n_unique = len(unique_classes)
    
    # Create confusion matrix
    conf_matrix = np.zeros((n_unique, n_unique))
    for i in range(len(y_true_classes)):
        true_idx = np.where(unique_classes == y_true_classes[i])[0][0]
        pred_idx = np.where(unique_classes == y_pred_classes[i])[0][0]
        conf_matrix[true_idx, pred_idx] += 1
    
    # Normalize confusion matrix, handle division by zero
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero, replace zeros with 1
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm_conf_matrix = conf_matrix / row_sums
    
    # Ensure figure directory exists
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    # Prepare labels, add special label for "no class" case
    xticklabels = []
    yticklabels = []
    for c in unique_classes:
        if c == 2**n_classes:
            xticklabels.append("None")
            yticklabels.append("None")
        else:
            xticklabels.append(bin(c)[2:].zfill(n_classes))
            yticklabels.append(bin(c)[2:].zfill(n_classes))
    
    # Use seaborn to plot heatmap
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=xticklabels,
               yticklabels=yticklabels)
    
    plt.title(f'Normalized Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add class name annotation
    plt.figtext(0.5, 0.01, f'Classes: {", ".join(class_names)} + None', ha='center')
    
    plt.tight_layout()
    
    # Save image
    if is_sum:
        plt.savefig(f'figure/confusion_matrix_fold{fold+1}_sum.png')
    else:
        plt.savefig(f'figure/confusion_matrix_fold{fold+1}_epoch{epoch+1}.png')
    
    plt.close()
 # Test function
def test(Test_data_loader, probs_Switcher, Saved_Model, Scaled_Model):
    model.load_state_dict(torch.load(Saved_Model, map_location=device, weights_only=False))
    model.eval()

    n = 0

    # Initialize variables to store probabilities and labels
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

            # Collect probabilities and labels
            probs_sigmoid = torch.sigmoid(output)
            all_probs.append(probs_sigmoid.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(output, target, probs_Switcher)

            Test_F1 += test_f1
            Test_accuracy += test_accuracy
            Test_precision += test_precision
            Test_recall += test_recall
            Test_specificity += test_specificity

        # Calculate AUC-ROC
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate AUC for each class
        auc_scores = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_scores.append(auc)
            except ValueError:
                print(f"Class {i} has only one class present in test set.")
                auc_scores.append(float('nan'))

        # Calculate macro and weighted average AUC
        macro_auc = np.nanmean(auc_scores)
        weighted_auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')

        Test_F1 /= n
        Test_accuracy /= n
        Test_precision /= n
        Test_recall /= n
        Test_specificity /= n

        return Test_F1, Test_accuracy, Test_precision, Test_recall, Test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels

 # Threshold calculation function
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

 # Metric calculation function
def f1_score_func(logits, labels, probs_Switcher):
    probs_sigmoid = torch.sigmoid(logits)
    probs = probs_sigmoid.detach().cpu().numpy()
    labels = labels.cpu().detach().numpy()

    preds = np.float64(probs >= probs_Switcher)

    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Specificity calculation
    cm = multilabel_confusion_matrix(labels, preds)
    specificities = []
    for i in range(cm.shape[0]):
        tn, fp = cm[i][0, 0], cm[i][0, 1]
        if (tn + fp) == 0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
        specificities.append(specificity)
    specificity = np.mean(specificities) * 100  # Use macro average

    return f1 * 100, accuracy * 100, precision * 100, recall * 100, preds, specificity

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

 # Custom prediction function
def predict(data, probs_Switcher):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(data)
        probs_sigmoid = torch.sigmoid(output)
        probs = probs_sigmoid.cpu().detach().numpy()
        preds = np.float64(probs >= probs_Switcher)
        return preds

 # SHAP analysis
def Shap_predict(X):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X)
        probs_sigmoid = torch.sigmoid(output)
        probs = probs_sigmoid.cpu().detach()
    return probs[:, class_idx]

def model_wrapper(X):   # This function wraps the model for SHAP library calls
    output = Shap_model.forward(X)
    if isinstance(output, torch.Tensor):
        output = output.cpu().detach().numpy()
    return output[:, class_idx]

def global_shap_analysis(model, background_data, test_data, feature_names, class_names, output_dir='figure/global_shap'):
    """
    Perform global SHAP analysis and generate visualization plots
    
    Args:
        model -- trained model
        background_data -- background data for SHAP explainer
        test_data -- test data for generating SHAP values
        feature_names -- list of feature names
        class_names -- list of class names
        output_dir -- output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get device of the model
    device = next(model.parameters()).device

    # Check feature name length
    if len(feature_names) != test_data.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]
        print(f"Warning: Auto-generated feature names, length {len(feature_names)}")
    
    # Create explainer, ensure data and model are on the same device
    explainer = shap.KernelExplainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().detach().numpy(),
        background_data
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(test_data, nsamples=500)
    # Print shape and type info
    print("shap_values 类型:", type(shap_values))
    
    # Check if shap_values is a list or array
    if isinstance(shap_values, list):
        print("shap_values 是列表，长度:", len(shap_values))
        if len(shap_values) > 0:
            print("第一个元素形状:", np.array(shap_values[0]).shape)
    else:
        print("shap_values 形状:", shap_values.shape)
    
    # Store SHAP values for all classes
    all_shap_values = []
    
    # Process SHAP values for each class
    for i, class_name in enumerate(class_names):
        print(f"Calculating global SHAP values for class {class_name} ...")
        
        # Get SHAP values for the corresponding class based on type
        if isinstance(shap_values, list):
            # If list, each element corresponds to a class
            if i < len(shap_values):
                class_shap_values = shap_values[i]
            else:
                print(f"Warning: Class index {i} exceeds shap_values list length {len(shap_values)}")
                continue
        else:
            # If array, assume shape is (samples, features, classes)
            class_shap_values = shap_values[:, :, i]
            
        print(f"SHAP value shape for class {class_name}:", np.array(class_shap_values).shape)
        all_shap_values.append(class_shap_values)
        
        # Create SHAP explanation object
        shap_exp = shap.Explanation(
            values=class_shap_values,
            data=test_data,
            feature_names=feature_names
        )

        # Plot bar chart - use English naming
        plt.figure(figsize=(10, 8))
        try:
            # Calculate feature importance (mean absolute SHAP value)
            feature_importance = np.abs(np.array(class_shap_values)).mean(0)
            # Get sorted indices
            sorted_idx = np.argsort(feature_importance)
            # Select most important features
            top_features = min(20, len(feature_names))
            plt.barh(range(top_features), feature_importance[sorted_idx[-top_features:]])
            plt.yticks(range(top_features), [feature_names[i] for i in sorted_idx[-top_features:]])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Feature Importance for Category {class_name}')
        except Exception as e:
            print(f"Error when plotting bar chart: {e}")
            try:
                shap.summary_plot(class_shap_values, test_data, feature_names=feature_names,
                                  plot_type="bar", show=False, max_display=20)
            except Exception as e2:
                print(f"Failed to plot bar chart with shap.summary_plot: {e2}")
        
        plt.tight_layout()
        # Save file with English naming
        plt.savefig(f'{output_dir}/global_bar_plot_category_{class_name}.png')
        plt.close()
        
        # Plot summary chart - use English naming
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(class_shap_values, test_data, feature_names=feature_names, show=False, max_display=20)
            plt.title(f'SHAP Summary Plot for Category {class_name}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/global_summary_plot_category_{class_name}.png')
        except Exception as e:
            print(f"Error when plotting summary chart: {e}")
        plt.close()

    # Plot polynomial-SHAP plot of the data.
    # Combine SHAP values into 3D array, handle both list and array cases
    if isinstance(shap_values, list):
        # Ensure all elements are arrays and have the same shape
        all_arrays = [np.array(sv) for sv in all_shap_values]
        if all(arr.shape == all_arrays[0].shape for arr in all_arrays):
            shap_3d = np.stack(all_arrays, axis=2)
        else:
            print("Warning: SHAP value shapes for different classes are inconsistent, skipping polynomial plot")
            return all_shap_values
    else:
        shap_3d = shap_values
    
    # Calculate mean absolute SHAP value for each feature in each class
    mean_abs_shap = np.abs(shap_3d).mean(axis=0)  # 形状：(特征数, 类别数)
    agg_shap_df = pd.DataFrame(mean_abs_shap, columns=class_names, index=feature_names)

    # Sort by total feature importance (simulate example plot feature order)
    feature_order = agg_shap_df.sum(axis=1).sort_values(ascending=False).index
    agg_shap_df = agg_shap_df.loc[feature_order]

    plt.figure(figsize=(18, 8))
    bottom = np.zeros(len(agg_shap_df))
    colors = sns.color_palette("tab10", len(class_names))  # Generate colors for each class

    for i, disease in enumerate(class_names):
        plt.bar(
            agg_shap_df.index,
            agg_shap_df[disease],
            bottom=bottom,
            label=disease,
            color=colors[i],
            edgecolor="black",  # Show bar border
            linewidth=0.5
        )
        bottom += agg_shap_df[disease]

    plt.xlabel("Top Most Important Features in Predicting Liver Disease", fontsize=12)
    plt.ylabel("mean(|SHAP value|) / average impact on model output magnitude", fontsize=12)
    plt.title("Polynomial-SHAP plot of the data", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate and right-align feature labels
    plt.legend(
        title="",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
        frameon=False  # Do not show legend border
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/polynomial_shap_plot.png")
    plt.show()

    return all_shap_values

 # Add global SHAP explanation function
def map_pca_shap_to_original_features(shap_values, pca_model, feature_names, class_names, output_dir='figure/original_feature_shap'):
    """
    Map SHAP values of PCA features back to original feature space
    
    Args:
        shap_values -- list of SHAP values for PCA features
        pca_model -- trained PCA model
        feature_names -- list of original feature names
        class_names -- list of class names
        output_dir -- output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get PCA components
    components = pca_model.components_
    
    # Process each class
    for i, class_name in enumerate(class_names):
        print(f"Mapping SHAP values of class {class_name} back to original feature space...")
        
        # Get SHAP values for current class
        if isinstance(shap_values, list):
            class_shap_values = shap_values[i]
        else:
            class_shap_values = shap_values
        
        # Calculate importance of original features
        # Dot product of SHAP values and PCA components
        original_importance = np.zeros(len(feature_names))
        
        # For each sample's SHAP value
        for sample_idx in range(class_shap_values.shape[0]):
            # For each PCA feature
            for pca_idx in range(class_shap_values.shape[1]):
                # Assign SHAP value to original feature
                for feat_idx in range(len(feature_names)):
                    # Weight is the contribution of the original feature in the PCA component
                    weight = abs(components[pca_idx, feat_idx])
                    # Assign SHAP value to original feature by weight
                    original_importance[feat_idx] += abs(class_shap_values[sample_idx, pca_idx]) * weight
        
        # Normalize importance scores
        original_importance = original_importance / original_importance.sum() * 100
        
        # Create DataFrame for original feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': original_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(f'{output_dir}/original_feature_importance_{class_name}.csv', index=False)
        
        # Plot top 20 most important original features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['Importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Relative Importance (%)')
        plt.title(f'Top 20 Original Features Importance for {class_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/original_feature_importance_{class_name}.png')
        plt.close()
        
        print(f"Top 10 most important original features for class {class_name}:")
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

    plt.savefig(f'figure/Receiver Operating Characteristic (ROC) Curves.png')

if __name__ == '__main__':  
    # columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    # columns_to_encode = ['MCQ160C', 'MCQ160F']
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    # columns_to_encode = ['MCQ160B', 'MCQ160D']

    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)


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
            std[i] = 1e-8


    from sklearn.feature_selection import VarianceThreshold


    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8

    features = (features - mean_f) / std_f
    features_val = (features_val - mean_f) / (std_f + 1e-8)
    features = features.reshape(features.shape[0], -1)

    print("PCA降维前，训练集形状：", features.shape)
    print("PCA降维前，验证集形状：", features_val.shape)


    # include PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    pca_selector = VarianceThreshold(threshold=0.01)
    features = pca_selector.fit_transform(pca.fit_transform(features))
    features_val = pca_selector.transform(pca.transform(features_val))

    # Update PCA feature names to PC1, PC2, ..., PCn
    pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1] + 1)]
    print("PCA降维后，训练集形状：", features.shape)
    print("PCA降维后，验证集形状：", features_val.shape)
    
    Shap_features = features_val.copy()
    # pause = input("Press Enter to continue...")

    # Recalculate mean and std for PCA features
    mean_pca, std_pca = np.mean(features, axis=0), np.std(features, axis=0)

    folds_data_index = split_data_5fold(features)

    num_x = len(Covariates_features)
    num_folds = len(folds_data_index)

    # the list to store accuracies, precisions, recalls, f1_scores, macro_auc, and weighted_auc for each x
    all_accuracies = [[] for _ in range(num_x)]
    all_precisions = [[] for _ in range(num_x)]
    all_recalls = [[] for _ in range(num_x)]
    all_f1_scores = [[] for _ in range(num_x)]
    all_macro_auc = [[] for _ in range(num_x)]
    all_weighted_auc = [[] for _ in range(num_x)]


    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  

        l1_weight = 0.070
        num_classes = len(columns_to_encode)  # Adjust the number of classes
        num_epochs = 350
        batch_size = 256
        input_length = batch_size
        for x in range(len(Covariates_features)):


            model = DNNModel(num_classes=num_classes, input_dim=features.shape[1])

            # temperature scaling model
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


            all_preds = np.empty((0, len(columns_to_encode)))
            all_targets = np.empty((0, len(columns_to_encode)))

            valid_f1_scores = []
            valid_precision_scores = []


            train_losses = []
            valid_f1_history = []
            valid_accuracy_history = []
            valid_precision_history = []
            valid_recall_history = []

            # early stopping parameters
            best_valid_f1 = 0
            patience = 15  # patience for early stopping
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

                    # L1 regularization
                    l1_criterion = nn.L1Loss()
                    l1_loss = 0
                    for param in model.parameters():
                        l1_loss += l1_criterion(param, torch.zeros_like(param))
                    loss += l1_weight * l1_loss  # l1_weight is the weight for L1 regularization, can be adjusted as needed

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)

                average_train_loss = train_loss / len(Train_data_loader.dataset)
                scheduler.step()

                # record training loss
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

                        # save predictions and targets for confusion matrix
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

                # record validation metrics
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

                # plot confusion matrix
                if epoch + 1 >= 300:
                    cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
                    cm = multilabel_confusion_matrix(target.cpu().numpy(), preds)
                    print(cm_sum)
                    print(cm)

                    # add code to plot confusion matrix
                    plot_confusion_matrix(cm, columns_to_encode, fold, epoch)
                    plot_confusion_matrix(cm_sum, columns_to_encode, fold, epoch, is_sum=True)

                # early stopping: check if validation F1 score has improved
                if valid_f1 > best_valid_f1:
                    best_valid_f1 = valid_f1
                    counter = 0
                    best_epoch = epoch
                    # save best model
                    torch.save(model.state_dict(), f'single/best_model_{fold}_{x}.ckpt')
                    # save scaled model
                    torch.save(scaled_model.state_dict(), f'single/best_scaled_model_{fold}_{x}.ckpt')
                else:
                    counter += 1
                    print(f'EarlyStopping counter: {counter} out of {patience}')
                    if counter >= patience:
                        print(f'Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with F1: {best_valid_f1:.2f}%')
                        # plot confusion matrix for current fold when early stopping
                        cm_sum = multilabel_confusion_matrix(all_targets, all_preds)
                        print(f"Confusion matrix when early stopping (Fold {fold+1}):")
                        print(cm_sum)
                        # plot confusion matrix
                        plot_confusion_matrix(cm_sum, columns_to_encode, fold, epoch, is_sum=True)
                        
                        break

            # plot learning curve
            plt.figure(figsize=(12, 10))

            # create two subplots
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

            # ensure figure directory exists
            if not os.path.exists('figure'):
                os.makedirs('figure')
                
            plt.savefig(f'figure/learning_curve_fold{fold+1}_x{x+1}.png')
            plt.close()       
            # save model
            # torch.save(model.state_dict(), f'single/best_model_{fold}_{x}.ckpt')
            torch.save(model.state_dict(), f'single/model_{fold}_{x}.ckpt')  

            # test set
            Test_F1, Test_accuracy, Test_precision, Test_recall, Test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt', Scaled_Model=None)
            Scaled_F1, Scaled_accuracy, Scaled_precision, Scaled_recall, Scaled_specificity, Scaled_macro_auc, Scaled_weighted_auc, _, _, _ = test(Test_data_loader, probs_Switcher, f'single/model_{fold}_{x}.ckpt', Scaled_Model = scaled_model)
            # plot_roc_curves(all_labels, all_probs, columns_to_encode)
            print(
                f'fold [{fold + 1}/5]: '
                f'Test F1: {Test_F1:.2f}% | '
                f'Accuracy: {Test_accuracy:.2f}% | '
                f'Precision: {Test_precision:.2f}% | '
                f'Recall: {Test_recall:.2f}% | '
                f'Specificity: {Test_specificity:.2f}% | '
                f'Macro AUC: {macro_auc:.3f} | '
                f'Weighted AUC: {weighted_auc:.3f}'
            )

            print(
                f'fold [{fold + 1}/5]: '
                f'Scaled F1: {Scaled_F1:.2f}% | '
                f'Scaled Accuracy: {Scaled_accuracy:.2f}% | '
                f'Scaled Precision: {Scaled_precision:.2f}% | '
                f'Scaled Recall: {Scaled_recall:.2f}% | '
                f'Scaled Specificity: {Scaled_specificity:.2f}% | '
                f'Scaled Macro AUC: {Scaled_macro_auc:.3f} | '
                f'Scaled Weighted AUC: {Scaled_weighted_auc:.3f}'
            )
            
            with open('Test.csv', 'a') as f:
                f.write(
                    f'fold [{fold + 1}/5], '
                    f'Test F1: {Test_F1:.2f}%, '
                    f'Accuracy: {Test_accuracy:.2f}%, '
                    f'Precision: {Test_precision:.2f}%, '
                    f'Recall: {Test_recall:.2f}%, '
                    f'Specificity: {Test_specificity:.2f}%, '
                    f'Macro AUC: {macro_auc:.3f}, '
                    f'Weighted AUC: {weighted_auc:.3f}\n'
                )

            with open('Scaled.csv', 'a') as f:
                f.write(
                    f'fold [{fold + 1}/5], '
                    f'Scaled F1: {Scaled_F1:.2f}%, '
                    f'Scaled Accuracy: {Scaled_accuracy:.2f}%, '
                    f'Scaled Precision: {Scaled_precision:.2f}%, '
                    f'Scaled Recall: {Scaled_recall:.2f}%, '
                    f'Scaled Specificity: {Scaled_specificity:.2f}%, '
                    f'Scaled Macro AUC: {Scaled_macro_auc:.3f}, '
                    f'Scaled Weighted AUC: {Scaled_weighted_auc:.3f}\n'
                )

            # save current x and fold metrics
            all_accuracies[x].append(Test_accuracy)
            all_precisions[x].append(Test_precision)
            all_recalls[x].append(Test_recall)
            all_f1_scores[x].append(Test_F1)
            all_macro_auc[x].append(macro_auc)
            all_weighted_auc[x].append(weighted_auc)

    
    if len(selected_folds_input) == 5:
        # plot accuracy line chart, x is the number of added covariates
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
        
        plt.savefig(f'figure/Comparison of Accuracy for Different x Values Across Folds.png')

       
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
      
        plt.savefig(f'figure/Comparison of Precision for Different x Values Across Folds.png')

     
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

        plt.savefig(f'figure/Comparison of Recall for Different x Values Across Folds.png')


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

        plt.savefig(f'figure/Comparison of F1 Score for Different x Values Across Folds.png')


    # origin = pd.DataFrame(features_val, columns=all_feature_names)
    pca_feature_names = [f'PC{i}' for i in range(1, features_val.shape[1] + 1)]
    origin = pd.DataFrame(features_val, columns=pca_feature_names)  




    COLUMNS = origin.columns  
    num_classes = labels_val.shape[1] 


    K = 32  


    Shap_features_filtered = pca_selector.transform(Shap_features)
    background_data = shap.kmeans(Shap_features_filtered, K)


    n_samples_for_global_shap = min(500, len(Shap_features_filtered)) 
    global_shap_indices = np.random.choice(len(Shap_features_filtered), n_samples_for_global_shap, replace=False)
    global_shap_samples = Shap_features_filtered[global_shap_indices]

    selected_indices = np.random.choice(len(Shap_features), 3501, replace=True)
    selected_features_val = Shap_features[selected_indices]

    ALL_shap_exp = {}
    ALL_top_features = {}


    # create SHAP model
    Shap_model = Model_VGG.Trained_VGG11(model, num_epochs, mean_pca, std_pca, device).eval()

    # execute SHAP analysis
    print("Starting global SHAP analysis...")
    global_shap_values = global_shap_analysis(
        model=Shap_model,
        background_data=background_data,
        test_data=global_shap_samples,
        feature_names=pca_feature_names,
        class_names=columns_to_encode,
        output_dir='figure/global_shap'
    )
    print("Global SHAP analysis completed!")

    # Map SHAP values of PCA features back to original feature space
    print("Starting to map SHAP values back to original feature space...")
    original_feature_importance = map_pca_shap_to_original_features(
        shap_values=global_shap_values,
        pca_model=pca,  # Use previously trained PCA model
        feature_names=all_feature_names,  # Original feature names
        class_names=columns_to_encode,
        output_dir='figure/original_feature_shap'
    )
    print("Mapping SHAP values back to original feature space completed!")

    for class_idx in range(num_classes):

        # Use SHAP to evaluate feature importance
        print(f"Evaluating feature importance for class {columns_to_encode[class_idx]} using SHAP...")

        explainer = shap.KernelExplainer(model_wrapper, background_data)
        shap_values = explainer.shap_values(selected_features_val, nsamples=256, main_effects=False, interaction_index=None)

        base_values = explainer.expected_value

        if isinstance(base_values, (int, float)):
            base_values = np.full(len(selected_features_val), base_values)

        shap_exp = shap.Explanation(shap_values, data=selected_features_val, feature_names=all_feature_names, base_values=base_values)

        # Save shap_exp
        ALL_shap_exp[columns_to_encode[class_idx]] = shap_exp

        # Plot SHAP summary bar chart
        shap.plots.bar(shap_exp, max_display=16)
        # Save to figure folder, named by the plot title
        plt.savefig(f'figure/SHAP Summary Plot for category{columns_to_encode[class_idx]}.png')

        # Plot SHAP beeswarm summary chart
        shap.plots.beeswarm(shap_exp, max_display=16)
        # Save to figure folder, named by the plot title
        plt.savefig(f'figure/SHAP Beeswarm Plot for category{columns_to_encode[class_idx]}.png')

        # Feature ablation experiment
        print(f"Calculating feature importance for class {columns_to_encode[class_idx]}...")
        out = pd.DataFrame({'tag': predict(features_val, probs_Switcher)[:, class_idx]})
        importance_dict = {}  # Used to store the importance calculation result for each feature

        for key in COLUMNS:
            copy = origin.copy()
            copy[key] = copy[key].sample(frac=1, random_state=1).reset_index()[key]
            cp_out = predict(copy.values, probs_Switcher)[:, class_idx]
            # Convert out['tag'] to numpy.ndarray before subtraction
            diff = (out['tag'].values - cp_out).flatten()
            importance_dict[key] = diff ** 2
            print('key = ', key, ' affect = ', importance_dict[key].sum() ** 0.5)

        # Merge all columns into out at once
        importance_df = pd.DataFrame(importance_dict)
        out = pd.concat([out, importance_df], axis=1)

        importance_result = (pd.DataFrame(out.sum(axis=0)) ** 0.5).sort_values(by=0, ascending=False)
        print(f"Feature importance ranking result for class {class_idx}:")
        print(importance_result)

        # Plot bar chart
        plt.figure(figsize=(15, 6))
        top_features = importance_result.iloc[1:].head(64)  # Take the top 64 features
        plt.bar(top_features.index, top_features[0])
        plt.xlabel('Features')
        plt.ylabel('Importance of Features')
        plt.title(f'Bar chart of the top 64 feature importances for category{columns_to_encode[class_idx]}')
        plt.xticks(rotation=45, fontsize=6)
        plt.tight_layout()
        #plt.show()
        # Save as png in figure folder, named by the plot title
        plt.savefig(f'figure/Bar chart of the top 64 feature importances for category{columns_to_encode[class_idx]}.png')



        # Select top SHAP important features (here, take the top 209 by mean absolute value)
        shap_importance = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(shap_importance)[::-1]
        top_209_features = np.array(all_feature_names)[sorted_indices[:209]]
        ALL_top_features[columns_to_encode[class_idx]] = top_209_features

    # Stratified analysis: stratify by disease severity, gender, and ethnicity. Currently using multivariate linear regression to show linear correlation of features in different strata. Some features cannot be inferred for linear correlation.
    # SHAP plots can supplement to show how nonlinear features affect the outcome (nonlinear analysis?), currently using waterfall plots (visualization of SHAP value impact for a group) and force plots (individual SHAP value impact). Reference images are in the folder.

    stratify_variable = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']  # disease severity, gender, ethnicity

    for i in stratify_variable:
        # Get unique values for stratification variable
        strata = Multilay_origin[i].unique()

        for class_idx in range(num_classes):
            top_209_features = ALL_top_features[columns_to_encode[class_idx]]

            for stratum in strata:
                # Select data for current stratum
                stratum_mask = Multilay_origin[i] == stratum
                stratum_indices = np.where(stratum_mask)[0]
                # Ensure only select data with the same indices as selected_features_val
                common_indices = np.intersect1d(stratum_indices, selected_indices)
                stratum_data = Multilay_origin[stratum_mask].copy()
                labels_stratum = labels_val[stratum_indices]

                # Build dataset for linear regression
                regression_data = stratum_data[top_209_features].copy()

                regression_data['disease_label'] = labels_stratum[:, class_idx].astype(int)

                # Generate boolean array with the same length as shap_values.data
                bool_tf = np.isin(selected_indices, common_indices)

                # Here calculate the mean SHAP value for a group, used for stratified analysis
                try:
                    # Get the shap.Explanation object for the current class
                    shap_exp = ALL_shap_exp[columns_to_encode[class_idx]]

                    # Filter shap.Explanation object
                    filtered_shap_values = shap_exp.values[bool_tf]
                    filtered_data = shap_exp.data[bool_tf]
                    filtered_base_values = shap_exp.base_values[bool_tf]

                    # Create filtered shap.Explanation object
                    filtered_shap_exp = shap.Explanation(
                        values=filtered_shap_values,
                        data=filtered_data,
                        feature_names=shap_exp.feature_names,
                        base_values=filtered_base_values
                    )

                    new_shap = new_shap_values(ALL_shap_exp[columns_to_encode[class_idx]], bool_tf=bool_tf, method='mean')
                    # Ensure the object passed to shap.plots.waterfall is a shap.Explanation object
                    shap.plots.waterfall(new_shap.get_explanation())

                    shap.plots.beeswarm(filtered_shap_exp, max_display=64)
                except Exception as e:
                    print(f"Error plotting SHAP waterfall plot, class {columns_to_encode[class_idx]}, {i} = {stratum}, error: {e}")

                # Build linear regression formula
                if np.var(regression_data['disease_label']) == 0:
                    print(f"Dependent variable 'disease_label' has no variance, skipping linear regression for class {columns_to_encode[class_idx]} at {i} = {stratum}.")
                else:
                    formula = 'disease_label ~ ' + ' + '.join(top_209_features)
                    try:
                        Ols_model = smf.ols(formula, data=regression_data).fit(cov_type='HC3')

                        # Output regression results
                        print(f"\nMultivariate linear regression results for class {columns_to_encode[class_idx]} at {i} = {stratum}:")
                        print(Ols_model.summary())

                        # Output correlation description
                        print(f"\nFor class {columns_to_encode[class_idx]}, when {i} = {stratum}, covariate-adjusted model shows:")
                        for feature in top_209_features:
                            coef = Ols_model.params[feature]
                            p_value = Ols_model.pvalues[feature]
                            conf_int = Ols_model.conf_int().loc[feature]
                            if p_value < 0.05:
                                if coef > 0:
                                    print(
                                        f"{feature} is positively correlated with the disease label. Specifically, for each unit increase in {feature}, the probability of {columns_to_encode[class_idx]} increases by {coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% to {conf_int[1] * 100:.2f}%).")
                                else:
                                    print(
                                        f"{feature} is negatively correlated with the disease label. Specifically, for each unit increase in {feature}, the probability of {columns_to_encode[class_idx]} decreases by {-coef * 100:.2f}% (95% CI {conf_int[0] * 100:.2f}% to {conf_int[1] * 100:.2f}%).")
                            else:
                                print(f"No significant linear relationship found between {feature} and the disease label (p = {p_value:.4f}).")
                    except Exception as e:
                        print(f"Linear regression model fitting failed for class {columns_to_encode[class_idx]}, {i} = {stratum}, error: {e}")


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

    # Add analyze_shap_performance_correlation function here
    def analyze_shap_performance_correlation(global_shap_values, test_metrics, class_names, output_dir='figure/shap_performance'):
        """
        Analyze the relationship between SHAP values and model performance metrics
        
        Args:
            global_shap_values -- list of global SHAP values
            test_metrics -- dictionary of test metrics, including 'f1', 'accuracy', 'precision', 'recall', etc.
            class_names -- list of class names
            output_dir -- output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Calculate mean absolute SHAP value for each class
        mean_abs_shap = []
        for i, shap_values in enumerate(global_shap_values):
            mean_abs_shap.append(np.mean(np.abs(shap_values)))
            
        # Create comparison plots between performance and SHAP values
        metrics = ['f1', 'accuracy', 'precision', 'recall']
        for metric in metrics:
            if metric in test_metrics:
                plt.figure(figsize=(10, 6))
                plt.scatter(mean_abs_shap, test_metrics[metric], c=range(len(class_names)), cmap='viridis')
                
                # Add class labels
                for i, class_name in enumerate(class_names):
                    plt.annotate(class_name, 
                                (mean_abs_shap[i], test_metrics[metric][i]),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
                
                # Add trend line
                z = np.polyfit(mean_abs_shap, test_metrics[metric], 1)
                p = np.poly1d(z)
                plt.plot(mean_abs_shap, p(mean_abs_shap), "r--", alpha=0.8)
                
                plt.xlabel('Mean Absolute SHAP Value')
                plt.ylabel(f'{metric.capitalize()} (%)')
                plt.title(f'Relationship between SHAP Value and {metric.capitalize()}')
                plt.colorbar(label='Class Index')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/shap_vs_{metric}.png')
                plt.close()
    
    # If global SHAP values have been calculated, analyze their relationship with model performance
    if 'global_shap_values' in locals():
        # Collect test metrics
        test_metrics = {
            'f1': [all_f1_scores[0][i] for i in range(len(columns_to_encode))],
            'accuracy': [all_accuracies[0][i] for i in range(len(columns_to_encode))],
            'precision': [all_precisions[0][i] for i in range(len(columns_to_encode))],
            'recall': [all_recalls[0][i] for i in range(len(columns_to_encode))]
        }
        
        # Analyze the relationship between SHAP values and performance
        analyze_shap_performance_correlation(
            global_shap_values=global_shap_values,
            test_metrics=test_metrics,
            class_names=columns_to_encode,
            output_dir='figure/shap_performance'
        )
