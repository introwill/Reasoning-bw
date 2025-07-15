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

matplotlib.use('TKAgg')  # Used to resolve errors when plotting
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


# Data standardization and cross-validation split
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
def plot_confusion_matrix(cm, classes, fold, epoch, title='Confusion Matrix', cmap=plt.cm.Blues, is_sum=False):
    """
    Plot and display the confusion matrix with normalization
    
    Parameters:
    cm -- confusion matrix, for multi-label problems, shape is (n_classes, 2, 2)
    classes -- list of class names
    fold -- current fold number
    epoch -- current epoch number
    title -- chart title
    cmap -- color map
    is_sum -- whether it's a cumulative confusion matrix
    """
    # Ensure figure directory exists
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # Handle multi-label confusion matrix
    if cm.ndim == 3 and cm.shape[1] == 2 and cm.shape[2] == 2:
        # Multi-label case, plot separate confusion matrix for each class
        for i, class_name in enumerate(classes):
            plt.figure(figsize=(8, 6))
            
            # Extract current class confusion matrix
            cm_i = cm[i]
            
            # Create a copy of the original confusion matrix
            cm_original = cm_i.copy()
            
            # Normalize confusion matrix
            cm_row_sum = cm_i.sum(axis=1)
            cm_normalized = np.zeros_like(cm_i, dtype=float)
            for j in range(cm_i.shape[0]):
                if cm_row_sum[j] > 0:
                    cm_normalized[j] = cm_i[j] / cm_row_sum[j]
            
            # Plot normalized confusion matrix
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
            plt.title(f'{title} - {class_name} (Fold {fold+1}, Epoch {epoch})')
            plt.colorbar()
            
            # Set labels
            labels = ['Negative', 'Positive']
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels)
            plt.yticks(tick_marks, labels)
            
            # Add original values and normalized percentages to each cell
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
        
        # Create a summary visualization of the confusion matrix
        plt.figure(figsize=(15, 10))
        
        # Calculate performance metrics for each class
        metrics = []
        for i, class_name in enumerate(classes):
            tn, fp = cm[i][0, 0], cm[i][0, 1]
            fn, tp = cm[i][1, 0], cm[i][1, 1]
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append([class_name, accuracy, precision, recall, specificity, f1])
        
        # Create table
        columns = ['Class', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'] 
        cell_text = [[f"{m[0]}", f"{m[1]:.2f}", f"{m[2]:.2f}", f"{m[3]:.2f}", f"{m[4]:.2f}", f"{m[5]:.2f}"] for m in metrics]
        
        # Draw table
        table = plt.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.axis('off')
        
        plt.title(f'Performance Metrics Summary (Fold {fold+1})', fontsize=14)
        
        # save the summary figure
        if is_sum:
            plt.savefig(f'figure/confusion_matrix_summary_fold{fold+1}_sum.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figure/confusion_matrix_summary_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    else:
        # Single-label case, original code
        plt.figure(figsize=(12, 10))

        # Create a copy of the original confusion matrix
        cm_original = cm.copy()

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Handle potential division by zero
        cm_normalized = np.nan_to_num(cm_normalized)

        # Plot normalized confusion matrix
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.title(f'{title} (Fold {fold+1}, Epoch {epoch})')
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add original values and normalized percentages to each cell
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if np.sum(cm_original[i]) > 0:  # 避免除以零
                plt.text(j, i, f"{cm_original[i, j]}\n({cm_normalized[i, j]:.2f})",
                        horizontalalignment="center", 
                        color="white" if cm_normalized[i, j] > 0.5 else "black", 
                        fontsize=8)
        
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # save image
        if is_sum:
            plt.savefig(f'figure/confusion_matrix_normalized_fold{fold+1}_sum.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figure/confusion_matrix_normalized_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
        
        plt.close()

def generate_multilabel_confusion_matrix(y_true, y_pred, fold, epoch):
    """
    Generate a 32×32 confusion matrix for multi-label classification
    
    Parameters:
    y_true -- true labels, shape (n_samples, 5)
    y_pred -- predicted labels, shape (n_samples, 5)
    fold -- current fold number
    epoch -- current epoch number
    """
    # Ensure figure directory exists
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # Convert multi-label to single category labels (0-31)
    def multilabel_to_class(labels):
        return np.sum(labels * np.array([2**i for i in range(labels.shape[1])]), axis=1).astype(int)
    
    # Convert true labels and predicted labels
    y_true_class = multilabel_to_class(y_true)
    y_pred_class = multilabel_to_class(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_class, y_pred_class, labels=range(32))
    
    # Create class labels
    class_names = []
    for i in range(32):
        # Convert number to binary representation, then pad to 5 digits
        binary = format(i, '05b')
        # Create label, e.g. "10110" means diseases 1, 3, 4 are present
        class_names.append(binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix
    cm_row_sum = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if cm_row_sum[i] > 0:
            cm_normalized[i] = cm[i] / cm_row_sum[i]
    
    # Plot normalized confusion matrix
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'32×32 Confusion Matrix (Fold {fold+1}, Epoch {epoch})', fontsize=16)
    plt.colorbar()
    
    # Set labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    # Add values to each cell
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:  # Only show non-zero values
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center", 
                    color="white" if cm_normalized[i, j] > thresh else "black", 
                    fontsize=6)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    # save image
    plt.savefig(f'figure/confusion_matrix_32x32_fold{fold+1}_epoch{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate performance metrics for each class
    precision = np.zeros(32)
    recall = np.zeros(32)
    f1 = np.zeros(32)
    
    # Calculate precision, recall and F1 score for each class
    for i in range(32):
        # Calculate true positives, false positives and false negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate precision and recall
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0
            
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0
            
        # Calculate F1 score
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0
    
    return cm, precision, recall, f1



def test_xgb(X_test, y_test, models, probs_Switcher):
    all_probs = np.zeros((X_test.shape[0], len(models)))
    

    for i, model in enumerate(models):
        dtest = xgb.DMatrix(X_test)
        all_probs[:, i] = model.predict(dtest)
    
    test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(all_probs, y_test, probs_Switcher)

    # Calculate AUC-ROC
    auc_scores = []
    for i in range(y_test.shape[1]):
        try:
            auc_score = roc_auc_score(y_test[:, i], all_probs[:, i])
            auc_scores.append(auc_score)
        except ValueError:
            print(f"Class {i} has only one class present in test set.")
            auc_scores.append(float('nan'))

    # Calculate macro and weighted average AUC
    macro_auc = np.nanmean(auc_scores)
    weighted_auc = roc_auc_score(y_test, all_probs, average='weighted', multi_class='ovr')
    
    return test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, y_test, preds

# Threshold calculation function
def Probs_Switcher(probs, labels):
    probs_Switcher = np.array([])
    
    for i in range(labels.shape[1]):
        split_labels_np = labels[:, i]
        split_probs_np = probs[:, i]
        precision, recall, thresholds = precision_recall_curve(split_labels_np, split_probs_np)
        precision = precision * 0.6  # This weight can be adjusted as needed
        recall = recall
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        index = argmax(f1_scores)
        if len(thresholds) > index:
            probs_Switcher = np.append(probs_Switcher, thresholds[index])
        else:
            # If index exceeds thresholds range, use default threshold 0.5
            probs_Switcher = np.append(probs_Switcher, 0.5)

    return probs_Switcher

# Metrics calculation function
def f1_score_func(probs, labels, probs_Switcher):
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
    specificity = np.mean(specificities) * 100  # Using macro average

    return f1 * 100, accuracy * 100, precision * 100, recall * 100, preds, specificity

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

# Modified VGG feature extractor class
class DNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=92):
        super(DNNFeatureExtractor, self).__init__()
        # Create a new feature extraction network suitable for our input dimensions
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),  # Input dimension is 93 (number of PCA features)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        return self.feature_extractor(x)

# DNN training function
def train_dnn_extractor(model, train_loader, val_loader, device, num_epochs=100):
    model.train()
    criterion = nn.MSELoss()  # Using MSE loss for feature learning
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data, _ in train_loader:  # We don't need labels, performing unsupervised learning
            data = data.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)  # Autoencoder-style training
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.float().to(device)
                output = model(data)
                val_loss += criterion(output, data).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

        # Early stopping mechanism
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_dnn_extractor.pth')

    # Load best model
    model.load_state_dict(torch.load('best_dnn_extractor.pth'))
    return model

# Modify feature extraction function
def extract_vgg_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.float().to(device)
            # Extract features
            feature = model(data)
            
            features.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    features = np.vstack(features)
    labels = np.vstack(labels)
    
    return features, labels

# Test function
def test_xgb(X_test, y_test, models, probs_Switcher):
    all_probs = np.zeros((X_test.shape[0], len(models)))

    # Use corresponding model for each class to make predictions
    for i, model in enumerate(models):
        dtest = xgb.DMatrix(X_test)
        all_probs[:, i] = model.predict(dtest)
    
    test_f1, test_accuracy, test_precision, test_recall, preds, test_specificity = f1_score_func(all_probs, y_test, probs_Switcher)

    # Calculate AUC-ROC
    auc_scores = []
    for i in range(y_test.shape[1]):
        try:
            auc_score = roc_auc_score(y_test[:, i], all_probs[:, i])
            auc_scores.append(auc_score)
        except ValueError:
            print(f"Class {i} has only one class present in test set.")
            auc_scores.append(float('nan'))

    # Calculate macro and weighted average AUC
    macro_auc = np.nanmean(auc_scores)
    weighted_auc = roc_auc_score(y_test, all_probs, average='weighted', multi_class='ovr')
    
    return test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, y_test, preds

# SHAP analysis
def global_shap_analysis(models, background_data, test_data, feature_names, class_names, 
                        pca_model=None, original_feature_names=None, output_dir='figure/global_shap'):
    """
    Enhanced SHAP analysis with feature reverse mapping
    
    Parameters:
    models -- trained XGBoost model list
    background_data -- background data for SHAP explainer
    test_data -- test data for generating SHAP values
    feature_names -- feature name list
    class_names -- class name list
    pca_model -- PCA model object used
    original_feature_names -- original feature name list
    output_dir -- output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check feature name length
    if len(feature_names) != test_data.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(test_data.shape[1])]
        print(f"Warning: Automatically generating feature names, length is {len(feature_names)}")
    
    # Store SHAP values for all classes
    all_shap_values = []
    
    # Process SHAP values for each class
    for i, (class_name, model) in enumerate(zip(class_names, models)):
        print(f"Calculating global SHAP values for class {class_name}...")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(test_data)
        all_shap_values.append(shap_values)
        
        # Plot bar chart
        plt.figure(figsize=(10, 8))
        try:
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(0)
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
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/global_bar_plot_category_{class_name}.png')
        plt.close()
        
        # Plot summary chart
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values, test_data, feature_names=feature_names, show=False, max_display=20)
            plt.title(f'SHAP Summary Plot for Category {class_name}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/global_summary_plot_category_{class_name}.png')
        except Exception as e:
            print(f"Error when plotting summary chart: {e}")
        plt.close()

    # Plot polynomial chart
    mean_abs_shap = np.zeros((len(feature_names), len(class_names)))
    for i, shap_values in enumerate(all_shap_values):
        mean_abs_shap[:, i] = np.abs(shap_values).mean(axis=0)
    
    agg_shap_df = pd.DataFrame(mean_abs_shap, columns=class_names, index=feature_names)

    # Sort by feature importance sum
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

    # New feature reverse mapping analysis
    if pca_model is not None and original_feature_names is not None and len(original_feature_names) > 0:
        print("\nStarting feature importance reverse mapping analysis...")
        
        # Calculate SHAP importance for PCA components
        pca_shap_importance = np.abs(np.array(all_shap_values)).mean(axis=(0,1))
        
        # Ensure dimensions match
        if len(pca_shap_importance) != pca_model.components_.shape[0]:
            print(f"Warning: SHAP importance dimension ({len(pca_shap_importance)}) does not match PCA component count ({pca_model.components_.shape[0]}), will truncate or pad")
            min_dim = min(len(pca_shap_importance), pca_model.components_.shape[0])
            pca_shap_importance = pca_shap_importance[:min_dim]
            pca_components = pca_model.components_[:min_dim, :]
        else:
            pca_components = pca_model.components_
        
        # Calculate original feature importance = PCA component importance × PCA loading matrix
        original_feature_importance = np.dot(pca_shap_importance, np.abs(pca_components))  # Using truncated pca_components

        # Create and save original feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': original_feature_names.values if hasattr(original_feature_names, 'values') else original_feature_names,
            'importance': original_feature_importance
        }).sort_values('importance', ascending=False)

        # Save to CSV
        importance_df.to_csv(f'{output_dir}/original_feature_importance.csv', index=False)

        # Plot original feature importance
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

    # Plot ROC curve for each class
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
    Main function containing the primary execution logic of the program
    """
    # set label encoding columns
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']

    # Load data
    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)
    
    
    features_val = features.copy()
    labels_val = labels.copy()
    
    # data augmentation using MLSMOTE
    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)
    
    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)
    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)
    
    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)

    # Standardization
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8
    
    features = (features - mean_f) / std_f
    features_val = (features_val - mean_f) / (std_f + 1e-8)
    
  
        
    filtered_feature_names = all_feature_names.copy()

   
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features)
    features_val_pca = pca.transform(features_val)

   
    train_labels = labels.copy()  #
    val_labels = labels_val.copy()  #
    

    

    pca_feature_names = [f'PC{i}' for i in range(1, features_pca.shape[1] + 1)]
    print("After PCA dimensionality reduction, training set shape:", features_pca.shape)
    print("After PCA dimensionality reduction, validation set shape:", features_val_pca.shape)
    

    folds_data_index = split_data_5fold(features_val_pca)
    

    num_classes = len(columns_to_encode)
    num_epochs = 100  
    batch_size = 256
    

    selected_folds_input = '1,2,3,4,5'  

    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]
    

    all_fold_results = []
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue

        print(f"\n========== Processing Fold {fold + 1} ==========")
        
        # 
        trainX = features_pca[train_index]
        trainY = labels[train_index]
        valX = features_val_pca[validation_index]
        valY = labels_val[validation_index]
        testX = features_val_pca[test_indices]
        testY = labels_val[test_indices]
        
        # create datasets and dataloaders
        Train_data = NetDataset(trainX, trainY)
        Validation_data = NetDataset(valX, valY)
        Test_data = NetDataset(testX, testY)
        
        Train_data_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        Validation_data_loader = DataLoader(Validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
        Test_data_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        
        # Phase 1: Train VGG model as feature extractor
        print("Phase 1: Training VGG model as feature extractor...")
        
        # Initialize VGG model
        input_dim = features_pca.shape[1]
        model = Model_VGG.VGG11(num_classes=num_classes, in_channels=features.shape[1], epoch=num_epochs).to(device)
        
        # ensure model parameters are float type
        model = model.float()  

        # Print model parameter count
        total_num, trainable_num = get_parameter_number(model)
        print(f'VGG model total parameters: {total_num}, trainable parameters: {trainable_num}')
        
       
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        

        best_valid_f1 = 0
        patience = 10
        counter = 0
        best_epoch = 0
        best_model = None
        
        # training loop
        for epoch in range(1, num_epochs + 1):

            model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []
            
            for data, target in Train_data_loader:
                data, target = data.to(device), target.to(device).float()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                

                pred = torch.sigmoid(output).detach().cpu().numpy()
                train_preds.append(pred)
                train_true.append(target.cpu().numpy())
            

            train_loss /= len(Train_data_loader)
            train_preds = np.concatenate(train_preds)
            train_true = np.concatenate(train_true)
            train_f1 = f1_score(train_true, (train_preds > 0.5).astype(int), average='macro') * 100

            model.eval()
            valid_loss = 0.0
            valid_preds = []
            valid_true = []
            
            with torch.no_grad():
                for data, target in Validation_data_loader:
                    data, target = data.to(device), target.to(device).float()

                    output = model(data)
                    loss = criterion(output, target)
                    
                    valid_loss += loss.item()

                    pred = torch.sigmoid(output).cpu().numpy()
                    valid_preds.append(pred)
                    valid_true.append(target.cpu().numpy())

            valid_loss /= len(Validation_data_loader)
            valid_preds = np.concatenate(valid_preds)
            valid_true = np.concatenate(valid_true)
            valid_f1 = f1_score(valid_true, (valid_preds > 0.5).astype(int), average='macro') * 100
            

            scheduler.step()

            print(f'Epoch {epoch}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.2f}% | '
                  f'Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.2f}%')

            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                counter = 0
                best_epoch = epoch
                best_model = model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        model.load_state_dict(best_model)
        print(f'Using best model (Epoch {best_epoch}, Validation F1: {best_valid_f1:.2f}%)')
        

        class VGGFeatureExtractor(nn.Module):
            def __init__(self, vgg_model):
                super(VGGFeatureExtractor, self).__init__()
                self.features = nn.Sequential(*list(vgg_model.children())[:-1])  # 移除最后的分类层
                
            def forward(self, x):
                return self.features(x)
        

        feature_extractor = VGGFeatureExtractor(model).to(device)
        feature_extractor.eval()
        
        print("Phase 2: Using VGG to extract features, XGBoost for prediction...")
        

        def extract_vgg_features(model, data_loader, device):
            model.eval()
            features = []
            labels = []
            
            with torch.no_grad():
                for data, target in data_loader:
                    data = data.to(device)
                    output = model(data)
                    features.append(output.cpu().numpy())
                    labels.append(target.numpy())
            
            return np.concatenate(features), np.concatenate(labels)
        

        train_features, train_labels = extract_vgg_features(feature_extractor, Train_data_loader, device)
        val_features, val_labels = extract_vgg_features(feature_extractor, Validation_data_loader, device)
        test_features, test_labels = extract_vgg_features(feature_extractor, Test_data_loader, device)
        
        print(f"提取的训练特征形状: {train_features.shape}, 训练标签形状: {train_labels.shape}")
        print(f"提取的验证特征形状: {val_features.shape}, 验证标签形状: {val_labels.shape}")
        print(f"提取的测试特征形状: {test_features.shape}, 测试标签形状: {test_labels.shape}")
        

        xgb_models = []
        xgb_feature_names = [f'VGG_Feature_{i}' for i in range(train_features.shape[1])]
        

        probs_Switcher = np.zeros(num_classes)
        
        for i in range(num_classes):
            print(f"训练类别 {columns_to_encode[i]} 的XGBoost模型...")
            

            dtrain = xgb.DMatrix(train_features, label=train_labels[:, i])
            dval = xgb.DMatrix(val_features, label=val_labels[:, i])

            pos_count = np.sum(train_labels[:, i])
            neg_count = len(train_labels[:, i]) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['auc', 'logloss'],
                'max_depth': 4,  
                'eta': 0.05,  
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,  
                'gamma': 0.1,  
                'alpha': 0.2,  
                'lambda': 1.5,  
                'scale_pos_weight': scale_pos_weight,  
                'seed': 3407
            }
            

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=500,
                nfold=5,
                early_stopping_rounds=20,
                metrics=['auc'],
                seed=3407
            )
            
            
            best_rounds = cv_results.shape[0]
            print(f"Best number of iterations: {best_rounds}")
            
            
            evallist = [(dtrain, 'train'), (dval, 'eval')]
            bst = xgb.train(params, dtrain, best_rounds, evallist, verbose_eval=50)
            
         
            bst.save_model(f'xgb_model_fold{fold+1}_class{i}.json')
            xgb_models.append(bst)
            
            
            dval_pred = bst.predict(dval)
            precision, recall, thresholds = precision_recall_curve(val_labels[:, i], dval_pred)
            
            
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
            index = argmax(f1_scores)
            
            if len(thresholds) > index:
                probs_Switcher[i] = thresholds[index]
            else:
                probs_Switcher[i] = 0.5
            
        
        np.save(f'probs_switcher_fold{fold+1}.npy', probs_Switcher)
        
       
        test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels, all_preds = test_xgb(
            test_features, test_labels, xgb_models, probs_Switcher)
        
        # plot ROC curves
        plot_roc_curves(all_labels, all_probs, columns_to_encode)
        
        # plot confusion matrix
        cm = multilabel_confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, columns_to_encode, fold, num_epochs, is_sum=True)

        # generate multilabel confusion matrix
        cm_32x32, precision_32, recall_32, f1_32 = generate_multilabel_confusion_matrix(all_labels, all_preds, fold, num_epochs)
        
        # output results
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
        
        # Execute SHAP analysis
        print("Executing SHAP analysis...")
        

        shap_sample_size = min(100, test_features.shape[0])
        shap_indices = np.random.choice(test_features.shape[0], shap_sample_size, replace=False)
        shap_data = test_features[shap_indices]

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
        

        del train_features, val_features, test_features
        del train_labels, val_labels, test_labels
        torch.cuda.empty_cache()
    
    print("All folds processed successfully!\n")
    
    # Summarize results for all folds
    print("\n===== Summary Results =====")
    

    if os.path.exists('vgg_xgb_results.csv') and os.path.getsize('vgg_xgb_results.csv') > 0:
        try:
           
            with open('vgg_xgb_results.csv', 'r') as f:
                lines = f.readlines()
            
            
            f1_scores = []
            accuracies = []
            precisions = []
            recalls = []
            specificities = []
            macro_aucs = []
            weighted_aucs = []
            
            for i, line in enumerate(lines):
                try:
                   
                    import re
                    
                    
                    f1_match = re.search(r'Test F1: (\d+\.\d+)%', line)
                    if f1_match:
                        f1_scores.append(float(f1_match.group(1)))
                    
                   
                    acc_match = re.search(r'Accuracy: (\d+\.\d+)%', line)
                    if acc_match:
                        accuracies.append(float(acc_match.group(1)))
                    
                   
                    prec_match = re.search(r'Precision: (\d+\.\d+)%', line)
                    if prec_match:
                        precisions.append(float(prec_match.group(1)))
                    
                   
                    recall_match = re.search(r'Recall: (\d+\.\d+)%', line)
                    if recall_match:
                        recalls.append(float(recall_match.group(1)))
                    
                   
                    spec_match = re.search(r'Specificity: (\d+\.\d+)%', line)
                    if spec_match:
                        specificities.append(float(spec_match.group(1)))
                 
                    macro_match = re.search(r'Macro AUC: (\d+\.\d+)', line)
                    if macro_match:
                        macro_aucs.append(float(macro_match.group(1)))
                    
                    
                    weighted_match = re.search(r'Weighted AUC: (\d+\.\d+)', line)
                    if weighted_match:
                        weighted_aucs.append(float(weighted_match.group(1)))
                    
                except Exception as e:
                    print(f"Error processing line {i+1}: {line.strip()} - {e}")
            
            if len(f1_scores) > 0:
                # Calculate mean and standard deviation
                print(f"Average F1 Score: {np.mean(f1_scores):.2f}% ± {np.std(f1_scores):.2f}%")
                print(f"Average Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
                print(f"Average Precision: {np.mean(precisions):.2f}% ± {np.std(precisions):.2f}%")
                print(f"Average Recall: {np.mean(recalls):.2f}% ± {np.std(recalls):.2f}%")
                print(f"Average Specificity: {np.mean(specificities):.2f}% ± {np.std(specificities):.2f}%")
                print(f"Average Macro AUC: {np.mean(macro_aucs):.3f} ± {np.std(macro_aucs):.3f}")
                print(f"Average Weighted AUC: {np.mean(weighted_aucs):.3f} ± {np.std(weighted_aucs):.3f}")
                
               
                with open('vgg_xgb_summary.csv', 'w') as f:
                    f.write("Metric,Mean,Standard Deviation\n")
                    f.write(f"F1 Score,{np.mean(f1_scores):.2f}%,{np.std(f1_scores):.2f}%\n")
                    f.write(f"Accuracy,{np.mean(accuracies):.2f}%,{np.std(accuracies):.2f}%\n")
                    f.write(f"Precision,{np.mean(precisions):.2f}%,{np.std(precisions):.2f}%\n")
                    f.write(f"Recall,{np.mean(recalls):.2f}%,{np.std(recalls):.2f}%\n")
                    f.write(f"Specificity,{np.mean(specificities):.2f}%,{np.std(specificities):.2f}%\n")
                    f.write(f"Macro AUC,{np.mean(macro_aucs):.3f},{np.std(macro_aucs):.3f}\n")
                    f.write(f"Weighted AUC,{np.mean(weighted_aucs):.3f},{np.std(weighted_aucs):.3f}\n")
            else:
                print("No valid result data to summarize")
        except Exception as e:
            print(f"Error summarizing results: {e}")
            print("Please check if the vgg_xgb_results.csv file format is correct")
    else:
        print("Result file does not exist or is empty, cannot summarize results")

if __name__ == '__main__':  
    main()