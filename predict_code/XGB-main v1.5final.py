import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import argmax
from sklearn.metrics import precision_recall_curve
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

# Data standardization and cross-validation split
def split_data_5fold(input_data):
    """
    Split input data into 5 folds, each fold contains:
    - Training set: first 3 folds (60%)
    - Validation set: 4th fold (20%)
    - Test set: 5th fold (20%)
    Ratio adjusted to 3:1:1 (training:validation:test)
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
    

def plot_confusion_matrix(cm, class_names, fold, epoch, is_sum=False):
    """
    Plot confusion matrix image
    
    Parameters:
    cm -- confusion matrix, shape is (n_classes, 2, 2)
    class_names -- list of class names
    fold -- current fold number
    epoch -- current epoch number
    is_sum -- whether it's a cumulative confusion matrix
    """
    n_classes = len(class_names)
    
    
    if is_sum:
        
        y_true = all_targets
        y_pred = all_preds
    else:
       
        y_true = target
        y_pred = preds
    
    # Convert multi-label to multi-class encoding
    # For example, [0,1,0,0,1] is converted to binary "01001", then to decimal 9
    y_true_classes = np.zeros(len(y_true), dtype=int)
    y_pred_classes = np.zeros(len(y_pred), dtype=int)
    
    for i in range(len(y_true)):
        # Check for "no class" case
        if np.sum(y_true[i]) == 0:
            y_true_classes[i] = 2**n_classes  # Use an extra encoding to represent "no class"
        else:
            true_str = ''.join(map(str, y_true[i].astype(int)))
            y_true_classes[i] = int(true_str, 2)
        
        if np.sum(y_pred[i]) == 0:
            y_pred_classes[i] = 2**n_classes  # Use an extra encoding to represent "no class"
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
    
    # No longer normalize, use original counts directly
    # Save original confusion matrix for percentage display
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    percentage_matrix = conf_matrix / row_sums
    
    # Ensure figure directory exists
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
 
    plt.figure(figsize=(12, 10))
    
    # Prepare labels, add special labels for "no class" cases
    xticklabels = []
    yticklabels = []
    for c in unique_classes:
        if c == 2**n_classes:
            xticklabels.append("None")
            yticklabels.append("None")
        else:
            xticklabels.append(bin(c)[2:].zfill(n_classes))
            yticklabels.append(bin(c)[2:].zfill(n_classes))
    
    # seaborn plot hotmap - show original counts, change fmt to '.0f' instead of 'd'
    ax = sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues',
               xticklabels=xticklabels,
               yticklabels=yticklabels)
    
    # percentage annotations
    for i in range(n_unique):
        for j in range(n_unique):
            if conf_matrix[i, j] > 0:
                
                text = ax.text(j + 0.5, i + 0.5, f'({percentage_matrix[i, j]:.1%})',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.title(f'Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
   
    plt.figtext(0.5, 0.01, f'Classes: {", ".join(class_names)} + None', ha='center')
    
    plt.tight_layout()
    
  
    if is_sum:
        plt.savefig(f'figure/xgb_confusion_matrix_fold{fold+1}_sum.png')
    else:
        plt.savefig(f'figure/xgb_confusion_matrix_fold{fold+1}_epoch{epoch+1}.png')
    
    plt.close()
# Threshold calculation function
def Probs_Switcher(probs, labels):
    probs_Switcher = np.array([])
    
    for i in range(labels.shape[1]):
        split_labels_np = labels[:, i]
        split_probs_np = probs[:, i]
        precision, recall, thresholds = precision_recall_curve(split_labels_np, split_probs_np)
        # precision = precision * 0.6
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

# Test function
def test(X_test, y_test, models, probs_Switcher):
    all_probs = np.zeros((X_test.shape[0], len(models)))
    
    # Use corresponding model for prediction of each class
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
    
    # Calculate macro average and weighted average AUC
    macro_auc = np.nanmean(auc_scores)
    weighted_auc = roc_auc_score(y_test, all_probs, average='weighted', multi_class='ovr')
    
    return test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, y_test

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
    plt.savefig(f'figure/XGB_ROC_Curves.png')
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
    # Create results directory
    if not os.path.exists('figure'):
        os.makedirs('figure')
    if not os.path.exists('xgb_models'):
        os.makedirs('xgb_models')
    
    # Record start time
    start_time = time.time()
    
    # Set columns to encode
    columns_to_encode = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
    
    # 加载数据
    features, labels, all_feature_names, Covariates_features = open_excel('DR-CVD DataSet v1.2', columns_to_encode=columns_to_encode)

    # Feature count selection
    features_val = features
    labels_val = labels

    Shap_features = features_val

    Multilay_origin = pd.DataFrame(features_val, columns=all_feature_names)

    labels_DF = pd.DataFrame(labels, columns=columns_to_encode)
    data_DF = pd.DataFrame(features, columns=all_feature_names)
    
    # Data augmentation (same as VGG model)
    from Model import mlsmote
    X_sub, y_sub = mlsmote.get_minority_instace(data_DF, labels_DF)  # Getting minority instance of that datframe
    X_res, y_res = mlsmote.MLSMOTE(X_sub, y_sub, 500)  # Applying MLSMOTE to augment the dataframe

    features = np.concatenate((features, np.float64(X_res)), axis=0)
    labels = np.concatenate((labels, np.float64(y_res)), axis=0)

    # Feature processing
    # Variance filtering
    selector = VarianceThreshold(threshold=0.01)
    features = selector.fit_transform(features)
    features_val = selector.transform(features_val)

    # Update feature names
    mask = selector.get_support()
    all_feature_names = all_feature_names[mask]

    # Standardization
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    for i in range(len(std_f)):
        if std_f[i] == 0:
            std_f[i] = 1e-8

    features = (features - mean_f) / std_f
    features_val = (features_val - mean_f) / (std_f + 1e-8)
    
    # Shape before dimensionality reduction
    print("Before PCA dimensionality reduction, training set shape:", features.shape)
    print("Before PCA dimensionality reduction, validation set shape:", features_val.shape)

    # PCA dimensionality reduction
    pca = PCA(n_components=0.95)
    
    # Filter zero variance again after PCA
    pca_selector = VarianceThreshold(threshold=0.01)
    features = pca_selector.fit_transform(pca.fit_transform(features))
    features_val = pca_selector.transform(pca.transform(features_val))

    # Update PCA feature names
    pca_feature_names = [f'PC{i}' for i in range(1, features.shape[1] + 1)]
    print("After PCA dimensionality reduction, training set shape:", features.shape)
    print("After PCA dimensionality reduction, validation set shape:", features_val.shape)
    
    
    Shap_features = features_val.copy()

    
    folds_data_index = split_data_5fold(features)

    num_x = len(Covariates_features)
    num_folds = len(folds_data_index)


    all_accuracies = [[] for _ in range(num_x)]
    all_precisions = [[] for _ in range(num_x)]
    all_recalls = [[] for _ in range(num_x)]
    all_f1_scores = [[] for _ in range(num_x)]
    all_macro_auc = [[] for _ in range(num_x)]
    all_weighted_auc = [[] for _ in range(num_x)]

    # Let user input which fold numbers to run
    selected_folds_input = '1,2,3,4,5'
    selected_folds = [int(fold.strip()) - 1 for fold in selected_folds_input.split(',')]

    # Record results of all folds
    all_fold_results = []
    
    # Create log file
    log_filename = f'xgb_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(log_filename, 'w') as f:
        f.write("fold,x,test_f1,test_accuracy,test_precision,test_recall,test_specificity,macro_auc,weighted_auc\n")

    for fold, (train_index, validation_index, test_indices) in enumerate(folds_data_index):
        if fold not in selected_folds:
            continue  # Skip unselected folds

        num_classes = len(columns_to_encode) 
        
        for x in range(len(Covariates_features)):
            print(f"\nStarting training Fold {fold+1}, x={x+1}")
            
           
            Validation_size = len(features_val) // 5
            Test_size = Validation_size

            indices = np.arange(len(features_val))
            np.random.shuffle(indices)
            validation_index = indices[(fold + 1) * Validation_size: (fold + 2) * Validation_size] if fold < 4 else indices[4 * Validation_size:]
            test_indices = indices[fold * Test_size: (fold + 1) * Test_size]

            trainX = features[train_index]
            trainY = labels[train_index]
            valX = features_val[validation_index]
            valY = labels_val[validation_index]
            testX = features_val[test_indices]
            testY = labels_val[test_indices]
            
          
            all_preds = np.empty((0, len(columns_to_encode)))
            all_targets = np.empty((0, len(columns_to_encode)))
            
           
            models = []
            probs_val = np.zeros((valY.shape[0], num_classes))
            
            # Add dictionary for plotting learning curves
            train_metrics = {i: {'train_loss': [], 'val_loss': []} for i in range(num_classes)}

            for i in range(num_classes):
                print(f"Training model for class {columns_to_encode[i]}...")
                
                
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'alpha': 0.1,  
                    'lambda': 1.0, 
                    'scale_pos_weight': 2.5,  
                    'seed': 3407
                }
                
                
                dtrain = xgb.DMatrix(trainX, label=trainY[:, i])
                dval = xgb.DMatrix(valX, label=valY[:, i])
                
                
                evallist = [(dtrain, 'train'), (dval, 'eval')]

               
                train_losses = []
                val_losses = []
                
                
                class LossCallback(xgb.callback.TrainingCallback):
                    def after_iteration(self, model, epoch, evals_log):
                        train_loss = evals_log['train']['logloss'][-1]
                        val_loss = evals_log['eval']['logloss'][-1]
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        return False
                
               
                num_round = 350  
                bst = xgb.train(params, dtrain, num_round, evallist, 
                                early_stopping_rounds=15, verbose_eval=50,callbacks=[LossCallback()])

                
              
                train_metrics[i]['train_loss'] = train_losses
                train_metrics[i]['val_loss'] = val_losses
                
           
                bst.save_model(f'xgb_models/xgb_model_fold{fold+1}_class{i}_x{x+1}.json')
                
              
                models.append(bst)
                
              
                probs_val[:, i] = bst.predict(xgb.DMatrix(valX))

           
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
                plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
                plt.axvline(x=bst.best_iteration, color='r', linestyle='--', 
                           label=f'Best Iteration: {bst.best_iteration}')
                plt.xlabel('Boosting Iterations')
                plt.ylabel('Log Loss')
                plt.title(f'XGBoost Learning Curve - {columns_to_encode[i]} (Fold {fold+1}, x={x+1})')
                plt.legend()
                plt.grid(True)
                
           
                if not os.path.exists('figure'):
                    os.makedirs('figure')
                    
                plt.savefig(f'figure/xgb_learning_curve_fold{fold+1}_class{i}_x{x+1}.png')
                plt.close()
            
            # Calculate thresholds on validation set
            probs_Switcher = Probs_Switcher(probs_val, valY)
            print(f"Thresholds calculated on validation set: {probs_Switcher}")
            

            test_f1, test_accuracy, test_precision, test_recall, test_specificity, macro_auc, weighted_auc, auc_scores, all_probs, all_labels = test(testX, testY, models, probs_Switcher)
            
           
            plt.figure(figsize=(15, 10))
            
        
            rows = (num_classes + 1) // 2 
            cols = 2 if num_classes > 1 else 1
            
            for i in range(num_classes):
                plt.subplot(rows, cols, i + 1)
                plt.plot(range(1, len(train_metrics[i]['train_loss']) + 1), 
                         train_metrics[i]['train_loss'], label='Training Loss')
                plt.plot(range(1, len(train_metrics[i]['val_loss']) + 1), 
                         train_metrics[i]['val_loss'], label='Validation Loss')
                plt.axvline(x=models[i].best_iteration, color='r', linestyle='--', 
                           label=f'Best: {models[i].best_iteration}')
                plt.title(f'{columns_to_encode[i]}')
                plt.xlabel('Iterations')
                plt.ylabel('Log Loss')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'figure/xgb_all_learning_curves_fold{fold+1}_x{x+1}.png')
            plt.close()

          
            fold_result = {
                'fold': fold + 1,
                'x': x + 1,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_specificity': test_specificity,
                'macro_auc': macro_auc,
                'weighted_auc': weighted_auc,
                'auc_scores': auc_scores
            }
            all_fold_results.append(fold_result)
            
            # record results to log file
            with open(log_filename, 'a') as f:
                f.write(f"{fold+1},{x+1},{test_f1:.2f},{test_accuracy:.2f},{test_precision:.2f},{test_recall:.2f},{test_specificity:.2f},{macro_auc:.3f},{weighted_auc:.3f}\n")
            
           
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
            
          
            with open('XGB_Test.csv', 'a') as f:
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
            
         
            cm = multilabel_confusion_matrix(all_labels, np.float64(all_probs >= probs_Switcher))
            target = all_labels
            preds = np.float64(all_probs >= probs_Switcher)
            plot_confusion_matrix(cm, columns_to_encode, fold, 0)
            
      
            plot_roc_curves(all_labels, all_probs, columns_to_encode)
            
      
            all_accuracies[x].append(test_accuracy)
            all_precisions[x].append(test_precision)
            all_recalls[x].append(test_recall)
            all_f1_scores[x].append(test_f1)
            all_macro_auc[x].append(macro_auc)
            all_weighted_auc[x].append(weighted_auc)
    
  
    if len(selected_folds) == 5:
    
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_accuracies[x], label=f'Accuracy (x={x + 1})', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('Accuracy (%)')
        plt.title('XGBoost: Comparison of Accuracy for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_Comparison_of_Accuracy.png')
        plt.close()

  
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_precisions[x], label=f'Precision (x={x + 1})', marker='s')
        plt.xlabel('Fold Number')
        plt.ylabel('Precision (%)')
        plt.title('XGBoost: Comparison of Precision for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_Comparison_of_Precision.png')
        plt.close()

  
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_recalls[x], label=f'Recall (x={x + 1})', marker='^')
        plt.xlabel('Fold Number')
        plt.ylabel('Recall (%)')
        plt.title('XGBoost: Comparison of Recall for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_Comparison_of_Recall.png')
        plt.close()

   
        plt.figure(figsize=(12, 8))
        for x in range(num_x):
            plt.plot(range(1, num_folds + 1), all_f1_scores[x], label=f'F1 Score (x={x + 1})', marker='D')
        plt.xlabel('Fold Number')
        plt.ylabel('F1 Score (%)')
        plt.title('XGBoost: Comparison of F1 Score for Different x Values Across Folds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, num_folds + 1))
        plt.tight_layout()
        plt.savefig(f'figure/XGB_Comparison_of_F1_Score.png')
        plt.close()
    
    #

    # SHAP analysis
    print("\nStarting SHAP analysis...")
    
    
    analysis_fold = 0  
    if analysis_fold in selected_folds:
        # Use PCA feature names
        pca_feature_names = [f'PC{i}' for i in range(1, features_val.shape[1] + 1)]
        origin = pd.DataFrame(features_val, columns=pca_feature_names)
        COLUMNS = origin.columns
        num_classes = labels_val.shape[1]
        
        # Use SHAP to downsample background data
        K = 16
        background_data = shap.kmeans(features_val, K)  # K-means clustering, select K background data points
        
        # randomly select 3501 samples from validation set for SHAP analysis
        selected_indices = np.random.choice(len(features_val), min(3501, len(features_val)), replace=True)
        selected_features_val = features_val[selected_indices]
        
        ALL_shap_exp = {}
        ALL_top_features = {}
        
        
        models = []
        for i in range(num_classes):
            model_path = f'xgb_models/xgb_model_fold{analysis_fold+1}_class{i}_x{0+1}.json'
            if os.path.exists(model_path):
                model = xgb.Booster()
                model.load_model(model_path)
                models.append(model)
            else:
                print(f"Warning: Model file {model_path} does not exist")
                continue
        
        if len(models) == num_classes:
            # Perform SHAP analysis for each class
            for class_idx in range(num_classes):
                print(f"Using SHAP to evaluate feature importance for class {columns_to_encode[class_idx]}...")
                
                
                explainer = shap.TreeExplainer(models[class_idx])
                
               
                shap_values = explainer.shap_values(xgb.DMatrix(selected_features_val))
                
                
                shap_exp = shap.Explanation(
                    values=shap_values, 
                    data=selected_features_val,
                    feature_names=pca_feature_names,
                    base_values=np.full(len(selected_features_val), explainer.expected_value)
                )
                
               
                ALL_shap_exp[columns_to_encode[class_idx]] = shap_exp

                # Plot SHAP summary plot - using PCA features
                plt.figure(figsize=(12, 8))
                shap.plots.bar(shap_exp, max_display=16, show=False)
                plt.title(f'XGBoost SHAP Summary Plot (PCA Features) for {columns_to_encode[class_idx]}')
                plt.tight_layout()
                plt.savefig(f'figure/XGB_SHAP_Summary_PCA_{columns_to_encode[class_idx]}.png')
                plt.close()
                
                # plot beeswarm plot
                plt.figure(figsize=(12, 8))
                shap.plots.beeswarm(shap_exp, max_display=16, show=False)
                plt.title(f'XGBoost SHAP Beeswarm Plot (PCA Features) for {columns_to_encode[class_idx]}')
                plt.tight_layout()
                plt.savefig(f'figure/XGB_SHAP_Beeswarm_PCA_{columns_to_encode[class_idx]}.png')
                plt.close()
                
                # feature mapping to original features                
                pca_components = pca.components_
                
               
                feature_importance = np.zeros(len(all_feature_names))
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                
                for j, importance in enumerate(mean_abs_shap):
                    if j < len(pca_components):  
                        
                        feature_importance += importance * np.abs(pca_components[j])
                
               
                original_importance_df = pd.DataFrame({
                    'Feature': all_feature_names,
                    'Importance': feature_importance
                })
                original_importance_df = original_importance_df.sort_values('Importance', ascending=False)
                
               
                plt.figure(figsize=(15, 6))
                top_features = original_importance_df.head(min(64, len(original_importance_df)))
                plt.bar(top_features['Feature'], top_features['Importance'])
                plt.xlabel('Features')
                plt.ylabel('Mapped Importance')
                plt.title(f'XGBoost Mapped Feature Importance for {columns_to_encode[class_idx]}')
                plt.xticks(rotation=45, fontsize=6)
                plt.tight_layout()
                plt.savefig(f'figure/XGB_Mapped_Feature_Importance_{columns_to_encode[class_idx]}.png')
                plt.close()
                
               
                ALL_top_features[columns_to_encode[class_idx]] = top_features['Feature'].tolist()
                
               
                importance = models[class_idx].get_score(importance_type='gain')
                importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                
                plt.figure(figsize=(15, 6))
                xgb_top_features = importance_df.head(min(64, len(importance_df)))
                plt.bar(xgb_top_features['Feature'], xgb_top_features['Importance'])
                plt.xlabel('PCA Features')
                plt.ylabel('Importance')
                plt.title(f'XGBoost Native Feature Importance for {columns_to_encode[class_idx]}')
                plt.xticks(rotation=45, fontsize=6)
                plt.tight_layout()
                plt.savefig(f'figure/XGB_Native_Feature_Importance_{columns_to_encode[class_idx]}.png')
                plt.close()
                
              
                ALL_top_features[columns_to_encode[class_idx]] = top_features['Feature'].tolist()
                
                
            # Stratified analysis
            print("\nStarting stratified analysis...")
            stratify_variable = ['OPDURL4', 'RIAGENDR', 'RIDRETH1']  # Disease severity, gender, and race respectively
            
            
            multilay_df = pd.DataFrame(Multilay_origin)
            
            for i in stratify_variable:
                
                strata = multilay_df[i].unique()
                
                for class_idx in range(num_classes):
                    if columns_to_encode[class_idx] not in ALL_top_features:
                        continue
                        
                    top_features = ALL_top_features[columns_to_encode[class_idx]]
                    
                    for stratum in strata:
                        
                        stratum_mask = multilay_df[i] == stratum
                        stratum_indices = np.where(stratum_mask)[0]
                        
                        
                        common_indices = np.intersect1d(stratum_indices, selected_indices)
                        
                        if len(common_indices) == 0:
                            print(f"Warning: No data found in stratum {i}={stratum}")
                            continue
                            
                        
                        bool_tf = np.isin(selected_indices, common_indices)
                        
                        try:
                            
                            shap_exp = ALL_shap_exp[columns_to_encode[class_idx]]
                            
                            
                            filtered_shap_values = shap_exp.values[bool_tf]
                            filtered_data = shap_exp.data[bool_tf]
                            filtered_base_values = shap_exp.base_values[bool_tf]
                            
                            
                            filtered_shap_exp = shap.Explanation(
                                values=filtered_shap_values,
                                data=filtered_data,
                                feature_names=shap_exp.feature_names,
                                base_values=filtered_base_values
                            )
                            
                            
                            new_shap_obj = new_shap_values(shap_exp, bool_tf=bool_tf, method='mean')
                            
                            # plot SHAP summary plot
                            plt.figure(figsize=(12, 8))
                            shap.plots.waterfall(new_shap_obj.get_explanation(), show=False)
                            plt.title(f'XGBoost SHAP Waterfall Plot for {columns_to_encode[class_idx]}, {i}={stratum}')
                            plt.tight_layout()
                            plt.savefig(f'figure/XGB_SHAP_Waterfall_{columns_to_encode[class_idx]}_{i}_{stratum}.png')
                            plt.close()

                            # plot SHAP beeswarm plot
                            plt.figure(figsize=(12, 8))
                            shap.plots.beeswarm(filtered_shap_exp, max_display=16, show=False)
                            plt.title(f'XGBoost SHAP Beeswarm Plot for {columns_to_encode[class_idx]}, {i}={stratum}')
                            plt.tight_layout()
                            plt.savefig(f'figure/XGB_SHAP_Beeswarm_{columns_to_encode[class_idx]}_{i}_{stratum}.png')
                            plt.close()
                            
                        except Exception as e:
                            print(f"Error drawing SHAP plot, class {columns_to_encode[class_idx]}, stratum {i}={stratum}, error message: {e}")
    
    # Calculate running time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nXGBoost model training and evaluation completed!")
    print(f"Total running time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
    
    # Save overall results
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('XGB_all_results.csv', index=False)
    
    print("\nAll results have been saved to XGB_all_results.csv")