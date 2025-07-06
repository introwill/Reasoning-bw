import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def xgboost_classify(features, labels, class_names):
    """
    XGBoost多标签分类函数
    """
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'seed': 3407
    }
    
    metrics = {}
    for i, name in enumerate(class_names):
        dtrain = xgb.DMatrix(features, label=labels[:,i])
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        preds = model.predict(dtrain)
        metrics[name] = {
            'accuracy': accuracy_score(labels[:,i], preds>0.5),
            'f1_score': f1_score(labels[:,i], preds>0.5),
            'roc_auc': roc_auc_score(labels[:,i], preds)
        }
    
    print("\nXGBoost分类结果:")
    for name in class_names:
        print(f"{name}:")
        print(f"  Accuracy: {metrics[name]['accuracy']:.4f}")
        print(f"  F1 Score: {metrics[name]['f1_score']:.4f}")
        print(f"  ROC AUC: {metrics[name]['roc_auc']:.4f}")
    
    return metrics