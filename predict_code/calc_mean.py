import pandas as pd
import ast

# 读取CSV文件
df = pd.read_csv('XGB_all_results.csv')
df.columns = df.columns.str.strip()  # 新增：去除列名前后空格

# 修改列选择方式（原2:-1改为明确列名）
numeric_cols = ['test_f1', 'test_accuracy', 'test_precision', 
               'test_recall', 'test_specificity', 'macro_auc', 'weighted_auc']
# 计算各数值列均值（排除fold和x列）
mean_values = df.iloc[:, 2:-1].mean().to_dict()

# 特殊处理auc_scores列（每个位置单独求平均）
auc_scores = [ast.literal_eval(s) for s in df['auc_scores']]
mean_auc_scores = [
    round(sum([s[i] for s in auc_scores])/len(auc_scores), 4) 
    for i in range(5)
]

# 创建均值行
mean_row = {
    'fold': 'mean',
    'x': '',
    'test_f1': round(mean_values['test_f1'], 4),
    'test_accuracy': round(mean_values['test_accuracy'], 4),
    'test_precision': round(mean_values['test_precision'], 4),
    'test_recall': round(mean_values['test_recall'], 4),
    'test_specificity': round(mean_values['test_specificity'], 4),
    'macro_auc': round(mean_values['macro_auc'], 4),
    'weighted_auc': round(mean_values['weighted_auc'], 4),
    'auc_scores': f'[{", ".join(map(str, mean_auc_scores))}]'
}

# 添加均值行
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

# 保存结果（保持原有格式）
df.to_csv('XGB_all_results.csv', index=False, float_format='%.4f')