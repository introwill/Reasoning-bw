import pandas as pd
import ast

# Read CSV file
df = pd.read_csv('XGB_all_results.csv')
df.columns = df.columns.str.strip()  # Added: remove leading/trailing spaces from column names

# Modified column selection method (changed from original 2:-1 to explicit column names)
numeric_cols = ['test_f1', 'test_accuracy', 'test_precision', 
               'test_recall', 'test_specificity', 'macro_auc', 'weighted_auc']
# Calculate mean values for numeric columns (excluding fold and x columns)
mean_values = df.iloc[:, 2:-1].mean().to_dict()

# Special handling for auc_scores column (calculate average for each position separately)
auc_scores = [ast.literal_eval(s) for s in df['auc_scores']]
mean_auc_scores = [
    round(sum([s[i] for s in auc_scores])/len(auc_scores), 4) 
    for i in range(5)
]

# Create mean row
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

# Add mean row
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

# Save results (maintain original format)
df.to_csv('XGB_all_results.csv', index=False, float_format='%.4f')