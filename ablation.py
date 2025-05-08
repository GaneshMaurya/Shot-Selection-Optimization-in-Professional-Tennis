import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
import os

start_time = time.time()

print("Loading dataset...")
data = pd.read_csv('tennis_points.csv')
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")

# Drop rows with NaN in any column
data = data.dropna()
print(f"After cleaning, dataset has {data.shape[0]} rows")

# Columns to drop (identifiers or redundant)
columns_to_drop = [
    'match_id', 'player1', 'player2', 'tournament', 'score',
    'serve_full', 'first_serve_full', 'shot_1_full', 'shot_2_full',
    'shot_3_full', 'shot_4_full', 'shot_5_full', 'last_shot_full'
]

# Drop irrelevant columns
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
print(f"After dropping irrelevant columns, dataset has {data.shape[1]} columns")

# Identify categorical columns
categorical_columns = [
    'surface', 'serve_type', 'first_serve_type', 'shot_1_type', 'shot_2_type',
    'shot_3_type', 'shot_4_type', 'shot_5_type', 'last_shot_type',
    'serve_direction', 'shot_1_direction', 'shot_2_direction', 'shot_3_direction',
    'shot_4_direction', 'shot_5_direction', 'last_shot_direction'
]

# Label encode categorical columns
label_encoders = {}
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))  # Convert to string to handle mixed types
        label_encoders[col] = le
print("Categorical columns encoded")

# Define feature groups based on the ablation study
feature_groups = {
    'shot_direction': [col for col in data.columns if 'direction' in col.lower()],
    'shot_depth': [col for col in data.columns if 'depth' in col.lower()],
    'shot_type': [col for col in data.columns if 'type' in col.lower()],
    'rally_length': [col for col in data.columns if 'rally_length' in col.lower()],
    'server': [col for col in data.columns if 'server' in col.lower() or 'is_second_serve' in col.lower()],
    'shot_sequence': [col for col in data.columns if 'shot_' in col.lower()]
}

# Check unique values in outcome
print("Unique values in 'outcome' column:", data['outcome'].unique())

# Define features and target
X = data.drop(columns=['outcome'])
y = data['outcome']

# Map outcomes to numerical labels
outcome_mapping = {
    'Forced Error': 0,
    'Unforced Error': 1,
    'Winner': 2,
    'Ace': 3,
    'Double Fault': 4
}
y = y.map(outcome_mapping)

# Check for NaNs in y
if y.isna().any():
    print("NaNs found in y after mapping. Rows with NaNs:")
    print(data[y.isna()][['outcome']])
    data = data[~y.isna()]
    X = X[~y.isna()]
    y = y[~y.isna()]
    print(f"After removing NaNs in y, dataset has {data.shape[0]} rows")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Save test dataset
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('model_outputs/test_dataset.csv', index=False)
print("Testing dataset saved to model_outputs/test_dataset.csv")

# Define XGBoost model with best parameters from GridSearchCV
model_params = {
    'objective': 'multi:softprob',
    'max_depth': 7,
    'n_estimators': 300,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'merror',  # For multiclass classification
    'random_state': 42
}

# Initialize results dictionary
ablation_results = {
    'Model Configuration': [],
    'Test Accuracy': [],
    'Change': []
}

# Train and evaluate full model
print("Training full XGBoost model...")
full_model = XGBClassifier(**model_params)
full_model.fit(X_train, y_train)
y_pred_full = full_model.predict(X_test)
full_accuracy = accuracy_score(y_test, y_pred_full)
print(f"Full model accuracy: {full_accuracy:.4f}")
ablation_results['Model Configuration'].append('Full Model (all features)')
ablation_results['Test Accuracy'].append(f"{full_accuracy*100:.1f}")
ablation_results['Change'].append('-')

# Perform ablation studies
for group_name, group_features in feature_groups.items():
    if group_name == 'shot_sequence':
        # Using only shot sequence features
        X_train_subset = X_train[group_features]
        X_test_subset = X_test[group_features]
        config_name = 'Using Only Shot Sequence Features'
    else:
        # Remove the feature group
        X_train_subset = X_train.drop(columns=group_features)
        X_test_subset = X_test.drop(columns=group_features)
        config_name = f"Without {group_name.replace('_', ' ').title()} Features"
    
    print(f"Training model for {config_name}...")
    model = XGBClassifier(**model_params)
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    accuracy = accuracy_score(y_test, y_pred)
    change = (accuracy - full_accuracy) * 100
    
    ablation_results['Model Configuration'].append(config_name)
    ablation_results['Test Accuracy'].append(f"{accuracy*100:.1f}")
    ablation_results['Change'].append(f"{change:+.1f}")
    print(f"{config_name} accuracy: {accuracy:.4f}, Change: {change:+.1f}")

# Save ablation results to CSV
os.makedirs('model_outputs', exist_ok=True)
results_df = pd.DataFrame(ablation_results)
results_df.to_csv('model_outputs/ablation_study_results.csv', index=False)
print("Ablation study results saved to model_outputs/ablation_study_results.csv")

# Calculate and print total execution time
execution_time = time.time() - start_time
minutes, seconds = divmod(execution_time, 60)
print(f"Total execution time: {int(minutes)}:{seconds:.6f}")

print("\nLaTeX Table for Ablation Study Results:")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Ablation Study Results}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Model Configuration} & \\textbf{Test Accuracy} & \\textbf{Change} \\\\")
print("\\midrule")
for i in range(len(ablation_results['Model Configuration'])):
    print(f"{ablation_results['Model Configuration'][i]} & {ablation_results['Test Accuracy'][i]} & {ablation_results['Change'][i]} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
