import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import pickle
import logging
import warnings
import gdown
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import uuid
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(filename='training_log_extended.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def log_and_print(message):
    logger.info(message)
    print(message)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_and_print(f"Using device: {device}")

# Directory structure setup
BASE_OUTPUT_DIR = 'model_outputs'
MODEL_TYPES = ['transformer', 'mlp', 'lstm', 'xgboost']
for model_type in MODEL_TYPES:
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, model_type), exist_ok=True)

# Load and preprocess data
def load_data(file_path):
    log_and_print("Loading dataset...")
    try:
        df = pd.read_csv(file_path)
        log_and_print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        log_and_print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    try:
        # Standardize outcome labels
        outcome_mapping = {
            'Winner': 'Winner', 'Forced Error': 'Forced Error', 'Unforced Error': 'Unforced Error',
            'Ace': 'Winner', 'Double Fault': 'Unforced Error'
        }
        df['outcome'] = df['outcome'].map(outcome_mapping)
        df = df.dropna(subset=['outcome'])
        log_and_print(f"After cleaning, dataset has {len(df)} rows")

        # Select features
        feature_columns = [
            'serve_type', 'serve_direction', 'serve_depth', 'is_second_serve',
            'rally_length', 'shot_1_type', 'shot_1_direction', 'shot_1_depth',
            'shot_2_type', 'shot_2_direction', 'shot_2_depth',
            'shot_3_type', 'shot_3_direction', 'shot_3_depth',
            'shot_4_type', 'shot_4_direction', 'shot_4_depth',
            'shot_5_type', 'shot_5_direction', 'shot_5_depth',
            'last_shot_type', 'last_shot_direction', 'last_shot_depth'
        ]
        df_features = df[feature_columns].copy()
        df_target = df['outcome']

        # Handle missing values
        for col in df_features.columns:
            if df_features[col].dtype == 'object':
                df_features[col] = df_features[col].fillna('None')
            else:
                df_features[col] = df_features[col].fillna(0)

        # Convert categorical columns to strings
        categorical_columns = [col for col in df_features.columns if df_features[col].dtype == 'object']
        for col in categorical_columns:
            df_features[col] = df_features[col].astype(str)
        
        # Encode categorical features
        vocab_sizes = {}
        encoders = {}
        for col in categorical_columns:
            encoders[col] = LabelEncoder()
            df_features[col] = encoders[col].fit_transform(df_features[col])
            vocab_sizes[col] = len(encoders[col].classes_)

        # Encode target
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df_target)
        log_and_print(f"Class distribution: {dict(zip(target_encoder.classes_, np.bincount(y)))}")

        return df_features, y, categorical_columns, vocab_sizes, encoders, target_encoder
    except Exception as e:
        log_and_print(f"Error in preprocessing: {str(e)}")
        raise

# Custom Dataset
class TennisDataset(Dataset):
    def __init__(self, X, y, categorical_columns):
        self.X = X
        self.y = y
        self.categorical_columns = categorical_columns
        self.numerical_columns = [col for col in X.columns if col not in categorical_columns]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        try:
            cat_features = torch.tensor([self.X[col].iloc[idx] for col in self.categorical_columns], dtype=torch.long)
            num_features = torch.tensor([self.X[col].iloc[idx] for col in self.numerical_columns], dtype=torch.float32)
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return cat_features, num_features, label
        except Exception as e:
            log_and_print(f"Error in dataset item retrieval: {str(e)}")
            raise

# Transformer Model
class TennisTransformer(nn.Module):
    def __init__(self, vocab_sizes, num_numerical, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=3, dropout=0.3):
        super(TennisTransformer, self).__init__()
        self.d_model = d_model
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, d_model) for col, vocab_size in vocab_sizes.items()
        })
        self.numerical_layer = nn.Linear(num_numerical, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cat_features, num_features):
        try:
            cat_embeds = [self.embeddings[col](cat_features[:, i]) for i, col in enumerate(self.embeddings.keys())]
            cat_embeds = torch.stack(cat_embeds, dim=1)
            num_embeds = self.numerical_layer(num_features).unsqueeze(1)
            x = torch.cat([cat_embeds, num_embeds], dim=1).float()
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)
            x = self.dropout(x)
            return self.fc(x).float()
        except Exception as e:
            log_and_print(f"Error in transformer forward: {str(e)}")
            raise

# MLP Model
class TennisMLP(nn.Module):
    def __init__(self, vocab_sizes, num_numerical, hidden_dims=[128, 64], num_classes=3, dropout=0.3):
        super(TennisMLP, self).__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, 16) for col, vocab_size in vocab_sizes.items()
        })
        total_input_dim = sum([16 for _ in vocab_sizes]) + num_numerical
        layers = []
        prev_dim = total_input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, num_classes)

    def forward(self, cat_features, num_features):
        try:
            cat_embeds = [self.embeddings[col](cat_features[:, i]) for i, col in enumerate(self.embeddings.keys())]
            cat_embeds = torch.cat(cat_embeds, dim=1)
            x = torch.cat([cat_embeds, num_features], dim=1).float()
            x = self.mlp(x)
            return self.fc(x).float()
        except Exception as e:
            log_and_print(f"Error in MLP forward: {str(e)}")
            raise

# LSTM Model
class TennisLSTM(nn.Module):
    def __init__(self, vocab_sizes, num_numerical, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3):
        super(TennisLSTM, self).__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, 16) for col, vocab_size in vocab_sizes.items()
        })
        self.lstm = nn.LSTM(input_size=16*len(vocab_sizes) + num_numerical, 
                           hidden_size=hidden_dim, 
                           num_layers=num_layers, 
                           batch_first=True, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cat_features, num_features):
        try:
            cat_embeds = [self.embeddings[col](cat_features[:, i]) for i, col in enumerate(self.embeddings.keys())]
            cat_embeds = torch.cat(cat_embeds, dim=1)
            x = torch.cat([cat_embeds, num_features], dim=1).float().unsqueeze(1)
            _, (hn, _) = self.lstm(x)
            x = self.dropout(hn[-1])
            return self.fc(x).float()
        except Exception as e:
            log_and_print(f"Error in LSTM forward: {str(e)}")
            raise

# Training function with type checking
def train_model(model, train_loader, val_loader, model_type, num_epochs=50, patience=5, class_weights=None):
    start_time = datetime.now()
    try:
        if class_weights is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for cat_features, num_features, labels in train_loader:
                cat_features, num_features, labels = cat_features.to(device), num_features.to(device), labels.to(device)
                
                # Type checking
                if num_features.dtype != torch.float32:
                    num_features = num_features.float()
                
                optimizer.zero_grad()
                outputs = model(cat_features, num_features)
                if outputs.dtype != torch.float32:
                    outputs = outputs.float()
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for cat_features, num_features, labels in val_loader:
                    cat_features, num_features, labels = cat_features.to(device), num_features.to(device), labels.to(device)
                    if num_features.dtype != torch.float32:
                        num_features = num_features.float()
                    outputs = model(cat_features, num_features)
                    if outputs.dtype != torch.float32:
                        outputs = outputs.float()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * len(labels)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            log_and_print(f"[{model_type}] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'{BASE_OUTPUT_DIR}/{model_type}/best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log_and_print(f"[{model_type}] Early stopping triggered")
                    break

        training_time = (datetime.now() - start_time).total_seconds()
        return train_losses, val_losses, training_time
    except Exception as e:
        log_and_print(f"Error in training {model_type}: {str(e)}")
        raise

# Evaluation function
def evaluate_model(model, loader, target_encoder, model_type, model_class='nn'):
    try:
        model.eval()
        y_true, y_pred, y_scores = [], [], []
        
        if model_class == 'nn':
            with torch.no_grad():
                for cat_features, num_features, labels in loader:
                    cat_features, num_features = cat_features.to(device), num_features.to(device)
                    if num_features.dtype != torch.float32:
                        num_features = num_features.float()
                    outputs = model(cat_features, num_features)
                    if outputs.dtype != torch.float32:
                        outputs = outputs.float()
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(labels.numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_scores.append(torch.softmax(outputs, dim=1).cpu().numpy())
        else:  # XGBoost
            for cat_features, num_features, labels in loader:
                X_batch = np.hstack([cat_features.numpy(), num_features.numpy()])
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)
                y_true.extend(labels.numpy())
                y_pred.extend(preds)
                y_scores.append(probs)

        y_scores = np.vstack(y_scores)
        y_true_bin = np.eye(len(target_encoder.classes_))[y_true]

        # Classification report
        report = classification_report(y_true, y_pred, target_names=target_encoder.classes_, output_dict=True)
        log_and_print(f"\n[{model_type}] Classification Report:")
        log_and_print(classification_report(y_true, y_pred, target_names=target_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_type}')
        plt.savefig(f'{BASE_OUTPUT_DIR}/{model_type}/confusion_matrix.png')
        plt.close()

        # Precision-Recall curves
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(target_encoder.classes_):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            plt.plot(recall, precision, label=f'{class_name} (AP={np.mean(precision):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_type}')
        plt.legend()
        plt.savefig(f'{BASE_OUTPUT_DIR}/{model_type}/precision_recall_curve.png')
        plt.close()

        # ROC curves
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(target_encoder.classes_):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_type}')
        plt.legend()
        plt.savefig(f'{BASE_OUTPUT_DIR}/{model_type}/roc_curve.png')
        plt.close()

        return report, cm, y_true, y_pred, y_scores
    except Exception as e:
        log_and_print(f"Error in evaluation {model_type}: {str(e)}")
        raise

# Learning curve analysis
def learning_curve_analysis(X_train, y_train, X_test, y_test, categorical_columns, vocab_sizes, num_numerical, target_encoder):
    try:
        percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
        results = {model_type: {'accuracy': [], 'f1': [], 'training_time': []} for model_type in ['transformer', 'mlp', 'lstm']}
        
        for perc in percentages:
            log_and_print(f"\nTraining with {perc*100}% of training data")
            n_samples = int(len(X_train) * perc)
            X_subset = X_train.iloc[:n_samples]
            y_subset = y_train[:n_samples]
            
            train_dataset = TennisDataset(X_subset, y_subset, categorical_columns)
            test_dataset = TennisDataset(X_test, y_test, categorical_columns)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            # Compute class weights
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_subset), y=y_subset)
            
            for model_type in ['transformer', 'mlp', 'lstm']:
                if model_type == 'transformer':
                    model = TennisTransformer(vocab_sizes, num_numerical, num_classes=len(target_encoder.classes_)).to(device)
                elif model_type == 'mlp':
                    model = TennisMLP(vocab_sizes, num_numerical, num_classes=len(target_encoder.classes_)).to(device)
                else:  # lstm
                    model = TennisLSTM(vocab_sizes, num_numerical, num_classes=len(target_encoder.classes_)).to(device)

                train_losses, val_losses, training_time = train_model(model, train_loader, test_loader, model_type, class_weights=class_weights)
                report, _, _, _, _ = evaluate_model(model, test_loader, target_encoder, model_type)
                
                results[model_type]['accuracy'].append(report['accuracy'])
                results[model_type]['f1'].append(report['weighted avg']['f1-score'])
                results[model_type]['training_time'].append(training_time)

        # Plot learning curves
        plt.figure(figsize=(12, 5))
        for i, metric in enumerate(['accuracy', 'f1']):
            plt.subplot(1, 2, i+1)
            for model_type in results:
                plt.plot([p*100 for p in percentages], results[model_type][metric], label=model_type, marker='o')
            plt.xlabel('Training Data Percentage')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} vs Training Data Size')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{BASE_OUTPUT_DIR}/learning_curves.png')
        plt.close()

        # Plot training time
        plt.figure(figsize=(8, 6))
        for model_type in results:
            plt.plot([p*100 for p in percentages], results[model_type]['training_time'], label=model_type, marker='o')
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time vs Training Data Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{BASE_OUTPUT_DIR}/training_time.png')
        plt.close()

        return results
    except Exception as e:
        log_and_print(f"Error in learning curve analysis: {str(e)}")
        raise

# Precision-Recall imbalance analysis
def analyze_precision_recall(y_true, y_pred, y_scores, target_encoder, model_type):
    try:
        y_true_bin = np.eye(len(target_encoder.classes_))[y_true]
        
        # Per-class precision-recall analysis
        analysis = {}
        for i, class_name in enumerate(target_encoder.classes_):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            analysis[class_name] = {
                'precision': np.mean(precision),
                'recall': np.mean(recall),
                'class_freq': np.sum(y_true_bin[:, i]) / len(y_true)
            }
        
        # Visualize precision-recall differences
        plt.figure(figsize=(10, 6))
        classes = list(analysis.keys())
        precisions = [analysis[c]['precision'] for c in classes]
        recalls = [analysis[c]['recall'] for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, precisions, width, label='Precision')
        plt.bar(x + width/2, recalls, width, label='Recall')
        plt.xticks(x, classes)
        plt.ylabel('Score')
        plt.title(f'Precision vs Recall by Class - {model_type}')
        plt.legend()
        plt.savefig(f'{BASE_OUTPUT_DIR}/{model_type}/precision_recall_comparison.png')
        plt.close()
        
        # Log analysis
        log_and_print(f"\n[{model_type}] Precision-Recall Analysis:")
        for class_name, metrics in analysis.items():
            log_and_print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Class Freq={metrics['class_freq']:.4f}")

        return analysis
    except Exception as e:
        log_and_print(f"Error in precision-recall analysis: {str(e)}")
        raise

# XGBoost implementation with hyperparameter tuning
def train_xgboost(X_train, y_train, X_test, y_test, categorical_columns, target_encoder):
    try:
        X_train_np = np.hstack([X_train[cat].values.reshape(-1, 1) for cat in X_train.columns])
        X_test_np = np.hstack([X_test[cat].values.reshape(-1, 1) for cat in X_test.columns])
        
        # Hyperparameter tuning
        param_dist = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb = XGBClassifier(random_state=42)
        search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, cv=3, scoring='f1_weighted', n_jobs=-1)
        search.fit(X_train_np, y_train)
        
        best_model = search.best_estimator_
        log_and_print(f"Best XGBoost parameters: {search.best_params_}")
        
        # Feature importance
        feature_names = X_train.columns
        importance = best_model.feature_importances_
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(importance)[::-1]
        plt.bar(range(len(importance)), importance[sorted_idx])
        plt.xticks(range(len(importance)), feature_names[sorted_idx], rotation=45)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{BASE_OUTPUT_DIR}/xgboost/feature_importance.png')
        plt.close()
        
        # Log feature importance analysis
        log_and_print("\nTop 5 important features:")
        for idx in sorted_idx[:5]:
            log_and_print(f"{feature_names[idx]}: {importance[idx]:.4f}")
        
        return best_model
    except Exception as e:
        log_and_print(f"Error in XGBoost training: {str(e)}")
        raise

# Main execution
def main():
    start_time = datetime.now()
    CONFIG = {
        'log_metrics_file': '/kaggle/working/metrics.csv',
        'gdrive_file_id': '16IH03soaKK15gvOO4t84ohCP-n2abCYV',
        'csv_file_name': 'dataset_subset.csv',
    }

    try:
        # Load and preprocess data
        file_id = CONFIG['gdrive_file_id']
        output_path = os.path.join('/kaggle/working', CONFIG['csv_file_name'])
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
        df = load_data(output_path)
        X, y, categorical_columns, vocab_sizes, encoders, target_encoder = preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        log_and_print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Create datasets and dataloaders
        train_dataset = TennisDataset(X_train, y_train, categorical_columns)
        test_dataset = TennisDataset(X_test, y_test, categorical_columns)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        num_numerical = len([col for col in X.columns if col not in categorical_columns])
        
        # Train and evaluate neural network models
        nn_models = {
            'transformer': TennisTransformer(vocab_sizes, num_numerical, num_classes=len(target_encoder.classes_)).to(device),
            'mlp': TennisMLP(vocab_sizes, num_numerical, num_classes=len(target_encoder.classes_)).to(device),
            'lstm': TennisLSTM(vocab_sizes, num_numerical, num_classes=len(target_encoder.classes_)).to(device)
        }
        
        results = {}
        for model_type, model in nn_models.items():
            log_and_print(f"\nTraining {model_type} model...")
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            train_losses, val_losses, training_time = train_model(model, train_loader, test_loader, model_type, class_weights=class_weights)
            
            # Plot training history
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training History - {model_type}')
            plt.legend()
            plt.savefig(f'{BASE_OUTPUT_DIR}/{model_type}/training_history.png')
            plt.close()
            
            report, cm, y_true, y_pred, y_scores = evaluate_model(model, test_loader, target_encoder, model_type)
            pr_analysis = analyze_precision_recall(y_true, y_pred, y_scores, target_encoder, model_type)
            results[model_type] = {
                'report': report,
                'cm': cm,
                'training_time': training_time,
                'pr_analysis': pr_analysis
            }

        # Train and evaluate XGBoost
        log_and_print("\nTraining XGBoost model...")
        xgb_model = train_xgboost(X_train, y_train, X_test, y_test, categorical_columns, target_encoder)
        report, cm, y_true, y_pred, y_scores = evaluate_model(xgb_model, test_loader, target_encoder, 'xgboost', model_class='xgb')
        pr_analysis = analyze_precision_recall(y_true, y_pred, y_scores, target_encoder, 'xgboost')
        results['xgboost'] = {
            'report': report,
            'cm': cm,
            'pr_analysis': pr_analysis
        }

        # Learning curve analysis
        log_and_print("\nPerforming learning curve analysis...")
        learning_curve_results = learning_curve_analysis(X_train, y_train, X_test, y_test, categorical_columns, vocab_sizes, num_numerical, target_encoder)
        results['learning_curves'] = learning_curve_results

        # Save artifacts
        for model_type in MODEL_TYPES:
            output_dir = f'{BASE_OUTPUT_DIR}/{model_type}'
            with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results[model_type], f)
        with open(os.path.join(BASE_OUTPUT_DIR, 'target_encoder.pkl'), 'wb') as f:
            pickle.dump(target_encoder, f)
        with open(os.path.join(BASE_OUTPUT_DIR, 'feature_encoders.pkl'), 'wb') as f:
            pickle.dump(encoders, f)
        with open(os.path.join(BASE_OUTPUT_DIR, 'feature_columns.pkl'), 'wb') as f:
            pickle.dump(X.columns.tolist(), f)

        log_and_print(f"\nTotal execution time: {datetime.now() - start_time}")
        log_and_print(f"Artifacts saved in {BASE_OUTPUT_DIR}/")

    except Exception as e:
        log_and_print(f"Error in main execution: {str(e)}")
        raise

main()