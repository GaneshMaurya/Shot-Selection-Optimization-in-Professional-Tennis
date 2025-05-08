import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
import logging

# Setup logging
logging.basicConfig(filename='xgboost_training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def log_and_print(message):
    logger.info(message)
    print(message)

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

# Main execution
def main():
    # Load and preprocess data
    df = load_data('tennis_points.csv')
    X, y, categorical_columns, vocab_sizes, encoders, target_encoder = preprocess_data(df)

    # Train-test split (80-20, stratified)
    log_and_print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Verify split sizes
    log_and_print(f"Training set size: {X_train.shape[0]} samples")
    log_and_print(f"Test set size: {X_test.shape[0]} samples")

    # Train the XGBoost model with best parameters
    log_and_print("Training XGBoost model...")
    model = XGBClassifier(
        max_depth=7,
        n_estimators=300,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    log_and_print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    class_names = target_encoder.classes_
    log_and_print("Classification Report:")
    log_and_print(classification_report(y_test, y_pred, target_names=class_names))

    # Save the model as a .pkl file
    log_and_print("Saving model as xgboost_model.pkl...")
    with open('xgboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save the test split for inference
    log_and_print("Saving test split as test_data.csv...")
    test_data = X_test.copy()
    test_data['outcome'] = y_test
    # Decode numerical values back to original categories for readability
    for col in categorical_columns:
        test_data[col] = encoders[col].inverse_transform(test_data[col])
    test_data['outcome'] = target_encoder.inverse_transform(test_data['outcome'])
    test_data.to_csv('test_data.csv', index=False)

    # Save encoders for inference
    log_and_print("Saving encoders for inference...")
    with open('feature_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    with open('target_encoder.pkl', 'wb') as file:
        pickle.dump(target_encoder, file)

    log_and_print("Training and saving completed!")

if __name__ == "__main__":
    main()