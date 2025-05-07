# shot-optmization-model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import os
import time
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def load_data(csv_path):
    """
    Load the tennis point data from CSV
    """
    logger.info(f"Loading data from {csv_path}...")
    # Set low_memory=False to handle the mixed types warning
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} points with {len(df.columns)} features")
    
    # Print a sample
    logger.info("\nSample data (first 5 rows):")
    logger.info(df.head())
    
    # Print column information
    logger.info("\nColumn information:")
    for col in df.columns:
        non_null = df[col].count()
        pct_filled = (non_null / len(df)) * 100
        unique_vals = df[col].nunique()
        logger.info(f"{col}: {non_null}/{len(df)} non-null ({pct_filled:.1f}%) - {unique_vals} unique values")
    
    return df

def clean_data(df):
    """
    Clean and prepare the dataset for modeling
    """
    logger.info("\nCleaning dataset...")
    original_shape = df.shape
    
    # Map outcome values to standardized classes
    outcome_mapping = {
        'Winner': 'Winner',
        'Forced Error': 'Forced Error',
        'Unforced Error': 'Unforced Error',
        'Ace': 'Winner',
        'Serve Winner': 'Winner',
        'Double Fault': 'Unforced Error'
    }
    
    # Map the outcome column
    if 'outcome' in df.columns:
        logger.info("Mapping 'outcome' column to standardized classes...")
        df['outcome'] = df['outcome'].map(lambda x: outcome_mapping.get(x, 'Unknown'))
    elif 'result' in df.columns:
        logger.info("Mapping 'result' column to standardized classes...")
        df['outcome'] = df['result'].map(lambda x: outcome_mapping.get(x, 'Unknown'))
        df = df.drop('result', axis=1, errors='ignore')
    else:
        logger.error("No 'outcome' or 'result' column found in data!")
        raise ValueError("Missing target column")
    
    # Fill missing values
    logger.info("Filling missing values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(-1)
    
    # Remove points with unknown outcomes
    before_filter = len(df)
    df = df[df['outcome'] != 'Unknown']
    logger.info(f"Removed {before_filter - len(df)} points with unknown outcomes")
    
    # Check class distribution
    logger.info("\nClass distribution after cleaning:")
    class_counts = df['outcome'].value_counts()
    for outcome, count in class_counts.items():
        logger.info(f"{outcome}: {count} points ({count/len(df)*100:.1f}%)")
    
    logger.info(f"Data cleaning complete: {original_shape} -> {df.shape}")
    return df

def prepare_features(df):
    """
    Prepare features for model training
    """
    logger.info("\nPreparing features...")
    
    # Select features to use (exclude unnecessary columns)
    exclude_cols = ['match_id', 'player1', 'player2', 'score', 'outcome', 
                    'serve_full', 'first_serve_full', 'shot_1_full', 'shot_2_full', 
                    'shot_3_full', 'shot_4_full', 'shot_5_full', 'last_shot_full']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Selected {len(feature_cols)} feature columns")
    
    # Features and target
    X = df[feature_cols].copy()
    y = df['outcome']
    
    # Convert all columns to strings to avoid mixed type issues
    logger.info("Converting columns to consistent types...")
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_float_dtype(X[col]):
            X[col] = X[col].astype(str)
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Found {len(categorical_cols)} categorical columns")
    
    # One-hot encode categorical columns
    logger.info("Performing one-hot encoding...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(X[categorical_cols])
    
    # Get the feature names
    feature_names = encoder.get_feature_names_out(categorical_cols)
    logger.info(f"Generated {len(feature_names)} encoded features")
    
    # Convert to DataFrame
    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names)
    
    # Drop original categorical columns and add encoded ones
    X_numeric = X.drop(categorical_cols, axis=1)
    
    # Convert any remaining columns to numeric
    for col in X_numeric.columns:
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(0)
    
    X_numeric = X_numeric.reset_index(drop=True)
    X_final = pd.concat([X_numeric, encoded_df], axis=1)
    
    logger.info(f"Final feature matrix shape: {X_final.shape}")
    
    # Encode target variable to numeric labels
    logger.info("Encoding target variable to numeric labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Target classes encoded: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return X_final, y_encoded, encoder, feature_names, label_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    logger.info(f"\nSplitting data into train/test sets with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with simpler approach
    """
    logger.info("\nTraining XGBoost model...")
    start_time = time.time()
    
    # Create model with balanced class weights
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_estimators=100,  # Fixed number of estimators
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Train without early stopping
    logger.info("Fitting model...")
    model.fit(X_train, y_train, verbose=True)
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, X_test, y_test, label_encoder, output_dir="model_output"):
    """
    Evaluate the model and generate visualizations
    """
    logger.info("\nEvaluating model performance...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Map numeric predictions back to class names for reporting
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Classification report
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test_labels, y_pred_labels))
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    logger.info(f"Saved confusion matrix to {os.path.join(output_dir, 'confusion_matrix.png')}")
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(12, 10))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    logger.info(f"Saved feature importance plot to {os.path.join(output_dir, 'feature_importance.png')}")
    plt.close()
    
    return report, cm

def check_overfitting(model, X_train, y_train, X_test, y_test, output_dir="model_output"):
    """
    Compare train and test performance to detect overfitting
    """
    logger.info("\nChecking for overfitting...")
    
    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Difference: {train_accuracy - test_accuracy:.4f}")
    
    # Plot accuracy comparison
    plt.figure(figsize=(8, 6))
    plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.title('Training vs Test Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, 'overfitting_check.png'))
    logger.info(f"Saved overfitting check plot to {os.path.join(output_dir, 'overfitting_check.png')}")
    plt.close()
    
    # Check if overfitting
    if train_accuracy - test_accuracy > 0.05:
        logger.warning("WARNING: Model may be overfitting. Consider regularization or simpler model.")
    else:
        logger.info("Model does not show significant signs of overfitting.")
    
    return train_accuracy, test_accuracy

def save_model_artifacts(model, encoder, feature_names, label_encoder, report, output_dir="model_output"):
    """
    Save model and related artifacts
    """
    logger.info("\nSaving model artifacts...")
    
    # Save model
    model_path = os.path.join(output_dir, "shot_outcome_model.json")
    model.save_model(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save classification report
    with open(os.path.join(output_dir, "classification_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save feature names
    with open(os.path.join(output_dir, "feature_names.txt"), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    # Save label encoder classes
    with open(os.path.join(output_dir, "class_labels.json"), 'w') as f:
        json.dump({i: label for i, label in enumerate(label_encoder.classes_.tolist())}, f, indent=4)
    
    logger.info(f"Model artifacts saved to {output_dir}")

def create_prediction_function(model, encoder, feature_names, label_encoder):
    """
    Create a function for making predictions on new data
    """
    def predict_shot_outcome(serve_type, rally):
        """
        Predict the outcome of a shot
        
        Args:
            serve_type: The serve type (e.g., '6' for T serve)
            rally: List of shots in the rally (e.g., ['f3', 'b1'])
            
        Returns:
            Dictionary with prediction and probabilities
        """
        # Create a sample with all the columns that were used during training
        sample = pd.DataFrame(columns=feature_names)
        sample.loc[0] = 0  # Fill with zeros initially
        
        # Map features we have to the correct columns
        numerical_features = {
            'server': 0,
            'winner': 0,
            'is_second_serve': 0,
            'rally_length': len(rally),
            'tournament': 'unknown',
            'surface': 'unknown'
        }
        
        # Fill numerical features
        for feature, value in numerical_features.items():
            if feature in sample.columns:
                sample[feature] = value
        
        # Create one-hot encoded columns for categorical features
        serve_type_col = f"serve_type_{serve_type[0] if serve_type else '0'}"
        if serve_type_col in sample.columns:
            sample[serve_type_col] = 1
            
        # Extract serve direction if present
        serve_dir = next((c for c in serve_type if c in '123'), '0') if serve_type else '0'
        serve_dir_col = f"serve_direction_{serve_dir}"
        if serve_dir_col in sample.columns:
            sample[serve_dir_col] = 1
            
        # Extract serve depth if present
        serve_depth = next((c for c in serve_type if c in '789'), '0') if serve_type else '0'
        serve_depth_col = f"serve_depth_{serve_depth}"
        if serve_depth_col in sample.columns:
            sample[serve_depth_col] = 1
        
        # Process rally shots if available
        if len(rally) > 0:
            # First shot
            if len(rally) >= 1:
                shot_type = rally[0][0] if len(rally[0]) > 0 else '0'
                shot_dir = next((c for c in rally[0] if c in '123'), '0')
                shot_depth = next((c for c in rally[0] if c in '789'), '0')
                
                shot_type_col = f"shot_1_type_{shot_type}"
                if shot_type_col in sample.columns:
                    sample[shot_type_col] = 1
                    
                shot_dir_col = f"shot_1_direction_{shot_dir}"
                if shot_dir_col in sample.columns:
                    sample[shot_dir_col] = 1
                    
                shot_depth_col = f"shot_1_depth_{shot_depth}"
                if shot_depth_col in sample.columns:
                    sample[shot_depth_col] = 1
            
            # Second shot
            if len(rally) >= 2:
                shot_type = rally[1][0] if len(rally[1]) > 0 else '0'
                shot_dir = next((c for c in rally[1] if c in '123'), '0')
                shot_depth = next((c for c in rally[1] if c in '789'), '0')
                
                shot_type_col = f"shot_2_type_{shot_type}"
                if shot_type_col in sample.columns:
                    sample[shot_type_col] = 1
                    
                shot_dir_col = f"shot_2_direction_{shot_dir}"
                if shot_dir_col in sample.columns:
                    sample[shot_dir_col] = 1
                    
                shot_depth_col = f"shot_2_depth_{shot_depth}"
                if shot_depth_col in sample.columns:
                    sample[shot_depth_col] = 1
            
            # Third shot
            if len(rally) >= 3:
                shot_type = rally[2][0] if len(rally[2]) > 0 else '0'
                shot_dir = next((c for c in rally[2] if c in '123'), '0')
                shot_depth = next((c for c in rally[2] if c in '789'), '0')
                
                shot_type_col = f"shot_3_type_{shot_type}"
                if shot_type_col in sample.columns:
                    sample[shot_type_col] = 1
                    
                shot_dir_col = f"shot_3_direction_{shot_dir}"
                if shot_dir_col in sample.columns:
                    sample[shot_dir_col] = 1
                    
                shot_depth_col = f"shot_3_depth_{shot_depth}"
                if shot_depth_col in sample.columns:
                    sample[shot_depth_col] = 1
                    
            # Last shot features if they exist in our model
            last_shot = rally[-1]
            last_type = last_shot[0] if len(last_shot) > 0 else '0'
            last_dir = next((c for c in last_shot if c in '123'), '0')
            last_depth = next((c for c in last_shot if c in '789'), '0')
            
            last_type_col = f"last_shot_type_{last_type}"
            if last_type_col in sample.columns:
                sample[last_type_col] = 1
                
            last_dir_col = f"last_shot_direction_{last_dir}"
            if last_dir_col in sample.columns:
                sample[last_dir_col] = 1
                
            last_depth_col = f"last_shot_depth_{last_depth}"
            if last_depth_col in sample.columns:
                sample[last_depth_col] = 1
        
        # Make prediction
        pred_idx = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        
        # Convert prediction to class label
        prediction = label_encoder.inverse_transform([pred_idx])[0]
        
        # Create probabilities dictionary
        probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(proba)}
        
        return {
            'prediction': prediction,
            'probabilities': probabilities
        }
    
    return predict_shot_outcome

def test_prediction_function(predict_fn):
    """
    Test the prediction function with an example
    """
    logger.info("\nTesting prediction function with an example...")
    
    # Example: T serve followed by forehand return cross-court and backhand down the line
    serve = '6'  # T serve
    rally = ['f3', 'b1']  # Forehand to left, backhand to right
    
    logger.info(f"Example: Serve '{serve}' followed by rally {rally}")
    result = predict_fn(serve, rally)
    
    logger.info(f"Predicted outcome: {result['prediction']}")
    logger.info("Probabilities:")
    for outcome, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {outcome}: {prob:.4f} ({prob*100:.1f}%)")
    
    return result

def main():
    """
    Main function to run the entire pipeline
    """
    logger.info("============= Tennis Shot Outcome Prediction =============")
    
    # Set paths
    data_path = "tennis_points.csv"
    output_dir = "model_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the data
    df = load_data(data_path)
    
    # Step 2: Clean the data
    df = clean_data(df)
    
    # Step 3: Prepare features
    X, y, encoder, feature_names, label_encoder = prepare_features(df)
    
    # Step 4: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Train the model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Step 6: Evaluate the model
    report, cm = evaluate_model(model, X_test, y_test, label_encoder, output_dir)
    
    # Step 7: Check for overfitting
    train_accuracy, test_accuracy = check_overfitting(model, X_train, y_train, X_test, y_test, output_dir)
    
    # Step 8: Save model artifacts
    save_model_artifacts(model, encoder, feature_names, label_encoder, report, output_dir)
    
    # Step A: Create and test prediction function
    predict_fn = create_prediction_function(model, encoder, feature_names, label_encoder)
    test_prediction_function(predict_fn)
    
    logger.info("============= Training Pipeline Complete =============")
    
    return model, encoder, feature_names, label_encoder

if __name__ == "__main__":
    main()
