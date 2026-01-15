"""
Local Pipeline Runner - Runs the ML pipeline locally without Vertex AI
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DATA_PATH = "data/Housing.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def data_ingestion():
    """Step 1: Load the dataset"""
    logging.info("="*50)
    logging.info("STEP 1: DATA INGESTION")
    logging.info("="*50)
    
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"First 5 rows:\n{df.head()}")
    
    output_path = f"{OUTPUT_DIR}/raw_data.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved raw data to: {output_path}")
    
    return df


def preprocessing(df):
    """Step 2: Preprocess the data"""
    logging.info("="*50)
    logging.info("STEP 2: PREPROCESSING")
    logging.info("="*50)
    
    logging.info(f"Original shape: {df.shape}")
    
    # Handle missing values
    df = df.dropna()
    logging.info(f"After dropping NaN: {df.shape}")
    
    # Encode categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    logging.info(f"Encoded categorical columns: {categorical_columns}")
    
    # Scale numerical features (except target 'price')
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numerical_columns:
        numerical_columns.remove('price')
    
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    logging.info(f"Scaled numerical columns: {numerical_columns}")
    
    output_path = f"{OUTPUT_DIR}/preprocessed_data.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved preprocessed data to: {output_path}")
    
    return df


def training(df, hyperparameters):
    """Step 3: Train the model"""
    logging.info("="*50)
    logging.info("STEP 3: TRAINING")
    logging.info("="*50)
    
    logging.info(f"Hyperparameters: {hyperparameters}")
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=hyperparameters.get('random_state', 42)
    )
    
    logging.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # Initialize and train the model
    rf_model = RandomForestRegressor(
        n_estimators=hyperparameters.get('n_estimators', 100),
        max_depth=hyperparameters.get('max_depth', 10),
        random_state=hyperparameters.get('random_state', 42)
    )
    rf_model.fit(X_train, y_train)
    
    logging.info("Model training complete!")
    
    # Make predictions on validation set
    y_pred = rf_model.predict(X_val)
    
    # Calculate validation metrics
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    logging.info(f"Validation MSE: {mse:.2f}")
    logging.info(f"Validation R2: {r2:.4f}")
    
    # Save the model
    model_path = f"{OUTPUT_DIR}/model.joblib"
    joblib.dump(rf_model, model_path)
    logging.info(f"Model saved to: {model_path}")
    
    return rf_model, X.columns.tolist()


def evaluation(model, df, feature_names):
    """Step 4: Evaluate the model"""
    logging.info("="*50)
    logging.info("STEP 4: EVALUATION")
    logging.info("="*50)
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    logging.info("="*30)
    logging.info("FINAL METRICS:")
    logging.info("="*30)
    logging.info(f"MSE:  {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE:  {mae:.4f}")
    logging.info(f"R2:   {r2:.4f}")
    
    # Save metrics to file
    metrics_path = f"{OUTPUT_DIR}/metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("House Price Prediction - Model Metrics\n")
        f.write("="*40 + "\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
    logging.info(f"Metrics saved to: {metrics_path}")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y, y_pred, alpha=0.5, color='blue')
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title('Actual vs Predicted Prices')
    axes[0].grid(True, alpha=0.3)
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1].barh(feature_importance['feature'], feature_importance['importance'], color='green')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Feature Importance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{OUTPUT_DIR}/evaluation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"Plots saved to: {plot_path}")
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def main():
    """Run the complete pipeline"""
    logging.info("#"*60)
    logging.info("HOUSE PRICE PREDICTION PIPELINE - LOCAL EXECUTION")
    logging.info("#"*60)
    
    # Hyperparameters
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    # Step 1: Data Ingestion
    raw_data = data_ingestion()
    
    # Step 2: Preprocessing
    preprocessed_data = preprocessing(raw_data.copy())
    
    # Step 3: Training
    model, feature_names = training(preprocessed_data.copy(), hyperparameters)
    
    # Step 4: Evaluation
    metrics = evaluation(model, preprocessed_data, feature_names)
    
    logging.info("#"*60)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("#"*60)
    logging.info(f"Output files saved in: {OUTPUT_DIR}/")
    logging.info("  - raw_data.csv")
    logging.info("  - preprocessed_data.csv")
    logging.info("  - model.joblib")
    logging.info("  - metrics.txt")
    logging.info("  - evaluation_plots.png")
    
    return metrics


if __name__ == "__main__":
    main()
