from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

PROJECT_ID = "vertex-ai-484314"
REGION = "europe-west4"
REPOSITORY = "vertex-ai-pipeline-thang"
IMAGE_NAME = "training"
IMAGE_TAG = "latest"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

@component(
    base_image=BASE_IMAGE,
    output_component_file="training.yaml"
)
def training(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    hyperparameters: dict
):
    """Trains the model on the preprocessed dataset."""
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Load preprocessed dataset
    df = pd.read_csv(preprocessed_dataset.path)
    logging.info(f"Loaded preprocessed dataset with shape: {df.shape}")
    
    # 1. Split features and target (assuming 'price' is the target)
    X = df.drop('price', axis=1)
    y = df['price']
    
    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # 2. Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=hyperparameters.get('random_state', 42)
    )
    
    logging.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # 3. Initialize and train the model
    rf_model = RandomForestRegressor(
        n_estimators=hyperparameters.get('n_estimators', 100),
        max_depth=hyperparameters.get('max_depth', 10),
        random_state=hyperparameters.get('random_state', 42)
    )
    rf_model.fit(X_train, y_train)
    
    logging.info("Model training complete!")
    
    # 4. Make predictions
    y_pred = rf_model.predict(X_val)
    
    # 5. Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Log metrics
    metrics.log_metric("mse", float(mse))
    metrics.log_metric("r2_score", float(r2))
    
    # 6. Save the model
    joblib.dump(rf_model, model.path)
    logging.info(f"Model saved to: {model.path}")
    logging.info(f"Validation MSE: {mse:.2f}")
    logging.info(f"Validation R2: {r2:.2f}")
