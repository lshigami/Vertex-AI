from kfp.v2.dsl import component, Input, Output, Model, Dataset, Metrics, HTML

PROJECT_ID = "vertex-ai-484314"
REGION = "europe-west4"
REPOSITORY = "vertex-ai-pipeline-thang"
IMAGE_NAME = "training"
IMAGE_TAG = "latest"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

@component(
    base_image=BASE_IMAGE,
    output_component_file="evaluation.yaml"
)
def evaluation(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics],
    html: Output[HTML]
):
    """Evaluates the model's performance and generates visualizations."""
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import logging
    import base64
    from io import BytesIO
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load the model and dataset
    rf_model = joblib.load(model.path)
    df = pd.read_csv(preprocessed_dataset.path)
    
    logging.info(f"Loaded model and dataset with shape: {df.shape}")
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # 2. Make predictions
    y_pred = rf_model.predict(X)
    
    # 3. Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # 4. Save metrics
    metrics.log_metric("mse", float(mse))
    metrics.log_metric("rmse", float(rmse))
    metrics.log_metric("mae", float(mae))
    metrics.log_metric("r2_score", float(r2))
    
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")
    
    # 5. Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y, y_pred, alpha=0.5)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title('Actual vs Predicted Prices')
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Feature Importance')
    
    plt.tight_layout()
    
    # Convert plot to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # 6. Create and save HTML report
    html_content = f"""
    <html>
    <head><title>Model Evaluation Report</title></head>
    <body>
        <h1>House Price Prediction - Model Evaluation</h1>
        <h2>Performance Metrics</h2>
        <table border="1" style="border-collapse: collapse; padding: 10px;">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>MSE</td><td>{mse:.4f}</td></tr>
            <tr><td>RMSE</td><td>{rmse:.4f}</td></tr>
            <tr><td>MAE</td><td>{mae:.4f}</td></tr>
            <tr><td>R2 Score</td><td>{r2:.4f}</td></tr>
        </table>
        <h2>Visualizations</h2>
        <img src="data:image/png;base64,{img_base64}" />
    </body>
    </html>
    """
    
    with open(html.path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Evaluation report saved to: {html.path}")
