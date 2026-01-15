from kfp.v2.dsl import component, Input, Output, Dataset

PROJECT_ID = "vertex-ai-484314"
REGION = "europe-west4"
REPOSITORY = "vertex-ai-pipeline-thang"
IMAGE_NAME = "training"
IMAGE_TAG = "latest"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

@component(
    base_image=BASE_IMAGE,
    output_component_file="preprocessing.yaml"
)
def preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
):
    """Preprocesses the dataset for training."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Load the dataset
    df = pd.read_csv(input_dataset.path)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    
    # 1. Handle missing values
    df = df.dropna()
    logging.info(f"After dropping NaN: {df.shape}")
    
    # 2. Encode categorical features (yes/no columns)
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    logging.info(f"Encoded categorical columns: {list(categorical_columns)}")
    
    # 3. Scale numerical features (except target 'price')
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numerical_columns:
        numerical_columns.remove('price')
    
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    logging.info(f"Scaled numerical columns: {numerical_columns}")
    
    # 4. Save preprocessed dataset
    df.to_csv(preprocessed_dataset.path, index=False)
    logging.info(f"Preprocessed dataset saved to: {preprocessed_dataset.path}")
