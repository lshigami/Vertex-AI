from kfp.v2.dsl import component, Output, Dataset

PROJECT_ID = "vertex-ai-484314"
REGION = "europe-west4"
REPOSITORY = "vertex-ai-pipeline-thang"
IMAGE_NAME = "training"
IMAGE_TAG = "latest"
BUCKET_NAME = "gs://house-price-vertex-thang-2026"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

@component(
    base_image=BASE_IMAGE,
    output_component_file="data_ingestion.yaml"
)
def data_ingestion(
    dataset: Output[Dataset]
):
    """Loads and prepares the house price dataset."""
    import pandas as pd
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting data ingestion...")
    
    # 1. Load dataset from GCS bucket
    gcs_path = "gs://house-price-vertex-thang-2026/data/Housing.csv"
    df = pd.read_csv(gcs_path)
    
    logging.info(f"Loaded {len(df)} rows from {gcs_path}")
    logging.info(f"Columns: {list(df.columns)}")
    
    # 2. Save dataset to output artifact
    logging.info(f"Saving dataset to {dataset.path}...")
    df.to_csv(dataset.path, index=False)
    
    logging.info("Data ingestion complete!")
