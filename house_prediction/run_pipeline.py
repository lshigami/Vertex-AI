from kfp.v2 import compiler
from kfp.v2.dsl import pipeline
from google.cloud import aiplatform

from src.data_ingestion import data_ingestion
from src.preprocessing import preprocessing
from src.training import training
from src.evaluation import evaluation

# Configuration
PROJECT_ID = "vertex-ai-484314"
REGION = "europe-west4"
BUCKET_NAME = "gs://house-price-vertex-thang-2026"
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root_houseprice"

@pipeline(
    name="houseprice-pipeline",
    pipeline_root=PIPELINE_ROOT
)
def houseprice_pipeline():
    """House price prediction pipeline."""
    
    # Step 1: Data Ingestion
    ingestion_task = data_ingestion()
    
    # Step 2: Preprocessing
    preprocessing_task = preprocessing(
        input_dataset=ingestion_task.outputs["dataset"]
    )
    
    # Step 3: Training
    training_task = training(
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    )
    
    # Step 4: Evaluation
    evaluation_task = evaluation(
        model=training_task.outputs["model"],
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"]
    )


def main():
    # 1. Compile the pipeline
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=houseprice_pipeline,
        package_path='houseprice_pipeline.json'
    )
    print("Pipeline compiled to houseprice_pipeline.json")
    
    # 2. Initialize Vertex AI
    print(f"Initializing Vertex AI for project {PROJECT_ID}...")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # 3. Create and run pipeline job
    print("Creating pipeline job...")
    pipeline_job = aiplatform.PipelineJob(
        display_name="houseprice-pipeline-job",
        template_path="houseprice_pipeline.json",
        pipeline_root=PIPELINE_ROOT
    )
    
    print("Running pipeline... (this may take several minutes)")
    pipeline_job.run(sync=True)
    
    print("Pipeline completed!")


if __name__ == "__main__":
    main()
