"""
Google Cloud Platform training script for LSTM model
"""
import os
import numpy as np
from google.cloud import storage
from google.cloud import aiplatform
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME,
    DATA_DIR, MODELS_DIR, LSTM_CONFIG
)


class GCPTrainer:
    def __init__(self):
        self.project_id = GCP_PROJECT_ID
        self.region = GCP_REGION
        self.bucket_name = GCP_BUCKET_NAME
        self.storage_client = storage.Client(project=self.project_id)
        
    def upload_data_to_gcs(self, local_path, gcs_path):
        """Upload data to Google Cloud Storage"""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")
    
    def download_from_gcs(self, gcs_path, local_path):
        """Download from Google Cloud Storage"""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
    
    def create_training_job(self, display_name="lstm-trading-training"):
        """Create custom training job on Vertex AI"""
        aiplatform.init(project=self.project_id, location=self.region)
        
        # Define custom job
        job = aiplatform.CustomTrainingJob(
            display_name=display_name,
            script_path="src/train_local.py",
            container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest",
            requirements=["numpy", "pandas", "scikit-learn", "tensorflow"],
            model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-11:latest",
        )
        
        # Run the job
        model = job.run(
            replica_count=1,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
        )
        
        return model
    
    def upload_training_data(self):
        """Upload all necessary training data to GCS"""
        files_to_upload = [
            (DATA_DIR / "X_sequences.npy", "data/X_sequences.npy"),
            (DATA_DIR / "y_targets.npy", "data/y_targets.npy"),
            (DATA_DIR / "scaler.pkl", "data/scaler.pkl"),
        ]
        
        for local_file, gcs_path in files_to_upload:
            if local_file.exists():
                self.upload_data_to_gcs(str(local_file), gcs_path)


def main():
    """Main training workflow on GCP"""
    trainer = GCPTrainer()
    
    print("Step 1: Uploading training data to GCS...")
    trainer.upload_training_data()
    
    print("\nStep 2: Creating training job on Vertex AI...")
    print("Note: This will incur GCP charges!")
    
    # Uncomment to run actual training
    # model = trainer.create_training_job()
    # print(f"Training job completed. Model: {model.resource_name}")
    
    print("\nTo run training, uncomment the training job creation in the code.")
    print("Estimated cost: $1-3 per hour with GPU")


if __name__ == "__main__":
    main()
