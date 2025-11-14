"""
Submit training job to Google Cloud Vertex AI
This runs training on GCP's infrastructure with GPU
"""
import os
from google.cloud import aiplatform
from google.cloud import storage
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME

# Set credentials
key_path = Path(__file__).parent.parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)


def upload_training_script():
    """Upload training script to GCS"""
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(GCP_BUCKET_NAME)
    
    # Upload the training script
    script_path = Path(__file__).parent / "retrain_improved.py"
    blob = bucket.blob("reliance/scripts/retrain_improved.py")
    blob.upload_from_filename(str(script_path))
    print(f"✓ Training script uploaded to GCS")
    
    # Upload requirements
    req_path = Path(__file__).parent.parent / "requirements.txt"
    blob = bucket.blob("reliance/scripts/requirements.txt")
    blob.upload_from_filename(str(req_path))
    print(f"✓ Requirements uploaded to GCS")


def submit_training_job():
    """Submit custom training job to Vertex AI"""
    print("\n" + "="*70)
    print("SUBMITTING TRAINING JOB TO VERTEX AI")
    print("="*70)
    
    # Initialize Vertex AI
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        staging_bucket=f"gs://{GCP_BUCKET_NAME}"
    )
    
    # Define the custom training job
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="reliance-lstm-training",
        python_package_gcs_uri=f"gs://{GCP_BUCKET_NAME}/reliance/scripts/",
        python_module_name="retrain_improved",
        container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest",
    )
    
    print("\nJob Configuration:")
    print(f"  Display Name: reliance-lstm-training")
    print(f"  Region: {GCP_REGION}")
    print(f"  Container: TensorFlow 2.11 with GPU")
    print(f"  Machine: n1-standard-4")
    print(f"  GPU: NVIDIA Tesla T4")
    
    # Run the training job
    print("\nSubmitting job to Vertex AI...")
    print("⚠️  This will incur GCP charges (~$1-3 per hour)")
    
    confirm = input("\nProceed with cloud training? (yes/no): ")
    
    if confirm.lower() == 'yes':
        model = job.run(
            replica_count=1,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            base_output_dir=f"gs://{GCP_BUCKET_NAME}/reliance/training_output",
        )
        
        print("\n✓ Training job submitted successfully!")
        print(f"  Job Resource Name: {model.resource_name}")
        print(f"\nMonitor progress at:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={GCP_PROJECT_ID}")
        
        return model
    else:
        print("\n❌ Training job cancelled")
        return None


def submit_simple_training():
    """Submit a simpler training job using gcloud command"""
    print("\n" + "="*70)
    print("ALTERNATIVE: Submit via gcloud command")
    print("="*70)
    
    command = f"""
gcloud ai custom-jobs create \\
  --region={GCP_REGION} \\
  --display-name=reliance-lstm-training \\
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest,python-module=retrain_improved \\
  --python-package-uris=gs://{GCP_BUCKET_NAME}/reliance/scripts/retrain_improved.py
"""
    
    print("\nRun this command in your terminal:")
    print(command)


def main():
    """Main function"""
    print("="*70)
    print("GCP CLOUD TRAINING SETUP")
    print("="*70)
    
    print("\nOptions:")
    print("1. Train locally (current method) - FREE")
    print("2. Train on GCP Vertex AI with GPU - ~$1-3/hour")
    print("3. Show gcloud command for manual submission")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        print("\n✓ Continue using local training")
        print("Run: python src/retrain_improved.py")
        
    elif choice == "2":
        # Upload scripts first
        upload_training_script()
        
        # Submit job
        submit_training_job()
        
    elif choice == "3":
        submit_simple_training()
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
