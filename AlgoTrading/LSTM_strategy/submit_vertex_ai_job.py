"""
Submit training job to Google Cloud Vertex AI
"""
import os
from google.cloud import aiplatform
from google.cloud import storage
from datetime import datetime
import sys
from pathlib import Path

# Set credentials
key_path = Path(__file__).parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

PROJECT_ID = "brave-operand-477117-n3"
REGION = "asia-south1"
BUCKET_NAME = "lstm-trading-asia-south1"


def upload_training_package():
    """Upload training package to GCS"""
    print("Uploading training package to GCS...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Upload trainer package
    files_to_upload = [
        ("trainer/task.py", "reliance/trainer/task.py"),
        ("trainer/__init__.py", "reliance/trainer/__init__.py"),
        ("setup.py", "reliance/setup.py"),
    ]
    
    for local_file, gcs_path in files_to_upload:
        local_path = Path(__file__).parent / local_file
        if local_path.exists():
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            print(f"  ✓ Uploaded {local_file}")
    
    print("Training package uploaded!")
    return f"gs://{BUCKET_NAME}/reliance/"


def submit_training_job():
    """Submit custom training job to Vertex AI"""
    print("\n" + "="*70)
    print("SUBMITTING TRAINING JOB TO VERTEX AI")
    print("="*70)
    
    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}/staging"
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"reliance_lstm_{timestamp}"
    
    print(f"\nJob Configuration:")
    print(f"  Job Name: {job_name}")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Region: {REGION}")
    print(f"  Machine: n1-standard-4")
    print(f"  GPU: NVIDIA Tesla T4")
    print(f"  Container: TensorFlow 2.11 GPU")
    
    # Create custom job
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=job_name,
        python_package_gcs_uri=f"gs://{BUCKET_NAME}/reliance/",
        python_module_name="trainer.task",
        container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest",
    )
    
    print("\n⚠️  Cost Estimate:")
    print("  Machine (n1-standard-4): ~$0.19/hour")
    print("  GPU (Tesla T4): ~$0.35/hour")
    print("  Total: ~$0.54/hour")
    print("  Expected duration: 30-60 minutes")
    print("  Estimated cost: $0.30-$0.60")
    
    confirm = input("\nProceed with cloud training? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("\n❌ Training job cancelled")
        return None
    
    print("\nSubmitting job...")
    
    # Run the training job
    try:
        model = job.run(
            replica_count=1,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            base_output_dir=f"gs://{BUCKET_NAME}/reliance/training_output/{timestamp}",
            args=[
                f"--project-id={PROJECT_ID}",
                f"--bucket-name={BUCKET_NAME}",
            ],
            sync=False  # Don't wait for completion
        )
        
        print("\n✓ Training job submitted successfully!")
        
        if model:
            print(f"\nJob Details:")
            print(f"  Resource Name: {model.resource_name}")
        
        print(f"  Output: gs://{BUCKET_NAME}/reliance/training_output/{timestamp}")
        
        print(f"\nMonitor progress:")
        print(f"  Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print(f"  Command: gcloud ai custom-jobs list --region={REGION}")
        
        print(f"\nView logs:")
        print(f"  gcloud ai custom-jobs stream-logs {job_name} --region={REGION}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Job submission failed: {e}")
        print("\nTrying alternative method with gcloud CLI...")
        return None


def main():
    """Main function"""
    print("="*70)
    print("VERTEX AI TRAINING SUBMISSION")
    print("="*70)
    
    try:
        # Upload training package
        package_uri = upload_training_package()
        
        # Submit job
        job = submit_training_job()
        
        if job:
            print("\n" + "="*70)
            print("JOB SUBMITTED SUCCESSFULLY!")
            print("="*70)
            print("\nThe training is now running on Google Cloud.")
            print("You can close this terminal - training continues in the cloud.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
