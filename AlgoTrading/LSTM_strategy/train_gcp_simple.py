"""
Simple GCP training submission using gcloud CLI
"""
import subprocess
import os
from datetime import datetime
from pathlib import Path
from google.cloud import storage

# Set credentials
key_path = Path(__file__).parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

PROJECT_ID = "brave-operand-477117-n3"
REGION = "asia-south1"
BUCKET_NAME = "lstm-trading-asia-south1"


def upload_package():
    """Upload training package"""
    print("Uploading training package...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    files = [
        ("trainer/task.py", "reliance/trainer/task.py"),
        ("trainer/__init__.py", "reliance/trainer/__init__.py"),
        ("setup.py", "reliance/setup.py"),
    ]
    
    for local_file, gcs_path in files:
        local_path = Path(__file__).parent / local_file
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        print(f"  ✓ {local_file}")
    
    print("Package uploaded!")


def submit_job():
    """Submit job using gcloud CLI"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"reliance-lstm-{timestamp}"
    
    print("\n" + "="*70)
    print("SUBMITTING TO VERTEX AI")
    print("="*70)
    print(f"\nJob: {job_name}")
    print(f"Region: {REGION}")
    print(f"Machine: n1-standard-4 + Tesla T4 GPU")
    print(f"Cost: ~$0.50 per run")
    
    confirm = input("\nProceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled")
        return
    
    # Build gcloud command
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
        f"--python-package-uris=gs://{BUCKET_NAME}/reliance/",
        "--worker-pool-spec=" + ",".join([
            "machine-type=n1-standard-4",
            "replica-count=1",
            "accelerator-type=NVIDIA_TESLA_T4",
            "accelerator-count=1",
            "container-image-uri=gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest",
            "python-module=trainer.task"
        ]),
        f"--args=--project-id={PROJECT_ID},--bucket-name={BUCKET_NAME}"
    ]
    
    print("\nSubmitting...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✓ JOB SUBMITTED!")
        print(f"\nMonitor:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print(f"\nLogs:")
        print(f"  gcloud ai custom-jobs stream-logs {job_name} --region={REGION}")
    else:
        print(f"\n❌ Error: {result.stderr}")


def main():
    print("="*70)
    print("GCP CLOUD TRAINING")
    print("="*70)
    
    upload_package()
    submit_job()


if __name__ == "__main__":
    main()
