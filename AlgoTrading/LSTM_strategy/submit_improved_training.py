"""
Submit improved training job to Vertex AI
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


def upload_improved_package():
    """Upload improved training package"""
    print("\n" + "="*70)
    print("UPLOADING IMPROVED TRAINING PACKAGE")
    print("="*70)
    
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    files = [
        ("trainer/task_improved.py", "reliance/trainer/task.py"),
        ("trainer/__init__.py", "reliance/trainer/__init__.py"),
        ("setup.py", "reliance/setup.py"),
    ]
    
    for local_file, gcs_path in files:
        local_path = Path(__file__).parent / local_file
        if local_path.exists():
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            print(f"  ✓ {local_file} -> gs://{BUCKET_NAME}/{gcs_path}")
        else:
            print(f"  ✗ {local_file} not found")
    
    print("\n✓ Package uploaded!")


def submit_improved_job():
    """Submit improved training job"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"reliance-improved-{timestamp}"
    
    print("\n" + "="*70)
    print("SUBMITTING IMPROVED TRAINING JOB")
    print("="*70)
    print(f"\nJob Name: {job_name}")
    print(f"Region: {REGION}")
    print(f"\nImprovements:")
    print("  • Bidirectional LSTM for better context")
    print("  • Multi-head attention mechanism")
    print("  • Transformer and CNN-LSTM architectures")
    print("  • Focal loss for class imbalance")
    print("  • Better regularization (L1+L2, LayerNorm)")
    print("  • Class weighting")
    print("  • Data augmentation option")
    
    print("\nChoose machine type:")
    print("  1. GPU (n1-standard-4 + Tesla T4) - Faster, ~$0.54/hr")
    print("  2. CPU only (n1-standard-4) - Slower, ~$0.19/hr")
    
    choice = input("\nSelect (1 or 2): ").strip()
    
    if choice == "1":
        machine_type = "n1-standard-4"
        use_gpu = True
        container = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest"
        cost = "~$0.50-0.80"
    elif choice == "2":
        machine_type = "n1-standard-4"
        use_gpu = False
        container = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest"
        cost = "~$0.30-0.50"
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\nConfiguration:")
    print(f"  Machine: {machine_type}")
    print(f"  GPU: {'Yes (Tesla T4)' if use_gpu else 'No (CPU only)'}")
    print(f"  Estimated cost: {cost}")
    print(f"  Note: Trains 4 models sequentially (respects quota limit)")
    
    confirm = input("\nProceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    # Build worker pool spec (use executor-image-uri for Python packages)
    worker_spec_parts = [
        f"machine-type={machine_type}",
        "replica-count=1",
        f"executor-image-uri={container}",
        "python-module=trainer.task"
    ]
    
    # Add GPU only if selected
    if use_gpu:
        worker_spec_parts.insert(2, "accelerator-type=NVIDIA_TESLA_T4")
        worker_spec_parts.insert(3, "accelerator-count=1")
    
    # Build gcloud command with args
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
        f"--project={PROJECT_ID}",
        f"--python-package-uris=gs://{BUCKET_NAME}/reliance/",
        "--worker-pool-spec=" + ",".join(worker_spec_parts),
        f"--args=--project-id={PROJECT_ID},--bucket-name={BUCKET_NAME},--augment"
    ]
    
    print("\nSubmitting job...")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✓ JOB SUBMITTED SUCCESSFULLY!")
        print("="*70)
        print(f"\nMonitor at:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print(f"\nStream logs:")
        print(f"  gcloud ai custom-jobs stream-logs {job_name} --region={REGION}")
        print(f"\nResults will be saved to:")
        print(f"  gs://{BUCKET_NAME}/reliance/training_output/")
    else:
        print(f"\n❌ Error submitting job:")
        print(result.stderr)
        if "quota" in result.stderr.lower():
            print("\n💡 Quota issue detected. Try CPU-only option or check:")
            print(f"  https://console.cloud.google.com/iam-admin/quotas?project={PROJECT_ID}")


def main():
    print("="*70)
    print("IMPROVED GCP TRAINING SUBMISSION")
    print("="*70)
    
    # Upload package
    upload_improved_package()
    
    # Submit job
    submit_improved_job()


if __name__ == "__main__":
    main()
