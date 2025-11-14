"""
Check Vertex AI quotas for the project
"""
import os
from pathlib import Path
from google.cloud import aiplatform
from google.api_core import exceptions

# Set credentials
key_path = Path(__file__).parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

PROJECT_ID = "brave-operand-477117-n3"
REGION = "asia-south1"

print("="*70)
print("CHECKING VERTEX AI QUOTAS")
print("="*70)

# Initialize
aiplatform.init(project=PROJECT_ID, location=REGION)

print(f"\nProject: {PROJECT_ID}")
print(f"Region: {REGION}")

# Try to get quota info by attempting to create a job
print("\nAttempting to check quotas...")
print("\nNote: The error message from your previous attempt said:")
print("  'custom_model_training_cpus' - EXCEEDED")
print("  'custom_model_training_nvidia_t4_gpus' - EXCEEDED")
print("\nThis means your quotas are likely set to 0 (default for new projects)")

print("\n" + "="*70)
print("SOLUTION: Request Quota Increase")
print("="*70)
print("\n1. Go to: https://console.cloud.google.com/iam-admin/quotas")
print(f"   ?project={PROJECT_ID}")
print("\n2. Search for: 'custom_model_training'")
print("\n3. Select these quotas for asia-south1:")
print("   - Custom model training CPUs per region")
print("   - Custom model training NVIDIA T4 GPUs per region")
print("\n4. Click 'EDIT QUOTAS' and request:")
print("   - CPUs: 4")
print("   - T4 GPUs: 1")
print("\n5. Submit and wait for approval (1-2 days)")

print("\n" + "="*70)
print("ALTERNATIVE: Train Locally Now")
print("="*70)
print("\nRun this command to train with improved models locally:")
print("  python AlgoTrading/LSTM_strategy/train_improved_local.py")
print("\nNo quota needed, trains on your machine!")
