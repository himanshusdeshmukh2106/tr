"""
Test GCP connection and setup
"""
import os
from pathlib import Path
from google.cloud import storage
from google.cloud import aiplatform

# Set credentials
key_path = Path(__file__).parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

print("=" * 60)
print("Testing GCP Connection")
print("=" * 60)

# Test 1: Storage
print("\n[1/3] Testing Cloud Storage...")
try:
    storage_client = storage.Client(project="brave-operand-477117-n3")
    buckets = list(storage_client.list_buckets())
    print(f"✓ Connected! Found {len(buckets)} buckets:")
    for bucket in buckets:
        print(f"  - {bucket.name}")
except Exception as e:
    print(f"✗ Storage connection failed: {e}")

# Test 2: AI Platform
print("\n[2/3] Testing Vertex AI...")
try:
    aiplatform.init(project="brave-operand-477117-n3", location="asia-south1")
    print("✓ Vertex AI initialized successfully!")
except Exception as e:
    print(f"✗ Vertex AI initialization failed: {e}")

# Test 3: Upload test file
print("\n[3/3] Testing file upload...")
try:
    bucket = storage_client.bucket("lstm-trading-asia-south1")
    blob = bucket.blob("test/connection_test.txt")
    blob.upload_from_string("GCP connection successful!")
    print("✓ Test file uploaded successfully!")
    print(f"  Location: gs://lstm-trading-asia-south1/test/connection_test.txt")
except Exception as e:
    print(f"✗ File upload failed: {e}")

print("\n" + "=" * 60)
print("GCP Setup Complete! ✓")
print("=" * 60)
print("\nYou can now:")
print("1. Run data preparation: python src/data_preparation.py")
print("2. Train locally: python src/train_local.py")
print("3. Train on GCP: python src/train_gcp.py")
