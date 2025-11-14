# Google Cloud Platform Setup Guide

## Prerequisites
- Google Cloud account
- GCP project created
- Billing enabled

## Step 1: Enable Required APIs
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable compute.googleapis.com
```

## Step 2: Create Service Account
```bash
# Create service account
gcloud iam service-accounts create lstm-trading-sa \
    --display-name="LSTM Trading Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:lstm-trading-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:lstm-trading-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create ~/lstm-trading-key.json \
    --iam-account=lstm-trading-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Step 3: Create Cloud Storage Bucket
```bash
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l us-central1 gs://lstm-trading-bucket/
```

## Step 4: Set Environment Variables
Create a `.env` file in the project root:
```
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCP_BUCKET_NAME=lstm-trading-bucket
GCP_SERVICE_ACCOUNT_KEY=/path/to/lstm-trading-key.json
```

## Step 5: Authenticate
```bash
# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/lstm-trading-key.json"

# Or in Python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/lstm-trading-key.json"
```

## Step 6: Test Connection
```python
from google.cloud import storage

client = storage.Client()
buckets = list(client.list_buckets())
print(f"Connected! Found {len(buckets)} buckets")
```

## Vertex AI Training
For training on Vertex AI, you can use:
- **Custom training jobs** with TensorFlow
- **Pre-built containers** for faster setup
- **Hyperparameter tuning** for optimization

## Cost Estimation
- Storage: ~$0.02/GB/month
- Training (n1-standard-4 + 1 GPU): ~$1-3/hour
- Prediction: Pay per request

## Security Best Practices
1. Never commit service account keys to git
2. Use IAM roles with least privilege
3. Enable VPC Service Controls for production
4. Rotate keys regularly
5. Use Secret Manager for sensitive data
