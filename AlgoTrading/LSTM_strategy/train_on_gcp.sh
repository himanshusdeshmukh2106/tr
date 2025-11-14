#!/bin/bash
# Submit training job to Vertex AI using gcloud CLI

PROJECT_ID="brave-operand-477117-n3"
REGION="asia-south1"
BUCKET_NAME="lstm-trading-asia-south1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="reliance_lstm_${TIMESTAMP}"

echo "=========================================="
echo "SUBMITTING TRAINING JOB TO VERTEX AI"
echo "=========================================="
echo ""
echo "Job Name: ${JOB_NAME}"
echo "Region: ${REGION}"
echo "Machine: n1-standard-4 + Tesla T4 GPU"
echo ""
echo "Cost: ~\$0.54/hour"
echo "Duration: ~30-60 minutes"
echo "Estimated: \$0.30-\$0.60"
echo ""

read -p "Proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Submitting job..."

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --python-package-uris=gs://${BUCKET_NAME}/reliance/ \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest,python-module=trainer.task \
  --args="--project-id=${PROJECT_ID},--bucket-name=${BUCKET_NAME}"

echo ""
echo "=========================================="
echo "JOB SUBMITTED!"
echo "=========================================="
echo ""
echo "Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"
