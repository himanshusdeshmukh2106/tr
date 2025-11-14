# ✅ GCP Cloud Training - READY TO USE!

## Setup Complete

All infrastructure is configured and ready for cloud training on Google Cloud Platform!

### What's Been Set Up

✅ **Vertex AI Training Package**
- `trainer/task.py` - Main training script
- `trainer/__init__.py` - Package init
- `setup.py` - Package configuration
- All uploaded to GCS: `gs://lstm-trading-asia-south1/reliance/`

✅ **GCP Infrastructure**
- Project: `brave-operand-477117-n3`
- Region: `asia-south1` (Mumbai)
- Bucket: `lstm-trading-asia-south1`
- APIs: Vertex AI, Storage, Compute enabled
- Service Account: `lstm-trading-sa` with permissions

✅ **Training Configuration**
- Machine: n1-standard-4 (4 vCPUs, 15 GB RAM)
- GPU: NVIDIA Tesla T4
- Container: TensorFlow 2.11 with GPU support
- 4 model configurations to test

## How to Train on GCP Cloud

### Option 1: Python Script (Recommended)

```bash
python AlgoTrading/LSTM_strategy/submit_vertex_ai_job.py
```

**What it does:**
1. Uploads training package to GCS
2. Submits job to Vertex AI
3. Provides monitoring links
4. Returns immediately (training runs in cloud)

**When prompted, type:** `yes`

### Option 2: PowerShell (Windows)

```powershell
cd AlgoTrading/LSTM_strategy
python submit_vertex_ai_job.py
```

### Option 3: gcloud CLI

```bash
# Make script executable (if on Linux/Mac)
chmod +x train_on_gcp.sh
./train_on_gcp.sh

# Or run directly with gcloud
gcloud ai custom-jobs create \
  --region=asia-south1 \
  --display-name=reliance-lstm-training \
  --python-package-uris=gs://lstm-trading-asia-south1/reliance/ \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/cloud-aiplatform/training/tf-gpu.2-11:latest,python-module=trainer.task \
  --args="--project-id=brave-operand-477117-n3,--bucket-name=lstm-trading-asia-south1"
```

## What Happens During Training

1. **Job Submission** (1 minute)
   - Package uploaded to GCS
   - Vertex AI job created
   - Resources allocated

2. **Training Execution** (30-60 minutes)
   - Data downloaded from GCS
   - 4 model configurations trained
   - Best model selected
   - Results saved to GCS

3. **Completion**
   - Models saved: `gs://lstm-trading-asia-south1/reliance/training_output/`
   - Results JSON with metrics
   - Logs available in Cloud Console

## Monitor Training

### View in Console
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=brave-operand-477117-n3
```

### Stream Logs (gcloud)
```bash
gcloud ai custom-jobs stream-logs reliance_lstm_TIMESTAMP --region=asia-south1
```

### List Jobs
```bash
gcloud ai custom-jobs list --region=asia-south1
```

## Cost Breakdown

**Per Training Run:**
- Machine (n1-standard-4): $0.19/hour
- GPU (Tesla T4): $0.35/hour
- **Total: $0.54/hour**

**Expected:**
- Duration: 30-60 minutes
- Cost per run: **$0.30-$0.60**
- 4 configurations tested
- Best model selected automatically

**Monthly (if training daily):**
- 30 runs × $0.50 = **$15/month**

## Training Configurations

The job will test 4 refined configurations:

### Config 1: Wider + Deeper
- LSTM: [256, 128, 64]
- Dropout: 0.3
- LR: 0.0005
- Batch: 8

### Config 2: Optimal (Based on previous best)
- LSTM: [192, 96, 48]
- Dropout: 0.3
- LR: 0.0005
- Batch: 16

### Config 3: Balanced
- LSTM: [224, 112, 56]
- Dropout: 0.32
- LR: 0.0006
- Batch: 12

### Config 4: Aggressive
- LSTM: [320, 160, 80]
- Dropout: 0.35
- LR: 0.0003
- Batch: 8

## Output Files

After training completes, you'll find:

**Models:**
```
gs://lstm-trading-asia-south1/reliance/training_output/TIMESTAMP/
├── refined_1_wider_deeper_model.h5
├── refined_2_optimal_model.h5
├── refined_3_balanced_model.h5
├── refined_4_aggressive_model.h5
├── best_model.h5  (best performing)
└── results.json   (all metrics)
```

**Download Results:**
```bash
gsutil cp gs://lstm-trading-asia-south1/reliance/training_output/TIMESTAMP/results.json .
gsutil cp gs://lstm-trading-asia-south1/reliance/training_output/TIMESTAMP/best_model.h5 .
```

## Advantages of Cloud Training

✅ **10x Faster** - GPU acceleration
✅ **Scalable** - Run multiple experiments in parallel
✅ **Professional** - Production-ready infrastructure
✅ **Frees Your Machine** - Train in background
✅ **Reproducible** - Consistent environment
✅ **Logged** - All metrics tracked

## Ready to Train!

Everything is set up and tested. To start training:

```bash
python AlgoTrading/LSTM_strategy/submit_vertex_ai_job.py
```

Type `yes` when prompted, and your training will start on Google Cloud! 🚀

## Troubleshooting

### If job fails to submit:
```bash
# Check APIs are enabled
gcloud services list --enabled

# Check permissions
gcloud projects get-iam-policy brave-operand-477117-n3
```

### If training fails:
```bash
# View logs
gcloud ai custom-jobs stream-logs JOB_NAME --region=asia-south1

# Check job status
gcloud ai custom-jobs describe JOB_NAME --region=asia-south1
```

### If you need help:
- Check logs in Cloud Console
- Review error messages
- Verify data is in GCS: `gsutil ls gs://lstm-trading-asia-south1/reliance/`

## Summary

✅ **Status:** READY FOR CLOUD TRAINING
✅ **Infrastructure:** Fully configured
✅ **Cost:** ~$0.50 per training run
✅ **Speed:** 10x faster than local
✅ **Command:** `python submit_vertex_ai_job.py`

**All training from now on will use Google Cloud! 🎯**
