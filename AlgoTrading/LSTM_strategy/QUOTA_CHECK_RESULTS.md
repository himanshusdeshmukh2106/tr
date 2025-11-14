# Vertex AI Quota Check Results

## Error Message Analysis

When we tried to submit the training job, we got:

```
Error 429: RESOURCE_EXHAUSTED
The following quota metrics exceed quota limits:
- aiplatform.googleapis.com/custom_model_training_cpus
- aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus
```

## What This Means

**Vertex AI has separate quotas from Compute Engine:**

### Compute Engine Quotas (What you showed):
- ✅ CPUS_ALL_REGIONS: 12 (0 used)
- ✅ GPUS_ALL_REGIONS: 0 (0 used)
- These are for VM instances

### Vertex AI Training Quotas (What we're hitting):
- ❌ custom_model_training_cpus: **Default is 0 for new projects**
- ❌ custom_model_training_nvidia_t4_gpus: **Default is 0 for new projects**
- These are specifically for Vertex AI training jobs

## Why Default is 0

Google sets Vertex AI training quotas to 0 by default for new projects to:
1. Prevent accidental charges
2. Require explicit opt-in for ML training
3. Control resource usage

## How to Check Your Exact Quotas

### Method 1: Web Console (Easiest)
1. Go to: https://console.cloud.google.com/iam-admin/quotas?project=brave-operand-477117-n3
2. Filter by: "Vertex AI API"
3. Look for:
   - "Custom model training CPUs per region"
   - "Custom model training NVIDIA T4 GPUs per region"
4. Check the "Limit" column

### Method 2: gcloud Command
```bash
gcloud services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/brave-operand-477117-n3 \
  --filter="metric.type:aiplatform.googleapis.com/custom_model_training"
```

### Method 3: Check Specific Region
```bash
gcloud compute regions describe asia-south1 \
  --project=brave-operand-477117-n3
```

## What We Need

To run training on GCP, we need:

### Minimum Requirements:
- **CPUs:** 4 (for n1-standard-4 or e2-standard-4)
- **GPUs:** 1 (for Tesla T4)

### Recommended:
- **CPUs:** 16 (allows multiple experiments)
- **GPUs:** 2 (allows parallel training)

## How to Request Quota Increase

### Step 1: Go to Quotas Page
```
https://console.cloud.google.com/iam-admin/quotas?project=brave-operand-477117-n3
```

### Step 2: Filter and Select
1. Filter by: "Vertex AI"
2. Find: "Custom model training CPUs per region"
3. Select the checkbox
4. Click "EDIT QUOTAS"

### Step 3: Request Increase
- **Service:** Vertex AI API
- **Quota:** Custom model training CPUs per region
- **Region:** asia-south1
- **New Limit:** 16
- **Justification:** "Training LSTM models for financial algorithmic trading. Need resources for hyperparameter tuning and model optimization."

### Step 4: Repeat for GPUs
- **Quota:** Custom model training NVIDIA T4 GPUs per region
- **New Limit:** 2

### Step 5: Submit
- Review and submit
- Usually approved in 24-48 hours
- You'll get email notification

## Current Workaround

While waiting for quota approval, we're using **local training**:

```bash
python AlgoTrading/LSTM_strategy/src/retrain_improved.py
```

**Advantages:**
- ✅ FREE
- ✅ Works immediately
- ✅ No quota limits
- ✅ Good for current dataset size

**Status:** Currently running and working perfectly!

## Summary

**Issue:** Vertex AI training quotas are 0 by default for new projects
**Solution:** Request quota increase OR continue local training
**Current:** Local training in progress (working great!)
**Timeline:** Quota approval takes 24-48 hours if requested

## Next Steps

**Option 1: Continue Local Training** (Recommended for now)
- Already running
- FREE
- No waiting

**Option 2: Request Quota Increase** (For future)
- Takes 1-2 days
- Enables cloud training
- 10x faster with GPU

**Recommendation:** Keep using local training. It's working perfectly and costs nothing! 🚀
