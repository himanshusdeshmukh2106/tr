# Quota Increase Required

## Issue
Your GCP project has **0 quota** for Vertex AI training:
- `aiplatform.googleapis.com/custom_model_training_cpus` = 0
- `aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus` = 0

## Solution: Request Quota Increase

### Step 1: Go to Quotas Page
```
https://console.cloud.google.com/iam-admin/quotas?project=brave-operand-477117-n3
```

### Step 2: Filter for Vertex AI Quotas
1. In the search box, type: `custom_model_training`
2. Look for these quotas in **asia-south1** region:
   - **Custom model training CPUs per region**
   - **Custom model training NVIDIA T4 GPUs per region**

### Step 3: Request Increase
1. Check the boxes next to both quotas
2. Click "EDIT QUOTAS" at the top
3. Request:
   - **CPUs**: 4 (for n1-standard-4)
   - **GPUs**: 1 (for Tesla T4)
4. Fill in the form explaining you need it for ML model training
5. Submit

### Step 4: Wait for Approval
- Usually takes 1-2 business days
- You'll get an email when approved

## Alternative: Use Free Tier Options

### Option 1: Train Locally (No Quota Needed)
```bash
python AlgoTrading/LSTM_strategy/src/train_local.py
```
- Uses your local machine
- No GCP costs
- Slower without GPU

### Option 2: Use Google Colab (Free GPU)
1. Go to https://colab.research.google.com/
2. Upload your training script
3. Runtime > Change runtime type > GPU (T4)
4. Run training for free

### Option 3: Use Kaggle (Free GPU)
1. Go to https://www.kaggle.com/
2. Create a new notebook
3. Settings > Accelerator > GPU T4 x2
4. Get 30 hours/week free

## Current Workaround

Since you can't use GCP right now, let's train locally with the improved model:

```bash
# Train locally with improved architecture
python AlgoTrading/LSTM_strategy/train_improved_local.py
```

I can create this script for you if you want to proceed with local training while waiting for quota approval.

## Check Current Quotas
```bash
gcloud compute project-info describe --project=brave-operand-477117-n3
```

Or visit:
```
https://console.cloud.google.com/iam-admin/quotas?project=brave-operand-477117-n3&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Service_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22Vertex%2520AI%2520API_5C_22_22%257D%255D%22))
```
