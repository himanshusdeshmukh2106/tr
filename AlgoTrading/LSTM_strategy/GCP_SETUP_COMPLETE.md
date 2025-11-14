# ✅ GCP Setup Complete!

## Your Configuration

**Project ID:** `brave-operand-477117-n3`  
**Region:** `asia-south1` (Mumbai)  
**Bucket:** `gs://lstm-trading-asia-south1`  
**Service Account:** `lstm-trading-sa@brave-operand-477117-n3.iam.gserviceaccount.com`

## What's Been Set Up

### ✓ APIs Enabled
- Vertex AI Platform API
- Cloud Storage API
- Compute Engine API

### ✓ Service Account Created
- Name: `lstm-trading-sa`
- Roles:
  - AI Platform User (for training jobs)
  - Storage Admin (for data management)

### ✓ Cloud Storage Bucket
- Name: `lstm-trading-asia-south1`
- Location: `asia-south1`
- Storage Class: STANDARD
- Test file uploaded successfully

### ✓ Authentication
- Service account key: `lstm-trading-key.json`
- Environment variable set: `GOOGLE_APPLICATION_CREDENTIALS`
- Connection tested and verified ✓

## Files Created

```
AlgoTrading/LSTM_strategy/
├── lstm-trading-key.json       # Service account credentials (DO NOT COMMIT)
├── .env                         # Environment variables (DO NOT COMMIT)
└── test_gcp_connection.py      # Connection test script
```

## Next Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python src/data_preparation.py
```
This will:
- Download SPY data from Yahoo Finance
- Calculate 25+ technical indicators
- Create LSTM sequences
- Save to `data/` folder

### 3. Train Model (Choose One)

**Option A: Train Locally (Free, slower)**
```bash
python src/train_local.py
```
- Uses your local CPU/GPU
- Takes ~10-30 minutes
- Good for testing

**Option B: Train on GCP (Paid, faster)**
```bash
python src/train_gcp.py
```
- Uses Vertex AI with GPU
- Takes ~5-10 minutes
- Cost: ~$1-3 per training run
- Production-ready

### 4. Run Backtest
```bash
python src/backtest.py
```

### 5. Or Run Complete Pipeline
```bash
python run_pipeline.py --step all
```

## Cost Estimation

### Storage
- **Cost:** ~$0.02/GB/month
- **Expected:** <$1/month for this project

### Training (Vertex AI)
- **Machine:** n1-standard-4 + NVIDIA T4 GPU
- **Cost:** ~$1-3 per hour
- **Training time:** ~5-10 minutes per run
- **Expected:** $0.50-$1 per training session

### Predictions
- **Cost:** Pay per request
- **Expected:** Minimal for backtesting

**Total estimated cost:** <$5/month for development

## Security Notes

⚠️ **IMPORTANT:**
1. `lstm-trading-key.json` contains sensitive credentials
2. `.env` contains your configuration
3. Both are in `.gitignore` - DO NOT commit them
4. Rotate keys regularly in production
5. Use Secret Manager for production deployments

## Verify Setup

Run the test script anytime:
```bash
python test_gcp_connection.py
```

## Troubleshooting

### Authentication Error
```bash
# Set environment variable manually
export GOOGLE_APPLICATION_CREDENTIALS="lstm-trading-key.json"
# Or on Windows PowerShell:
$env:GOOGLE_APPLICATION_CREDENTIALS = "lstm-trading-key.json"
```

### Permission Denied
```bash
# Check service account permissions
gcloud projects get-iam-policy brave-operand-477117-n3
```

### Bucket Access Error
```bash
# List buckets
gsutil ls -p brave-operand-477117-n3

# Test upload
echo "test" | gsutil cp - gs://lstm-trading-asia-south1/test.txt
```

## GCP Console Links

- **Project Dashboard:** https://console.cloud.google.com/home/dashboard?project=brave-operand-477117-n3
- **Vertex AI:** https://console.cloud.google.com/vertex-ai?project=brave-operand-477117-n3
- **Cloud Storage:** https://console.cloud.google.com/storage/browser?project=brave-operand-477117-n3
- **IAM & Admin:** https://console.cloud.google.com/iam-admin?project=brave-operand-477117-n3

## Ready to Build Your Strategy! 🚀

Everything is configured and tested. Now you can:
1. Define your trading strategy
2. Customize the model architecture
3. Add custom features
4. Run experiments
5. Deploy to production

**Tell me your trading strategy and I'll implement it!**
