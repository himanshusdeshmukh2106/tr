# Quick Command Reference

## Setup Verification
```bash
# Test GCP connection
python test_gcp_connection.py

# Check GCP project
gcloud config get-value project

# List buckets
gsutil ls -p brave-operand-477117-n3

# Check service accounts
gcloud iam service-accounts list
```

## Data Pipeline
```bash
# Download and prepare data
python src/data_preparation.py

# Check prepared data
ls data/
```

## Training
```bash
# Train locally (free)
python src/train_local.py

# Train on GCP (paid)
python src/train_gcp.py

# Run full pipeline
python run_pipeline.py --step all
python run_pipeline.py --step data
python run_pipeline.py --step train
```

## Backtesting
```bash
# Run backtest
python src/backtest.py

# View results
ls results/
```

## GCP Management
```bash
# Upload file to bucket
gsutil cp local_file.txt gs://lstm-trading-asia-south1/

# Download from bucket
gsutil cp gs://lstm-trading-asia-south1/file.txt ./

# List bucket contents
gsutil ls gs://lstm-trading-asia-south1/

# Delete file
gsutil rm gs://lstm-trading-asia-south1/file.txt

# Check Vertex AI jobs
gcloud ai custom-jobs list --region=asia-south1
```

## Environment
```bash
# Set credentials (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS = "lstm-trading-key.json"

# Set credentials (Bash)
export GOOGLE_APPLICATION_CREDENTIALS="lstm-trading-key.json"

# View environment
cat .env
```

## Python Quick Tests
```python
# Test imports
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "from google.cloud import storage; print('GCP OK')"
python -c "import yfinance as yf; print('yfinance OK')"

# Quick data check
python -c "import pandas as pd; df = pd.read_csv('data/processed_data.csv'); print(df.shape)"
```

## Monitoring
```bash
# Watch training logs
tail -f logs/*.log

# Check model files
ls models/

# View training history
python -c "import pandas as pd; print(pd.read_csv('models/training_history.csv'))"
```

## Cleanup (if needed)
```bash
# Delete bucket contents
gsutil -m rm -r gs://lstm-trading-asia-south1/*

# Delete service account key
gcloud iam service-accounts keys delete KEY_ID --iam-account=lstm-trading-sa@brave-operand-477117-n3.iam.gserviceaccount.com

# Delete service account
gcloud iam service-accounts delete lstm-trading-sa@brave-operand-477117-n3.iam.gserviceaccount.com
```
