# Run Improved Training on GCP

## Quick Start

```bash
python AlgoTrading/LSTM_strategy/submit_improved_training.py
```

When prompted:
1. Select **1** for GPU (faster) or **2** for CPU only (cheaper)
2. Type **yes** to confirm

## What It Does

✅ Trains 4 improved models **sequentially** (respects your quota limit of 1 GPU):
1. Improved LSTM with Bidirectional layers
2. LSTM with Focal Loss (better for imbalanced data)
3. Transformer architecture
4. CNN-LSTM Hybrid

✅ Each model uses:
- Multi-head attention
- Better regularization (L1+L2)
- Class weighting for imbalanced data
- Data augmentation
- LayerNormalization

✅ Automatically saves the best model

## Expected Improvements

| Metric | Before | Target |
|--------|--------|--------|
| Accuracy | 60% | 65-75% |
| AUC | 0.61 | 0.70-0.80 |
| F1 Score | Low | Balanced |

## Monitor Progress

After submission, monitor at:
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=brave-operand-477117-n3
```

Or stream logs:
```bash
gcloud ai custom-jobs stream-logs reliance-improved-TIMESTAMP --region=asia-south1
```

## Cost
- **GPU**: ~$0.54/hour × 1-2 hours = $0.50-1.00
- **CPU**: ~$0.19/hour × 3-4 hours = $0.60-0.80

## Results Location
```
gs://lstm-trading-asia-south1/reliance/training_output/
```

The best model will be saved as `best_model.h5` with full metrics in `training_results.json`
