# RELIANCE LSTM Training Status

## ✅ Training Started!

**Date:** November 12, 2025  
**Stock:** RELIANCE (NSE)  
**Timeframe:** Daily (1D)  
**Platform:** Google Cloud Platform (asia-south1)

## 📊 Data Summary

**Source File:** `Quote-Equity-RELIANCE-EQ-12-11-2024-to-12-11-2025.csv`

**Data Statistics:**
- Total days: 247
- After indicators: 197 days
- Training sequences: 137
- Sequence length: 60 days
- Features: 27 technical indicators

**Target Distribution:**
- Up days (1): 71 (51.8%)
- Down days (0): 66 (48.2%)
- Well balanced dataset ✓

## 🎯 Model Architecture

### Refined LSTM with Attention

**Layers:**
1. LSTM Layer 1: 128/256/192 units (configurable)
2. LSTM Layer 2: 64/128/96 units
3. Attention Mechanism
4. LSTM Layer 3: 32/64/48 units
5. Dense layers: 64 → 32
6. Output: Sigmoid (binary classification)

**Regularization:**
- L2 regularization on all LSTM layers
- Dropout: 0.3-0.4
- Batch Normalization
- Early Stopping (patience: 15)
- Learning Rate Reduction

## 🔧 Hyperparameter Search

Testing 3 configurations:

### Config 1: Baseline
- LSTM units: [128, 64, 32]
- Dropout: 0.3
- Learning rate: 0.001
- Batch size: 16
- Epochs: 150

### Config 2: Deeper
- LSTM units: [256, 128, 64]
- Dropout: 0.4
- Learning rate: 0.0005
- Batch size: 16
- Epochs: 150

### Config 3: Wider
- LSTM units: [192, 96, 48]
- Dropout: 0.35
- Learning rate: 0.0008
- Batch size: 8
- Epochs: 150

## 📈 Features Used (27 total)

### Price Data
- Open, High, Low, Close, Volume

### Moving Averages
- SMA: 10, 20, 50 periods
- EMA: 12, 21, 26 periods

### Momentum Indicators
- RSI (14 periods)
- MACD (line, signal, histogram)

### Volatility Indicators
- Bollinger Bands (upper, middle, lower, width)
- ATR (Average True Range)

### Trend Indicators
- ADX (Average Directional Index)

### Derived Features
- Returns (simple & log)
- Volume change
- Volatility (20-day rolling std)
- Price vs SMA20
- Price vs EMA21

## 🎓 Training Process

1. **Data Upload to GCS** ✓
   - Location: `gs://lstm-trading-asia-south1/reliance/`
   
2. **Hyperparameter Search** (In Progress)
   - Training 3 different configurations
   - Evaluating on validation set
   - Selecting best model based on F1 score

3. **Model Evaluation Metrics**
   - Validation Loss
   - Validation Accuracy
   - AUC (Area Under Curve)
   - Precision
   - Recall
   - F1 Score

4. **Best Model Selection**
   - Highest F1 score wins
   - Saved locally and uploaded to GCS

## 📁 Output Files

### Local Files
- `data/reliance_daily_processed.csv` - Processed data with indicators
- `data/X_reliance_daily.npy` - Input sequences
- `data/y_reliance_daily.npy` - Target labels
- `data/scaler_reliance_daily.pkl` - Feature scaler
- `models/reliance_lstm_best_final.h5` - Best trained model
- `models/hyperparameter_results.json` - All config results

### GCS Files
- `gs://lstm-trading-asia-south1/reliance/X_reliance_daily.npy`
- `gs://lstm-trading-asia-south1/reliance/y_reliance_daily.npy`
- `gs://lstm-trading-asia-south1/reliance/scaler_reliance_daily.pkl`
- `gs://lstm-trading-asia-south1/reliance/models/reliance_lstm_best_final.h5`

## 🚀 Next Steps

After training completes:

1. **Evaluate Model Performance**
   - Check validation metrics
   - Analyze predictions
   - Review confusion matrix

2. **Backtest Strategy**
   - Test on historical data
   - Calculate returns
   - Measure risk metrics

3. **Refine Model** (if needed)
   - Adjust hyperparameters
   - Add/remove features
   - Try different architectures

4. **Deploy for Trading**
   - Real-time predictions
   - Integration with broker API
   - Risk management system

## 📊 Expected Results

Based on similar models:

**Typical Performance:**
- Validation Accuracy: 55-65%
- AUC: 0.60-0.70
- F1 Score: 0.55-0.65

**Good Performance:**
- Validation Accuracy: 65-75%
- AUC: 0.70-0.80
- F1 Score: 0.65-0.75

**Excellent Performance:**
- Validation Accuracy: >75%
- AUC: >0.80
- F1 Score: >0.75

## ⏱️ Training Time

**Estimated Duration:**
- Per configuration: ~10-20 minutes
- Total (3 configs): ~30-60 minutes
- Depends on: CPU/GPU, data size, epochs

## 💰 GCP Costs

**Current Setup:**
- Storage: ~$0.01/month
- Training: Local (free)
- Data transfer: Minimal

**For GPU Training (future):**
- n1-standard-4 + T4 GPU: ~$1-3/hour
- Estimated cost per run: $0.50-$1.00

## 📝 Notes

- Training is running locally with GCS integration
- Model will be automatically uploaded to GCS
- All hyperparameter results will be saved
- Best model selected based on F1 score
- Can be deployed for real-time predictions

## 🔄 Continuous Improvement

The model will be refined through:
1. Regular retraining with new data
2. Hyperparameter optimization
3. Feature engineering
4. Architecture improvements
5. Ensemble methods

---

**Status:** Training in progress...  
**Check:** Run `python AlgoTrading/LSTM_strategy/src/train_reliance_gcp.py`
