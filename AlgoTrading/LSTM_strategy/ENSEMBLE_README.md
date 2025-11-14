# Ensemble Trading Model

Advanced ensemble model combining LSTM, XGBoost, Random Forest, and LightGBM for NIFTY 50 prediction.

## Models Included

1. **LSTM (Deep Learning)**
   - Bidirectional LSTM with attention
   - Captures temporal patterns
   - Best for sequence modeling

2. **XGBoost**
   - Gradient boosting
   - Handles non-linear relationships
   - Feature importance analysis

3. **Random Forest**
   - Ensemble of decision trees
   - Robust to overfitting
   - Good for noisy data

4. **LightGBM**
   - Fast gradient boosting
   - Efficient memory usage
   - High accuracy

## Data

- **Source**: NIFTY 50 Index (2021-2025)
- **Samples**: 882 sequences
- **Features**: 27 technical indicators
- **Sequence Length**: 60 days

## Training

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python AlgoTrading/LSTM_strategy/prepare_nifty_data.py

# Train ensemble
python AlgoTrading/LSTM_strategy/train_ensemble.py
```

## Ensemble Strategy

The ensemble uses **weighted voting** based on individual model AUC scores:
- Each model predicts probability (0-1)
- Predictions are weighted by validation AUC
- Final prediction = weighted average

## Expected Performance

Individual models typically achieve:
- LSTM: 50-60% accuracy
- XGBoost: 55-65% accuracy
- Random Forest: 50-60% accuracy
- LightGBM: 55-65% accuracy

**Ensemble**: 60-70% accuracy (better than any single model)

## Files

```
LSTM_strategy/
├── prepare_nifty_data.py      # Data preparation
├── train_ensemble.py          # Ensemble training
├── data/
│   ├── X_nifty50.npy         # Features
│   ├── y_nifty50.npy         # Labels
│   └── scaler_nifty50.pkl    # Scaler
└── models/
    └── ensemble/
        ├── lstm_model.h5      # LSTM weights
        ├── xgb_model.pkl      # XGBoost
        ├── rf_model.pkl       # Random Forest
        ├── lgbm_model.pkl     # LightGBM
        └── weights.pkl        # Ensemble weights
```

## Usage

```python
from train_ensemble import EnsembleModel
import numpy as np

# Load ensemble
ensemble = EnsembleModel()
# ... load models ...

# Predict
X_new = np.load("new_data.npy")
predictions = ensemble.predict_proba(X_new)
```

## Technical Indicators Used

- Moving Averages (SMA 10, 20, 50)
- Exponential Moving Averages (EMA 12, 21, 26)
- RSI (14 periods)
- MACD
- Bollinger Bands
- ATR (Average True Range)
- ADX (Average Directional Index)
- Volume indicators
- Price momentum
- Volatility measures

## Limitations

- **Market efficiency**: Hard to beat 60-70% accuracy
- **Daily data**: Intraday would be better
- **No fundamentals**: Only technical analysis
- **Overfitting risk**: Requires regular retraining

## Next Steps

1. Add more data sources (news, sentiment)
2. Use intraday 5-min candles
3. Implement risk management
4. Backtest with transaction costs
5. Deploy for paper trading

## GCP Training (Optional)

For faster training with GPU:

```bash
python AlgoTrading/LSTM_strategy/submit_improved_training.py
```

Requires GCP quota for Vertex AI.
