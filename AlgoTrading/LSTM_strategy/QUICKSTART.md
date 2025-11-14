# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
cd AlgoTrading/LSTM_strategy
pip install -r requirements.txt
```

### 2. Configure Environment (Optional - for GCP)
```bash
cp .env.example .env
# Edit .env with your GCP credentials
```

## Local Training (Recommended for Testing)

### Run Complete Pipeline
```bash
python run_pipeline.py --step all
```

### Or Run Steps Individually

**Step 1: Prepare Data**
```bash
python src/data_preparation.py
```
This will:
- Download SPY data from Yahoo Finance
- Add technical indicators
- Create sequences for LSTM
- Save processed data to `data/` folder

**Step 2: Train Model Locally**
```bash
python src/train_local.py
```
This will:
- Load prepared sequences
- Build and train LSTM model
- Save best model to `models/` folder
- Training takes ~10-30 minutes on CPU, ~5 minutes on GPU

**Step 3: Run Backtest**
```bash
python src/backtest.py
```

## Google Cloud Training (For Production)

### Prerequisites
1. GCP account with billing enabled
2. Service account with AI Platform permissions
3. Cloud Storage bucket created

### Setup GCP
Follow instructions in `config/gcp_setup.md`

### Run Training on GCP
```bash
python src/train_gcp.py
```

## Project Structure
```
LSTM_strategy/
├── config/              # Configuration files
│   ├── config.py       # Main config
│   └── gcp_setup.md    # GCP setup guide
├── src/                # Source code
│   ├── data_preparation.py
│   ├── lstm_model.py
│   ├── train_local.py
│   ├── train_gcp.py
│   └── backtest.py
├── data/               # Data files (created)
├── models/             # Trained models (created)
├── results/            # Backtest results (created)
└── run_pipeline.py     # Main pipeline runner
```

## Next Steps

After setup, you can:
1. Modify strategy parameters in `config/config.py`
2. Implement your custom strategy logic
3. Add more features or indicators
4. Experiment with different model architectures
5. Run walk-forward optimization

## Customization

### Change Ticker
Edit `config/config.py`:
```python
DATA_CONFIG = {
    "ticker": "AAPL",  # Change to any ticker
    ...
}
```

### Adjust Model Parameters
Edit `config/config.py`:
```python
LSTM_CONFIG = {
    "sequence_length": 60,
    "lstm_units": [128, 64, 32],
    ...
}
```

### Modify Trading Rules
Edit `config/config.py`:
```python
TRADING_CONFIG = {
    "position_size": 0.2,  # 20% per trade
    "stop_loss_pct": 0.03,  # 3% stop loss
    ...
}
```

## Ready for Your Strategy!
The framework is set up. Now tell me your trading strategy and I'll implement it! 🚀
