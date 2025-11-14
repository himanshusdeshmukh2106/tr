"""
Configuration file for LSTM trading strategy
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Google Cloud Platform settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "brave-operand-477117-n3")
GCP_REGION = os.getenv("GCP_REGION", "asia-south1")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "lstm-trading-asia-south1")
GCP_SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY", str(BASE_DIR / "lstm-trading-key.json"))

# Model hyperparameters
LSTM_CONFIG = {
    "sequence_length": 30,  # 30 candles = 2.5 hours of 5-min data
    "lstm_units": [128, 64, 32],  # LSTM layer sizes
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
}

# Feature engineering
TECHNICAL_INDICATORS = {
    "sma_periods": [10, 20, 50, 200],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "adx_period": 14,
}

# Trading parameters
TRADING_CONFIG = {
    "initial_capital": 100000,
    "position_size": 0.1,  # 10% of capital per trade
    "max_positions": 3,
    "transaction_cost": 0.001,  # 0.1% per trade
    "slippage": 0.0005,  # 0.05%
    "stop_loss_pct": 0.02,  # 2% stop loss
    "take_profit_pct": 0.05,  # 5% take profit
}

# Backtesting
BACKTEST_CONFIG = {
    "train_period": "2020-01-01",
    "test_start": "2023-01-01",
    "test_end": "2024-08-31",
    "walk_forward_window": 252,  # Trading days (1 year)
    "retrain_frequency": 63,  # Retrain every quarter
}

# Data sources
DATA_CONFIG = {
    "ticker": "SPY",  # Default ticker (use NSE stocks for Indian market)
    "interval": "5m",  # 5-minute intraday data
    "source": "yahoo",  # yfinance
}

# Trap Strategy Configuration
TRAP_STRATEGY_CONFIG = {
    # Trading windows (IST time)
    "entry_windows": [
        {"start": "09:15", "end": "09:30"},  # Morning window
        {"start": "10:00", "end": "11:00"},  # Mid-morning window
    ],
    "exit_time": "15:15",  # Exit all positions before market close
    
    # Entry conditions
    "max_candle_body_pct": 0.20,  # Max 0.20% candle body size
    "adx_min": 20,
    "adx_max": 36,
    "ema_period": 21,
    
    # Trap detection
    "trap_lookback": 5,  # Number of candles to look back for trap
    "min_trap_distance_pct": 0.05,  # Minimum distance from EMA for valid trap
    
    # Risk management
    "stop_loss_pct": 0.5,  # 0.5% stop loss
    "target_pct": 1.0,  # 1% target
    "trailing_stop_pct": 0.3,  # 0.3% trailing stop after 0.5% profit
}
