# LSTM Trading Strategy Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│  Yahoo Finance → Feature Engineering → Sequence Creation    │
│  (OHLCV Data)    (Technical Indicators)  (LSTM Input)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Model                                │
├─────────────────────────────────────────────────────────────┤
│  Input Layer (60 timesteps × N features)                    │
│       ↓                                                      │
│  LSTM Layer 1 (128 units) + Dropout + BatchNorm            │
│       ↓                                                      │
│  LSTM Layer 2 (64 units) + Dropout + BatchNorm             │
│       ↓                                                      │
│  LSTM Layer 3 (32 units) + Dropout                         │
│       ↓                                                      │
│  Dense Layers (64 → 32)                                     │
│       ↓                                                      │
│  Output Layer (Sigmoid) → Buy/Sell Probability             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Training Options                              │
├─────────────────────────────────────────────────────────────┤
│  Local Training          │  Google Cloud Training           │
│  • CPU/GPU               │  • Vertex AI                     │
│  • Quick testing         │  • Scalable                      │
│  • Free                  │  • Production-ready              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Backtesting Engine                            │
├─────────────────────────────────────────────────────────────┤
│  • Walk-forward validation                                   │
│  • Transaction costs & slippage                             │
│  • Risk management (stop-loss, position sizing)             │
│  • Performance metrics (Sharpe, drawdown, returns)          │
└─────────────────────────────────────────────────────────────┘
```

## Features Engineered

### Price-Based
- Returns (simple & log)
- Volatility (rolling std)
- Volume changes

### Trend Indicators
- SMA (10, 20, 50, 200 periods)
- EMA (12, 26 periods)
- MACD (line, signal, histogram)
- ADX (trend strength)

### Momentum Indicators
- RSI (14 periods)

### Volatility Indicators
- Bollinger Bands (upper, middle, lower, width)
- ATR (Average True Range)

## Model Architecture Details

**Input Shape**: (60, N_features)
- 60 timesteps lookback
- N_features = ~25-30 technical indicators

**LSTM Layers**:
1. 128 units (return sequences)
2. 64 units (return sequences)
3. 32 units (final state)

**Regularization**:
- Dropout (0.3 after LSTM, 0.2 after Dense)
- Batch Normalization
- Early Stopping
- Learning Rate Reduction

**Output**: Binary classification (0 = Sell/Hold, 1 = Buy)

## Training Strategy

### Local Training
- Uses TensorFlow/Keras
- Trains on CPU or local GPU
- Suitable for experimentation
- ~10-30 minutes training time

### GCP Training
- Vertex AI Custom Training Jobs
- GPU acceleration (NVIDIA T4)
- Scalable for large datasets
- Cost: ~$1-3/hour

## Backtesting Framework

### Position Management
- Position sizing: 10% of capital per trade
- Maximum positions: 3 concurrent
- Transaction costs: 0.1%
- Slippage: 0.05%

### Risk Management
- Stop loss: 2% per trade
- Take profit: 5% per trade
- Walk-forward validation to prevent overfitting

### Performance Metrics
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Number of Trades
- Win Rate

## Workflow

1. **Data Preparation** (`data_preparation.py`)
   - Download historical data
   - Calculate technical indicators
   - Create sequences
   - Scale features

2. **Model Training** (`train_local.py` or `train_gcp.py`)
   - Build LSTM architecture
   - Train with validation
   - Save best model

3. **Backtesting** (`backtest.py`)
   - Load trained model
   - Generate signals
   - Simulate trading
   - Calculate metrics

4. **Strategy Implementation** (Your custom logic)
   - Define entry/exit rules
   - Risk management
   - Portfolio optimization

## Next: Implement Your Strategy

The framework is ready. Now you can:
- Define your specific trading rules
- Customize signal generation
- Add additional features
- Optimize hyperparameters
- Deploy to production
