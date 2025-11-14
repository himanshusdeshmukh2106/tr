# LSTM-Based Trading Strategy with Google Cloud

## Overview
This strategy uses Long Short-Term Memory (LSTM) neural networks to predict price movements and generate trading signals. The model is trained on Google Cloud Platform for scalability.

## Architecture
- **Model**: LSTM with attention mechanism
- **Training**: Google Cloud AI Platform / Vertex AI
- **Features**: Technical indicators + price sequences
- **Backtesting**: Walk-forward validation with transaction costs

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Configure GCP credentials (see `config/gcp_setup.md`)
3. Prepare data: `python data_preparation.py`
4. Train model: `python train_gcp.py`
5. Backtest: `python backtest.py`

## Performance Metrics
- Cumulative Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

## Status
🚧 In Development
