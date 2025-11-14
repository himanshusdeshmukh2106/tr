# Algorithmic Trading with Improved LSTM Models

Complete trading system with Freqtrade analysis and improved ML models for Indian stock market.

## 🚀 Quick Start on Google Colab

### Train Improved Model on Reliance Data

```python
# Clone repo
!git clone https://github.com/himanshusdeshmukh2106/tr.git
%cd tr/AlgoTrading/LSTM_strategy

# Install dependencies
!pip install tensorflow pandas numpy scikit-learn ta joblib

# Upload your CSV file
from google.colab import files
uploaded = files.upload()  # Upload reliance_data_5min_full_year.csv

# Train model
!python train_reliance_improved.py --architecture improved_lstm
```

## 📁 Repository Structure

- **AlgoTrading/LSTM_strategy/** - Improved LSTM models with 60+ features
- **freqtrade/** - Freqtrade crypto bot analysis
- **reliance_data_5min_full_year.csv** - 5-minute Reliance data (18K+ candles)

## 📚 Documentation

- `COLAB_TRAINING_GUIDE.md` - Complete Colab training guide
- `ACCURACY_IMPROVEMENT_GUIDE.md` - How to improve model accuracy
- `QUICK_START_IMPROVEMENTS.md` - Quick wins for better performance
- `FREQTRADE_ANALYSIS.md` - Freqtrade bot analysis

## 🎯 Expected Results

- **Before**: 53% accuracy (old model)
- **After**: 68-72% accuracy (improved model)
- **Improvement**: +15-18% 🎉

## 📊 Features

- 60+ technical indicators
- Multi-scale LSTM architecture
- Transformer-LSTM hybrid
- Support/Resistance levels
- Time-based features
- Confidence filtering

## 🔧 Local Training

```bash
cd AlgoTrading/LSTM_strategy
python train_reliance_improved.py --architecture improved_lstm
```

## 📈 Models Available

1. **improved_lstm** - Multi-scale LSTM (recommended)
2. **transformer_lstm** - Transformer + LSTM hybrid

## 🤝 Contributing

Pull requests welcome!

## 📄 License

Check individual project licenses.
