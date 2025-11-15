# Use Trained Model in Google Colab
# Upload this file to Colab along with your model files

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import ta

print("="*80)
print("RELIANCE STOCK PREDICTION")
print("="*80)

# 1. Load model
print("\n1. Loading model...")
model = tf.keras.models.load_model('best_model.h5')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
print(f"✓ Model loaded")
print(f"✓ Features: {len(features)}")

# 2. Load data
print("\n2. Loading data...")
df = pd.read_csv('reliance_data_5min_full_year.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
print(f"✓ Loaded {len(df)} rows")

# 3. Add features
print("\n3. Adding features...")
df['Returns'] = df['Close'].pct_change()
df['Volume_Change'] = df['Volume'].pct_change()

for period in [5, 10, 20, 50]:
    df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
    df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()

df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()

macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

bb = ta.volatility.BollingerBands(df['Close'])
df['BB_Upper'] = bb.bollinger_hband()
df['BB_Lower'] = bb.bollinger_lband()
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
df['Volume_MA'] = df['Volume'].rolling(20).mean()
df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)

df['High_20'] = df['High'].rolling(20).max()
df['Low_20'] = df['Low'].rolling(20).min()
df['Distance_High'] = (df['High_20'] - df['Close']) / df['Close']
df['Distance_Low'] = (df['Close'] - df['Low_20']) / df['Close']

df['Hour'] = df.index.hour
df['Minute'] = df.index.minute
df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

df = df.dropna()
print(f"✓ {len(df)} rows after cleaning")

# 4. Create sequences
print("\n4. Creating sequences...")
X_scaled = scaler.transform(df[features])
sequence_length = 30

X_sequences = []
valid_indices = []
for i in range(sequence_length, len(X_scaled)):
    X_sequences.append(X_scaled[i-sequence_length:i])
    valid_indices.append(i)

X_sequences = np.array(X_sequences)
print(f"✓ {len(X_sequences)} sequences")

# 5. Predict
print("\n5. Predicting...")
predictions = model.predict(X_sequences, verbose=0).flatten()

results = df.iloc[valid_indices].copy()
results['Prediction'] = predictions
results['Confidence'] = np.maximum(predictions, 1 - predictions)
results['Signal'] = 'HOLD'
results.loc[predictions > 0.7, 'Signal'] = 'BUY'
results.loc[predictions < 0.3, 'Signal'] = 'SELL'

# 6. Results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nSignals:")
for signal in ['BUY', 'HOLD', 'SELL']:
    count = (results['Signal'] == signal).sum()
    pct = count / len(results) * 100
    print(f"  {signal}: {count:,} ({pct:.1f}%)")

latest = results.iloc[-1]
print(f"\n" + "="*80)
print("LATEST PREDICTION")
print("="*80)
print(f"Date: {latest.name}")
print(f"Price: ₹{latest['Close']:.2f}")
print(f"Prediction: {latest['Prediction']:.3f}")
print(f"Confidence: {latest['Confidence']:.1%}")
print(f"Signal: {latest['Signal']}")

if latest['Signal'] == 'BUY':
    print(f"\n🟢 BUY SIGNAL")
elif latest['Signal'] == 'SELL':
    print(f"\n🔴 SELL SIGNAL")
else:
    print(f"\n⚪ HOLD")

# Save
results.to_csv("predictions.csv")
print(f"\n✓ Saved to predictions.csv")

# Download
from google.colab import files
files.download('predictions.csv')

print("\nDone! 🎉")
