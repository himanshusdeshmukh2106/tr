"""
Ultra-Fast Test Optimization (30 minutes)
Tests 5 key configurations to validate approach
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import ta
from imblearn.over_sampling import SMOTE
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data file
import os
if os.path.exists("reliance_data_5min_full_year.csv"):
    DATA_FILE = "reliance_data_5min_full_year.csv"
elif os.path.exists("../../reliance_data_5min_full_year.csv"):
    DATA_FILE = "../../reliance_data_5min_full_year.csv"
else:
    DATA_FILE = "reliance_data_5min_full_year.csv"

print("="*70)
print("ULTRA-FAST TEST OPTIMIZATION (30 minutes)")
print("="*70)
print("Testing 5 strategic configurations\n")


def prepare_data(horizon, threshold, seq_len):
    """Minimal data prep"""
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Core features
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    # Target
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > threshold).astype(int)
    
    df = df.dropna()
    feature_cols = ['Returns', 'SMA_10', 'RSI', 'ATR']
    
    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    
    # Sequences
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(df['Target'].iloc[i])
    
    return np.array(X), np.array(y)


def build_model(input_shape):
    """Simple LSTM"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def test_config(name, horizon, threshold):
    """Test one config"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    print(f"horizon={horizon}, threshold={threshold:.4f}")
    
    try:
        # Prepare
        X, y = prepare_data(horizon, threshold, seq_len=30)
        
        class1_ratio = y.mean()
        print(f"Class 1: {class1_ratio:.1%}")
        
        if class1_ratio < 0.20 or class1_ratio > 0.45:
            print("⚠️  Skipped: Class imbalance")
            return None
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # SMOTE
        X_flat = X_train.reshape(X_train.shape[0], -1)
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 1).sum() - 1))
        X_bal, y_bal = smote.fit_resample(X_flat, y_train)
        X_bal = X_bal.reshape(-1, X_train.shape[1], X_train.shape[2])
        
        # Train
        model = build_model(X_train.shape[1:])
        model.fit(
            X_bal, y_bal,
            validation_data=(X_test, y_test),
            epochs=15,
            batch_size=64,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)],
            verbose=0
        )
        
        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        y_pred_bin = (y_pred > 0.5).astype(int).flatten()
        
        class0_acc = (y_pred_bin[y_test == 0] == y_test[y_test == 0]).mean()
        class1_acc = (y_pred_bin[y_test == 1] == y_test[y_test == 1]).mean()
        
        conf = np.maximum(y_pred, 1 - y_pred).flatten()
        high_conf = conf > 0.7
        hc_acc = (y_pred_bin[high_conf] == y_test[high_conf]).mean() if high_conf.sum() > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"  Overall: {acc:.1%}")
        print(f"  Class 0: {class0_acc:.1%}")
        print(f"  Class 1: {class1_acc:.1%} ⭐")
        print(f"  High Conf: {hc_acc:.1%}")
        
        return {
            'name': name,
            'horizon': horizon,
            'threshold': threshold,
            'overall': float(acc),
            'class0': float(class0_acc),
            'class1': float(class1_acc),
            'high_conf': float(hc_acc)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# Test 5 strategic configs
configs = [
    ("Current (Baseline)", 10, 0.0015),
    ("Lower Threshold", 10, 0.0012),
    ("Higher Threshold", 10, 0.0018),
    ("Shorter Horizon", 8, 0.0015),
    ("Longer Horizon", 12, 0.0015),
]

results = []
for name, horizon, threshold in configs:
    result = test_config(name, horizon, threshold)
    if result:
        results.append(result)

# Summary
print(f"\n\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}\n")

if not results:
    print("❌ No successful configs")
else:
    # Sort by Class 1
    results.sort(key=lambda x: x['class1'], reverse=True)
    
    print("Ranked by Class 1 Accuracy:")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"{i+1}. {r['name']:<20} Class1: {r['class1']:.1%}  Overall: {r['overall']:.1%}  HC: {r['high_conf']:.1%}")
    
    best = results[0]
    print(f"\n🏆 BEST: {best['name']}")
    print(f"   Class 1: {best['class1']:.1%} ⭐")
    print(f"   Parameters: horizon={best['horizon']}, threshold={best['threshold']:.4f}")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'best': best,
        'all_results': results
    }
    
    Path('test_optimization_results').mkdir(exist_ok=True)
    with open('test_optimization_results/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved to: test_optimization_results/results.json")

print(f"\n{'='*70}")
print("✅ TEST COMPLETE!")
print(f"{'='*70}")
print("\nNext steps:")
print("1. If Class 1 > 45%: Run quick_optimization.py (2-3 hours)")
print("2. If Class 1 > 50%: You found a good config! ⭐")
print("3. If Class 1 < 45%: Run hyperparameter_search.py (overnight)\n")
