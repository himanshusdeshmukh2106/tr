"""
Quick Hyperparameter Optimization (2-3 hours)
Tests strategic combinations based on analysis insights
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, GRU
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

RESULTS_DIR = Path("quick_optimization_results")
RESULTS_DIR.mkdir(exist_ok=True)


class QuickDataPrep:
    """Minimal data prep for speed"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def prepare(self, filepath, horizon, threshold, seq_len):
        """Load and prepare data quickly"""
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Core features
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Moving averages
        df['SMA_5'] = ta.trend.SMAIndicator(df['Close'], window=5).sma_indicator()
        df['SMA_10'] = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        
        # Momentum
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MOM'] = df['Close'].pct_change(periods=5)
        
        # Volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        # Target
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > threshold).astype(int)
        
        # Clean
        feature_cols = [col for col in df.columns if col not in ['Target', 'Future_Return']]
        df = df.dropna()
        
        # Scale
        scaled = self.scaler.fit_transform(df[feature_cols])
        
        # Sequences
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y)


def build_lstm(input_shape, units1, units2, dropout, lr):
    """Standard LSTM"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(units1, return_sequences=True))(inputs)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(units2))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.7)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_gru(input_shape, units1, units2, dropout, lr):
    """GRU alternative (faster than LSTM)"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(units1, return_sequences=True))(inputs)
    x = Dropout(dropout)(x)
    x = Bidirectional(GRU(units2))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.7)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_deep_lstm(input_shape, units, dropout, lr):
    """Deeper LSTM with 3 layers"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(units // 2, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(units // 4))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def test_config(config, prep):
    """Test a single configuration"""
    try:
        print(f"\n{'='*70}")
        print(f"Config #{config['id']}: {config['name']}")
        print(f"{'='*70}")
        print(f"horizon={config['horizon']}, threshold={config['threshold']:.4f}, seq={config['seq_len']}")
        print(f"model={config['model_type']}, units={config['units']}, dropout={config['dropout']}, lr={config['lr']}")
        
        # Prepare data
        X, y = prep.prepare(DATA_FILE, config['horizon'], config['threshold'], config['seq_len'])
        
        class1_ratio = y.mean()
        print(f"Class 1: {class1_ratio:.1%}")
        
        if class1_ratio < 0.22 or class1_ratio > 0.42:
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
        
        # Build model
        if config['model_type'] == 'lstm':
            model = build_lstm(X_train.shape[1:], config['units'][0], config['units'][1], config['dropout'], config['lr'])
        elif config['model_type'] == 'gru':
            model = build_gru(X_train.shape[1:], config['units'][0], config['units'][1], config['dropout'], config['lr'])
        else:  # deep_lstm
            model = build_deep_lstm(X_train.shape[1:], config['units'][0], config['dropout'], config['lr'])
        
        # Train
        history = model.fit(
            X_bal, y_bal,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=config['batch_size'],
            callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)],
            verbose=0
        )
        
        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        y_pred_bin = (y_pred > 0.5).astype(int).flatten()
        
        class0_acc = (y_pred_bin[y_test == 0] == y_test[y_test == 0]).mean() if (y_test == 0).sum() > 0 else 0
        class1_acc = (y_pred_bin[y_test == 1] == y_test[y_test == 1]).mean() if (y_test == 1).sum() > 0 else 0
        
        conf = np.maximum(y_pred, 1 - y_pred).flatten()
        high_conf = conf > 0.7
        hc_acc = (y_pred_bin[high_conf] == y_test[high_conf]).mean() if high_conf.sum() > 0 else 0
        
        # Score (prioritize Class 1)
        score = 0.2 * acc + 0.5 * class1_acc + 0.3 * hc_acc
        
        result = {
            **config,
            'overall': float(acc),
            'class0': float(class0_acc),
            'class1': float(class1_acc),
            'high_conf': float(hc_acc),
            'score': float(score),
            'epochs': len(history.history['loss'])
        }
        
        print(f"Overall: {acc:.1%} | Class0: {class0_acc:.1%} | Class1: {class1_acc:.1%} ⭐ | HC: {hc_acc:.1%}")
        print(f"Score: {score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Run quick optimization"""
    print("="*70)
    print("QUICK HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print("Strategy: Test strategic combinations (2-3 hours)\n")
    
    # Strategic configurations based on analysis
    configs = [
        # Baseline variations
        {'id': 1, 'name': 'Baseline', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30, 
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 2, 'name': 'Lower Threshold', 'horizon': 10, 'threshold': 0.0012, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 3, 'name': 'Higher Threshold', 'horizon': 10, 'threshold': 0.0018, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        # Horizon variations
        {'id': 4, 'name': 'Shorter Horizon', 'horizon': 8, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 5, 'name': 'Longer Horizon', 'horizon': 12, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        # Model size variations
        {'id': 6, 'name': 'Larger Model', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [96, 48], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 7, 'name': 'Smaller Model', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [48, 24], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        # Regularization
        {'id': 8, 'name': 'Less Dropout', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.2, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 9, 'name': 'More Dropout', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.4, 'lr': 0.001, 'batch_size': 64},
        
        # Learning rate
        {'id': 10, 'name': 'Lower LR', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.0005, 'batch_size': 64},
        
        {'id': 11, 'name': 'Higher LR', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.002, 'batch_size': 64},
        
        # Sequence length
        {'id': 12, 'name': 'Shorter Sequence', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 20,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 13, 'name': 'Longer Sequence', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 40,
         'model_type': 'lstm', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        # Alternative architectures
        {'id': 14, 'name': 'GRU Model', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'gru', 'units': [64, 32], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        {'id': 15, 'name': 'Deep LSTM', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'deep_lstm', 'units': [64, 0], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        
        # Best combinations from analysis
        {'id': 16, 'name': 'Optimal Combo 1', 'horizon': 10, 'threshold': 0.0014, 'seq_len': 28,
         'model_type': 'lstm', 'units': [80, 40], 'dropout': 0.28, 'lr': 0.0012, 'batch_size': 48},
        
        {'id': 17, 'name': 'Optimal Combo 2', 'horizon': 11, 'threshold': 0.0016, 'seq_len': 32,
         'model_type': 'lstm', 'units': [72, 36], 'dropout': 0.32, 'lr': 0.0008, 'batch_size': 56},
        
        {'id': 18, 'name': 'Aggressive', 'horizon': 12, 'threshold': 0.0020, 'seq_len': 35,
         'model_type': 'lstm', 'units': [96, 48], 'dropout': 0.25, 'lr': 0.0015, 'batch_size': 32},
        
        {'id': 19, 'name': 'Conservative', 'horizon': 8, 'threshold': 0.0010, 'seq_len': 25,
         'model_type': 'lstm', 'units': [48, 24], 'dropout': 0.35, 'lr': 0.0005, 'batch_size': 96},
        
        {'id': 20, 'name': 'Balanced GRU', 'horizon': 10, 'threshold': 0.0015, 'seq_len': 30,
         'model_type': 'gru', 'units': [80, 40], 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
    ]
    
    print(f"Total configs: {len(configs)}")
    print(f"Estimated time: {len(configs) * 8 / 60:.1f} hours\n")
    
    # Run tests
    prep = QuickDataPrep()
    results = []
    
    for config in configs:
        result = test_config(config, prep)
        if result:
            results.append(result)
            
            # Save progress
            with open(RESULTS_DIR / 'progress.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Analysis
    print(f"\n\n{'='*70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*70}\n")
    
    if not results:
        print("❌ No successful configs")
        return
    
    # Sort
    by_class1 = sorted(results, key=lambda x: x['class1'], reverse=True)
    by_score = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print("🏆 TOP 5 BY CLASS 1 ACCURACY:")
    print("-" * 70)
    for i, r in enumerate(by_class1[:5]):
        print(f"{i+1}. {r['name']}: Class1={r['class1']:.1%}, Overall={r['overall']:.1%}, Score={r['score']:.3f}")
    
    print(f"\n🎯 TOP 5 BY WEIGHTED SCORE:")
    print("-" * 70)
    for i, r in enumerate(by_score[:5]):
        print(f"{i+1}. {r['name']}: Score={r['score']:.3f}, Class1={r['class1']:.1%}, Overall={r['overall']:.1%}")
    
    # Best
    best = by_score[0]
    print(f"\n\n{'='*70}")
    print("🏆 BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Name: {best['name']}")
    print(f"\nPerformance:")
    print(f"  Score: {best['score']:.3f}")
    print(f"  Class 1: {best['class1']:.1%} ⭐")
    print(f"  Overall: {best['overall']:.1%}")
    print(f"  High Conf: {best['high_conf']:.1%}")
    print(f"\nParameters:")
    print(f"  horizon = {best['horizon']}")
    print(f"  threshold = {best['threshold']}")
    print(f"  sequence_length = {best['seq_len']}")
    print(f"  model_type = '{best['model_type']}'")
    print(f"  units = {best['units']}")
    print(f"  dropout = {best['dropout']}")
    print(f"  learning_rate = {best['lr']}")
    print(f"  batch_size = {best['batch_size']}")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'best': best,
        'top_5_class1': by_class1[:5],
        'top_5_score': by_score[:5],
        'all_results': results
    }
    
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    pd.DataFrame(results).to_csv(RESULTS_DIR / 'results.csv', index=False)
    
    print(f"\n💾 Saved to: {RESULTS_DIR}/")
    print(f"\n{'='*70}")
    print("✅ DONE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
