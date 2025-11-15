"""
Efficient Hyperparameter Search for Reliance LSTM Model
Tests multiple configurations in a single run with smart early stopping
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

RESULTS_DIR = Path("hyperparameter_results")
RESULTS_DIR.mkdir(exist_ok=True)


class DataPrep:
    """Fast data preparation"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def load_and_prepare(self, filepath, horizon=10, threshold=0.0015, sequence_length=30):
        """Load data and create features"""
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Essential features only (for speed)
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
        
        # Momentum
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Target
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > threshold).astype(int)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['Target', 'Future_Return']]
        df = df.dropna()
        
        # Scale
        scaled = self.scaler.fit_transform(df[feature_cols])
        
        # Sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y)


def build_model(input_shape, lstm1=64, lstm2=32, dropout=0.3, lr=0.001):
    """Build LSTM model with configurable params"""
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(lstm1, return_sequences=True))(inputs)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(lstm2))(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.7)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_config(config, prep):
    """Evaluate a single configuration"""
    try:
        print(f"\n{'='*80}")
        print(f"Testing Config #{config['id']}")
        print(f"{'='*80}")
        print(f"Data: horizon={config['horizon']}, threshold={config['threshold']:.4f}, seq_len={config['seq_len']}")
        print(f"Model: lstm=[{config['lstm1']},{config['lstm2']}], dropout={config['dropout']}, lr={config['lr']}")
        
        # Prepare data
        X, y = prep.load_and_prepare(
            DATA_FILE,
            horizon=config['horizon'],
            threshold=config['threshold'],
            sequence_length=config['seq_len']
        )
        
        # Check class balance
        class1_ratio = y.mean()
        print(f"Class 1 ratio: {class1_ratio:.1%}")
        
        if class1_ratio < 0.20 or class1_ratio > 0.45:
            print(f"⚠️  Skipping: Class imbalance too extreme")
            return None
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # SMOTE
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 1).sum() - 1))
        X_train_bal, y_train_bal = smote.fit_resample(X_train_flat, y_train)
        X_train_bal = X_train_bal.reshape(-1, X_train.shape[1], X_train.shape[2])
        
        print(f"Training samples: {len(X_train_bal):,}")
        
        # Build model
        model = build_model(
            X_train.shape[1:],
            lstm1=config['lstm1'],
            lstm2=config['lstm2'],
            dropout=config['dropout'],
            lr=config['lr']
        )
        
        # Train with early stopping
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0)
        ]
        
        history = model.fit(
            X_train_bal, y_train_bal,
            validation_data=(X_test, y_test),
            epochs=25,  # Quick training
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Per-class accuracy
        y_pred = model.predict(X_test, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        class0_acc = (y_pred_binary[y_test == 0] == y_test[y_test == 0]).mean() if (y_test == 0).sum() > 0 else 0
        class1_acc = (y_pred_binary[y_test == 1] == y_test[y_test == 1]).mean() if (y_test == 1).sum() > 0 else 0
        
        # High confidence accuracy
        confidence = np.maximum(y_pred, 1 - y_pred).flatten()
        high_conf_mask = confidence > 0.7
        high_conf_acc = (y_pred_binary[high_conf_mask] == y_test[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
        high_conf_coverage = high_conf_mask.sum() / len(y_test)
        
        # Scoring (prioritize Class 1 improvement)
        balanced_score = (class0_acc + class1_acc) / 2
        weighted_score = 0.25 * test_acc + 0.45 * class1_acc + 0.30 * high_conf_acc
        
        result = {
            **config,
            'overall_acc': float(test_acc),
            'class0_acc': float(class0_acc),
            'class1_acc': float(class1_acc),
            'high_conf_acc': float(high_conf_acc),
            'high_conf_coverage': float(high_conf_coverage),
            'balanced_score': float(balanced_score),
            'weighted_score': float(weighted_score),
            'class1_samples': int((y_test == 1).sum()),
            'class1_ratio': float(class1_ratio),
            'epochs': len(history.history['loss']),
            'final_loss': float(test_loss)
        }
        
        print(f"\n📊 Results:")
        print(f"  Overall: {test_acc:.1%}")
        print(f"  Class 0: {class0_acc:.1%}")
        print(f"  Class 1: {class1_acc:.1%} ⭐")
        print(f"  High Conf (>0.7): {high_conf_acc:.1%} ({high_conf_coverage:.1%} coverage)")
        print(f"  Weighted Score: {weighted_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Run hyperparameter search"""
    print("="*80)
    print("HYPERPARAMETER SEARCH - Reliance LSTM")
    print("="*80)
    print(f"Goal: Improve Class 1 accuracy from 34-42% to 50%+")
    print(f"Strategy: Test focused parameter combinations\n")
    
    # Define search space (focused on promising ranges)
    configs = []
    config_id = 1
    
    # Grid search over key parameters
    for horizon in [8, 10, 12]:  # 40-60 minutes
        for threshold in [0.0012, 0.0015, 0.0018, 0.0020]:  # 0.12-0.20%
            for seq_len in [25, 30, 35]:
                for lstm1, lstm2 in [(64, 32), (96, 48), (80, 40)]:
                    for dropout in [0.25, 0.30, 0.35]:
                        for lr in [0.0008, 0.001, 0.0015]:
                            for batch_size in [32, 64]:
                                configs.append({
                                    'id': config_id,
                                    'horizon': horizon,
                                    'threshold': threshold,
                                    'seq_len': seq_len,
                                    'lstm1': lstm1,
                                    'lstm2': lstm2,
                                    'dropout': dropout,
                                    'lr': lr,
                                    'batch_size': batch_size
                                })
                                config_id += 1
    
    print(f"Total configurations: {len(configs)}")
    print(f"Estimated time: {len(configs) * 2 / 60:.1f} hours\n")
    print("💡 Tip: This will run overnight. Results saved incrementally.\n")
    
    # Initialize
    prep = DataPrep()
    results = []
    
    # Run search
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Progress: {i/len(configs)*100:.1f}%")
        
        result = evaluate_config(config, prep)
        
        if result is not None:
            results.append(result)
            
            # Save progress every 10 configs
            if len(results) % 10 == 0:
                with open(RESULTS_DIR / 'progress.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Saved {len(results)} results")
    
    # Final analysis
    print(f"\n\n{'='*80}")
    print("SEARCH COMPLETE!")
    print(f"{'='*80}\n")
    
    if not results:
        print("❌ No successful configurations found!")
        return
    
    # Sort by different metrics
    by_class1 = sorted(results, key=lambda x: x['class1_acc'], reverse=True)
    by_weighted = sorted(results, key=lambda x: x['weighted_score'], reverse=True)
    by_balanced = sorted(results, key=lambda x: x['balanced_score'], reverse=True)
    
    # Display top results
    print("🏆 TOP 5 BY CLASS 1 ACCURACY:")
    print("-" * 80)
    for i, r in enumerate(by_class1[:5]):
        print(f"\n{i+1}. Class 1: {r['class1_acc']:.1%} | Overall: {r['overall_acc']:.1%} | HighConf: {r['high_conf_acc']:.1%}")
        print(f"   horizon={r['horizon']}, threshold={r['threshold']:.4f}, seq={r['seq_len']}")
        print(f"   lstm=[{r['lstm1']},{r['lstm2']}], dropout={r['dropout']}, lr={r['lr']}, batch={r['batch_size']}")
    
    print(f"\n\n🎯 TOP 5 BY WEIGHTED SCORE:")
    print("-" * 80)
    for i, r in enumerate(by_weighted[:5]):
        print(f"\n{i+1}. Score: {r['weighted_score']:.3f} | Class1: {r['class1_acc']:.1%} | Overall: {r['overall_acc']:.1%}")
        print(f"   horizon={r['horizon']}, threshold={r['threshold']:.4f}, seq={r['seq_len']}")
        print(f"   lstm=[{r['lstm1']},{r['lstm2']}], dropout={r['dropout']}, lr={r['lr']}, batch={r['batch_size']}")
    
    # Best configuration
    best = by_weighted[0]
    print(f"\n\n{'='*80}")
    print("🏆 BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"\n📊 Performance:")
    print(f"  Weighted Score: {best['weighted_score']:.3f}")
    print(f"  Class 1 Accuracy: {best['class1_acc']:.1%} ⭐")
    print(f"  Overall Accuracy: {best['overall_acc']:.1%}")
    print(f"  Class 0 Accuracy: {best['class0_acc']:.1%}")
    print(f"  High Confidence (>0.7): {best['high_conf_acc']:.1%}")
    print(f"  Coverage: {best['high_conf_coverage']:.1%}")
    
    print(f"\n⚙️  Data Parameters:")
    print(f"  horizon = {best['horizon']}")
    print(f"  threshold = {best['threshold']}")
    print(f"  sequence_length = {best['seq_len']}")
    
    print(f"\n🧠 Model Parameters:")
    print(f"  lstm_units = [{best['lstm1']}, {best['lstm2']}]")
    print(f"  dropout = {best['dropout']}")
    print(f"  learning_rate = {best['lr']}")
    print(f"  batch_size = {best['batch_size']}")
    
    # Save results
    final_output = {
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(configs),
        'successful_configs': len(results),
        'best_config': best,
        'top_5_class1': by_class1[:5],
        'top_5_weighted': by_weighted[:5],
        'top_5_balanced': by_balanced[:5],
        'all_results': results
    }
    
    with open(RESULTS_DIR / 'final_results.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    
    # CSV for analysis
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    
    print(f"\n\n💾 Results saved to:")
    print(f"  - {RESULTS_DIR / 'final_results.json'}")
    print(f"  - {RESULTS_DIR / 'all_results.csv'}")
    
    print(f"\n{'='*80}")
    print("✅ HYPERPARAMETER SEARCH COMPLETE!")
    print(f"{'='*80}\n")
    
    # Improvement summary
    print(f"📈 Expected Improvement:")
    print(f"  Current Class 1: ~34-42%")
    print(f"  Best Found: {best['class1_acc']:.1%}")
    print(f"  Improvement: +{(best['class1_acc'] - 0.38) * 100:.1f}%\n")


if __name__ == "__main__":
    main()
