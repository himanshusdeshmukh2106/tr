"""
COMPREHENSIVE HYPERPARAMETER OPTIMIZATION FOR 3-CLASS MODEL
Tests multiple combinations to find the best setup for predicting UP/DOWN moves
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import ta
from imblearn.over_sampling import SMOTE
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
import os
if os.path.exists("reliance_data_5min_full_year.csv"):
    DATA_FILE = "reliance_data_5min_full_year.csv"
elif os.path.exists("../../reliance_data_5min_full_year.csv"):
    DATA_FILE = "../../reliance_data_5min_full_year.csv"
else:
    DATA_FILE = "reliance_data_5min_full_year.csv"

RESULTS_DIR = Path("optimization_results")
RESULTS_DIR.mkdir(exist_ok=True)


class DataPrep:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    
    def add_features(self, df):
        # Candle characteristics
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        
        # Price action
        df['Returns'] = df['Close'].pct_change()
        df['Returns_2'] = df['Close'].pct_change(2)
        df['Returns_5'] = df['Close'].pct_change(5)
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['Price_to_SMA5'] = df['Close'] / df['SMA_5']
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_EMA21'] = df['Close'] / df['EMA_21']
        
        # Trend strength
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Volume
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        # Price levels
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Distance_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_Low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        # Time features
        df['Hour'] = df.index.hour
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        return df
    
    def create_target(self, df, horizon, threshold):
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = 1  # Neutral
        df.loc[df['Future_Return'] < -threshold, 'Target'] = 0  # Down
        df.loc[df['Future_Return'] > threshold, 'Target'] = 2   # Up
        return df
    
    def prepare_sequences(self, df, sequence_length):
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['Target', 'Future_Return']]
        
        df = df.dropna()
        
        # Remove correlated features
        corr_matrix = df[feature_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        feature_columns = [col for col in feature_columns if col not in to_drop]
        
        # Scale
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y)


def build_model(input_shape, lstm_units, dropout, learning_rate):
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True))(inputs)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(lstm_units[1]))(x)
    x = Dropout(dropout)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.7)(x)
    
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_config(config, X_train, y_train, X_test, y_test):
    """Train and evaluate a single configuration"""
    
    print(f"\n{'='*80}")
    print(f"Testing Config: {config}")
    print(f"{'='*80}")
    
    # Apply SMOTE
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_balanced, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Build model
    model = build_model(
        X_train.shape[1:],
        config['lstm_units'],
        config['dropout'],
        config['learning_rate']
    )
    
    # Class weights
    class_weight = {0: config['class_weight_down'], 1: 1.0, 2: config['class_weight_up']}
    
    # Train
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0)
    ]
    
    history = model.fit(
        X_train_balanced, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=30,
        batch_size=config['batch_size'],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0
    )
    
    # Evaluate
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    overall_acc = (y_pred == y_test).mean()
    
    class_accs = {}
    class_counts = {}
    for class_val in [0, 1, 2]:
        mask = y_test == class_val
        if mask.sum() > 0:
            class_accs[class_val] = (y_pred[mask] == y_test[mask]).mean()
            class_counts[class_val] = mask.sum()
        else:
            class_accs[class_val] = 0
            class_counts[class_val] = 0
    
    # High confidence metrics
    confidence = np.max(y_pred_probs, axis=1)
    high_conf_mask = confidence > 0.7
    high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
    
    # Count predictions at high confidence
    high_conf_preds = {}
    for class_val in [0, 1, 2]:
        high_conf_preds[class_val] = ((y_pred == class_val) & high_conf_mask).sum()
    
    # Custom score: Prioritize UP and DOWN accuracy
    # Penalize if model predicts mostly NEUTRAL
    trading_score = (class_accs[0] * 0.4 + class_accs[2] * 0.4 + class_accs[1] * 0.2)
    
    # Bonus if we get decent number of trading signals
    signal_ratio = (high_conf_preds[0] + high_conf_preds[2]) / (high_conf_preds[1] + 1)
    trading_score *= (1 + min(signal_ratio, 0.5))  # Up to 50% bonus
    
    results = {
        'config': config,
        'overall_accuracy': float(overall_acc),
        'class_0_accuracy': float(class_accs[0]),
        'class_1_accuracy': float(class_accs[1]),
        'class_2_accuracy': float(class_accs[2]),
        'class_counts': class_counts,
        'high_conf_accuracy': float(high_conf_acc),
        'high_conf_predictions': high_conf_preds,
        'trading_score': float(trading_score),
        'epochs_trained': len(history.history['loss'])
    }
    
    print(f"Overall Acc: {overall_acc:.3f}")
    print(f"DOWN Acc: {class_accs[0]:.3f} | NEUTRAL Acc: {class_accs[1]:.3f} | UP Acc: {class_accs[2]:.3f}")
    print(f"High Conf (>0.7): DOWN={high_conf_preds[0]}, NEUTRAL={high_conf_preds[1]}, UP={high_conf_preds[2]}")
    print(f"Trading Score: {trading_score:.3f}")
    
    return results


def run_optimization():
    """Run comprehensive hyperparameter search"""
    
    print("="*80)
    print("COMPREHENSIVE 3-CLASS HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print("\nTesting multiple combinations of:")
    print("- Thresholds: 0.2%, 0.25%, 0.3%, 0.4%, 0.5%")
    print("- Horizons: 5, 8, 10, 12 candles")
    print("- LSTM units, dropout, learning rate, class weights")
    print()
    
    # Load and prepare base data
    prep = DataPrep()
    df = prep.load_data(DATA_FILE)
    df = prep.add_features(df)
    
    # Define search space
    configs = []
    
    # Test different thresholds and horizons
    for threshold in [0.002, 0.0025, 0.003, 0.004, 0.005]:  # 0.2% to 0.5%
        for horizon in [5, 8, 10, 12]:
            for lstm_config in [[64, 32], [128, 64], [96, 48]]:
                for dropout in [0.2, 0.3, 0.4]:
                    for lr in [0.001, 0.0005]:
                        for class_weight in [1.5, 2.0, 2.5, 3.0]:
                            configs.append({
                                'threshold': threshold,
                                'horizon': horizon,
                                'sequence_length': 30,
                                'lstm_units': lstm_config,
                                'dropout': dropout,
                                'learning_rate': lr,
                                'batch_size': 64,
                                'class_weight_down': class_weight,
                                'class_weight_up': class_weight
                            })
    
    print(f"Total configurations to test: {len(configs)}")
    print("This will take a while... grab a coffee!\n")
    
    # Test top configs (limit to avoid excessive runtime)
    # Sample intelligently
    import random
    random.seed(42)
    if len(configs) > 50:
        # Sample diverse configs
        sampled_configs = random.sample(configs, 50)
    else:
        sampled_configs = configs
    
    all_results = []
    
    for i, config in enumerate(sampled_configs, 1):
        print(f"\n[{i}/{len(sampled_configs)}] ", end="")
        
        try:
            # Prepare data with this config
            df_copy = df.copy()
            df_copy = prep.create_target(df_copy, config['horizon'], config['threshold'])
            X, y = prep.prepare_sequences(df_copy, config['sequence_length'])
            
            # Check class distribution
            class_dist = {0: (y==0).sum(), 1: (y==1).sum(), 2: (y==2).sum()}
            
            # Skip if too imbalanced (>95% neutral)
            if class_dist[1] / len(y) > 0.95:
                print(f"Skipped (too neutral: {class_dist[1]/len(y)*100:.1f}%)")
                continue
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Evaluate
            results = evaluate_config(config, X_train, y_train, X_test, y_test)
            all_results.append(results)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    # Sort by trading score
    all_results.sort(key=lambda x: x['trading_score'], reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"optimization_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print top 10 results
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(all_results[:10], 1):
        config = result['config']
        print(f"\n{i}. Trading Score: {result['trading_score']:.3f}")
        print(f"   Threshold: {config['threshold']*100:.2f}% | Horizon: {config['horizon']} candles")
        print(f"   LSTM: {config['lstm_units']} | Dropout: {config['dropout']} | LR: {config['learning_rate']}")
        print(f"   Class Weights: {config['class_weight_down']}")
        print(f"   Accuracies - DOWN: {result['class_0_accuracy']:.3f} | UP: {result['class_2_accuracy']:.3f} | NEUTRAL: {result['class_1_accuracy']:.3f}")
        print(f"   High Conf Signals - DOWN: {result['high_conf_predictions'][0]} | UP: {result['high_conf_predictions'][2]}")
    
    print(f"\n✓ Full results saved to: {results_file}")
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    results = run_optimization()
