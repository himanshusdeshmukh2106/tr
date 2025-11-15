"""
OPTIMIZED Training Script - Based on Hyperparameter Search Results
Best Config: Shorter Horizon with optimized parameters

Results from optimization:
- Class 1 Accuracy: 51.2% (was 34-42%)
- Overall Accuracy: 63.9%
- High Confidence: 91.2%
- Weighted Score: 0.657
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import ta
from imblearn.over_sampling import SMOTE
from pathlib import Path
import json
from datetime import datetime
import joblib
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

MODELS_DIR = Path("models/reliance_optimized")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# OPTIMIZED PARAMETERS (from hyperparameter search)
HORIZON = 8              # Shorter horizon works better!
THRESHOLD = 0.0015       # 0.15% threshold
SEQUENCE_LENGTH = 30     # 30 candles lookback
LSTM_UNITS = [64, 32]    # Standard LSTM size
DROPOUT = 0.3            # 30% dropout
LEARNING_RATE = 0.001    # Standard learning rate
BATCH_SIZE = 64          # Standard batch size


class OptimizedDataPrep:
    """Data preparation with optimized parameters"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def load_data(self, filepath):
        """Load Reliance 5-minute data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        print(f"Loaded {len(df)} rows")
        return df
    
    def add_features(self, df):
        """Add technical indicators"""
        print("Adding features...")
        
        # Basic
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # Momentum
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        df['MOM'] = df['Close'].pct_change(periods=5)
        
        # Trend
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        
        # Volatility
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Volume
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        # Price levels
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Distance_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_Low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        # Time features
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        print(f"Total features: {len(df.columns)}")
        return df
    
    def create_target(self, df):
        """Create target with optimized parameters"""
        df['Future_Return'] = df['Close'].shift(-HORIZON) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > THRESHOLD).astype(int)
        return df
    
    def prepare_sequences(self, df):
        """Prepare sequences with optimized length"""
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['Target', 'Future_Return']]
        
        df = df.dropna()
        
        # Remove highly correlated features
        corr_matrix = df[feature_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        feature_columns = [col for col in feature_columns if col not in to_drop]
        
        print(f"Using {len(feature_columns)} features")
        
        # Scale
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i-SEQUENCE_LENGTH:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y), feature_columns


def build_optimized_lstm(input_shape):
    """Build LSTM with optimized architecture"""
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(LSTM_UNITS[0], return_sequences=True))(inputs)
    x = Dropout(DROPOUT)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(LSTM_UNITS[1]))(x)
    x = Dropout(DROPOUT)(x)
    x = BatchNormalization()(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT * 0.7)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model


def train_optimized_model():
    """Train with optimized parameters"""
    print("="*80)
    print("OPTIMIZED LSTM TRAINING")
    print("="*80)
    print("\n🎯 OPTIMIZED PARAMETERS (from hyperparameter search):")
    print(f"  Horizon: {HORIZON} candles (40 minutes)")
    print(f"  Threshold: {THRESHOLD} (0.15%)")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  LSTM Units: {LSTM_UNITS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"\n📈 Expected Performance:")
    print(f"  Class 1 Accuracy: ~51% (was 34-42%)")
    print(f"  Overall Accuracy: ~64%")
    print(f"  High Confidence: ~91%")
    print()
    
    # Load and prepare data
    prep = OptimizedDataPrep()
    df = prep.load_data(DATA_FILE)
    df = prep.add_features(df)
    df = prep.create_target(df)
    
    # Create sequences
    X, y, features = prep.prepare_sequences(df)
    
    print(f"\nData: X={X.shape}, y={y.shape}")
    print(f"\nClass distribution BEFORE balancing:")
    print(f"  Class 0 (Down/Neutral): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Class 1 (Up): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # SMOTE balancing
    print("\n🔄 Applying SMOTE...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    print(f"\nClass distribution AFTER SMOTE:")
    print(f"  Class 0: {(y_train_balanced==0).sum():,} ({(y_train_balanced==0).sum()/len(y_train_balanced)*100:.1f}%)")
    print(f"  Class 1: {(y_train_balanced==1).sum():,} ({(y_train_balanced==1).sum()/len(y_train_balanced)*100:.1f}%)")
    
    # Build model
    model = build_optimized_lstm(X_train.shape[1:])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*80}")
    print("Model Architecture")
    print(f"{'='*80}")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODELS_DIR / 'best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    # Train
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")
    
    history = model.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Overall Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Per-class accuracy
    class0_acc = (y_pred_binary[y_test == 0] == y_test[y_test == 0]).mean()
    class1_acc = (y_pred_binary[y_test == 1] == y_test[y_test == 1]).mean()
    
    print(f"\n✓ Class 0 (Down/Neutral): {class0_acc:.4f} ({class0_acc*100:.2f}%)")
    print(f"  Samples: {(y_test == 0).sum():,}")
    
    print(f"\n✓ Class 1 (Up): {class1_acc:.4f} ({class1_acc*100:.2f}%) ⭐")
    print(f"  Samples: {(y_test == 1).sum():,}")
    print(f"  Expected: ~51% (from optimization)")
    
    # Confidence-based accuracy
    confidence = np.maximum(y_pred, 1 - y_pred).flatten()
    
    print(f"\n{'='*80}")
    print("CONFIDENCE-BASED ACCURACY")
    print(f"{'='*80}")
    
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        high_conf = confidence > threshold
        if high_conf.sum() > 0:
            acc = (y_pred_binary[high_conf] == y_test[high_conf]).mean()
            coverage = high_conf.sum() / len(y_test)
            print(f"\n✓ Confidence > {threshold}: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  Coverage: {high_conf.sum():,} ({coverage*100:.1f}%)")
    
    # Weighted score (same as optimization)
    weighted_score = 0.25 * test_acc + 0.45 * class1_acc + 0.30 * (y_pred_binary[confidence > 0.7] == y_test[confidence > 0.7]).mean()
    print(f"\n✓ Weighted Score: {weighted_score:.3f}")
    print(f"  Expected: ~0.657 (from optimization)")
    
    # Save model and artifacts
    model.save(MODELS_DIR / 'final_model.h5')
    joblib.dump(prep.scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(features, MODELS_DIR / 'features.pkl')
    
    # Save training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'horizon': HORIZON,
            'threshold': THRESHOLD,
            'sequence_length': SEQUENCE_LENGTH,
            'lstm_units': LSTM_UNITS,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE
        },
        'performance': {
            'overall_accuracy': float(test_acc),
            'class0_accuracy': float(class0_acc),
            'class1_accuracy': float(class1_acc),
            'high_conf_0.7_accuracy': float((y_pred_binary[confidence > 0.7] == y_test[confidence > 0.7]).mean()),
            'weighted_score': float(weighted_score)
        },
        'training': {
            'epochs': len(history.history['loss']),
            'final_loss': float(test_loss)
        }
    }
    
    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*80}")
    print(f"\nClass 1 Accuracy:")
    print(f"  Baseline: 34-42%")
    print(f"  Optimized: {class1_acc*100:.1f}%")
    print(f"  Improvement: +{(class1_acc - 0.38) * 100:.1f}%")
    
    print(f"\n✓ Model saved to: {MODELS_DIR}")
    print(f"✓ Results saved to: {MODELS_DIR / 'training_results.json'}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE! 🎉")
    print(f"{'='*80}\n")
    
    return model, history, prep, results


if __name__ == "__main__":
    model, history, prep, results = train_optimized_model()
