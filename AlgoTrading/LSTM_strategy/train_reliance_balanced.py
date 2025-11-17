"""
AGGRESSIVE BALANCING VERSION: SMOTE + Heavy Class Weights
- Uses SMOTE to oversample minority class
- Uses class_weight={0: 1.0, 1: 5.0} to force model to focus on Class 1
- Goal: Improve Class 1 (Up) accuracy from 38% to 50%+
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, 
    BatchNormalization, Attention, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import SMOTE for oversampling
from imblearn.over_sampling import SMOTE

# Configuration
import os
if os.path.exists("reliance_data_5min_full_year.csv"):
    DATA_FILE = "reliance_data_5min_full_year.csv"
elif os.path.exists("../../reliance_data_5min_full_year.csv"):
    DATA_FILE = "../../reliance_data_5min_full_year.csv"
else:
    DATA_FILE = "reliance_data_5min_full_year.csv"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/reliance_balanced")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class BalancedDataPreparation:
    """Data preparation with proper class balancing"""
    
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
        """MINIMAL features - only what matters for price prediction"""
        print("Adding MINIMAL features (less is more!)...")
        
        # Candle characteristics (most important!)
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']  # Candle body size
        df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        
        # Price action
        df['Returns'] = df['Close'].pct_change()
        df['Returns_2'] = df['Close'].pct_change(2)
        df['Returns_5'] = df['Close'].pct_change(5)
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages (trend)
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['Price_to_SMA5'] = df['Close'] / df['SMA_5']
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_EMA21'] = df['Close'] / df['EMA_21']
        
        # Trend strength (ADX)
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Volume (momentum confirmation)
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        # Price levels (support/resistance)
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Distance_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_Low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        # Time of day (intraday patterns)
        df['Hour'] = df.index.hour
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        print(f"✓ Using {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")
        print("  Added: Candle body/wicks, EMA 21, ADX")
        print("  Keeping: Price action, MAs, volume, time")
        return df
    
    def create_binary_target(self, df, horizon=10, threshold=0.0015):
        """
        Binary classification: 0=Down/Neutral, 1=Up
        
        OPTIMIZED PARAMETERS (v2):
        - horizon=10 (50 minutes) - easier to predict than 5
        - threshold=0.0015 (0.15%) - BALANCED (not too high, not too low)
        
        Why 0.15%?
        - 0.1% = too many signals (noisy)
        - 0.2% = too few signals (can't learn)
        - 0.15% = just right! (Goldilocks zone)
        """
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > threshold).astype(int)
        return df
    
    def prepare_sequences(self, df, sequence_length=30):
        """Prepare sequences (shorter for faster training)"""
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['Target', 'Future_Return']]
        
        df = df.dropna()
        
        # Remove correlated features
        corr_matrix = df[feature_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        feature_columns = [col for col in feature_columns if col not in to_drop]
        
        print(f"Using {len(feature_columns)} features")
        
        # Scale
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y), feature_columns


def build_simple_lstm(input_shape):
    """Simpler, faster LSTM for binary classification"""
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)  # Binary output
    
    model = Model(inputs, outputs)
    return model


def train_balanced_model():
    """Train with proper balancing"""
    print("="*80)
    print("Training BALANCED Binary Classifier")
    print("="*80)
    
    # Load data
    prep = BalancedDataPreparation()
    df = prep.load_data(DATA_FILE)
    df = prep.add_features(df)
    df = prep.create_binary_target(df, horizon=5, threshold=0.001)
    
    # Create sequences
    X, y, features = prep.prepare_sequences(df, sequence_length=30)
    
    print(f"\n⚠️  AGGRESSIVE BALANCING STRATEGY:")
    print(f"  1. SMOTE: Oversample Class 1 to match Class 0")
    print(f"  2. Class Weights: {0: 1.0, 1: 5.0} - 5x penalty for Class 1 errors")
    print(f"  3. Goal: Class 1 accuracy > 50% (currently 38%)")
    print(f"\n  Horizon: 5 candles (25 minutes)")
    print(f"  Threshold: 0.1%")
    
    print(f"\nData: X={X.shape}, y={y.shape}")
    print(f"Class distribution BEFORE balancing:")
    print(f"  Class 0 (Down/Neutral): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Class 1 (Up): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # SMOTE: Oversample minority class
    print("\n🔄 Applying SMOTE to balance training data...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    print(f"\nClass distribution AFTER SMOTE:")
    print(f"  Class 0: {(y_train_balanced==0).sum():,} ({(y_train_balanced==0).sum()/len(y_train_balanced)*100:.1f}%)")
    print(f"  Class 1: {(y_train_balanced==1).sum():,} ({(y_train_balanced==1).sum()/len(y_train_balanced)*100:.1f}%)")
    
    # Build model
    model = build_simple_lstm(X_train.shape[1:])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # AGGRESSIVE class weights - force model to focus on Class 1
    class_weight = {0: 1.0, 1: 5.0}
    print(f"\n⚡ Using AGGRESSIVE class weights: {class_weight}")
    print(f"  This will force the model to focus 5x more on Class 1 (Up) predictions!")
    
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
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weight,  # Apply aggressive class weights
        verbose=1
    )
    
    # Evaluate
    print(f"\n{'='*80}")
    print("Evaluation")
    print(f"{'='*80}")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Per-class accuracy
    for class_val in [0, 1]:
        mask = y_test == class_val
        if mask.sum() > 0:
            class_acc = (y_pred_binary[mask] == y_test[mask]).mean()
            class_name = "Down/Neutral" if class_val == 0 else "Up"
            print(f"\n✓ Class {class_val} ({class_name}): {class_acc:.4f} ({class_acc*100:.2f}%)")
            print(f"  Samples: {mask.sum():,}")
    
    # Confidence-based accuracy
    confidence = np.maximum(y_pred, 1 - y_pred).flatten()
    for threshold in [0.6, 0.7, 0.8]:
        high_conf = confidence > threshold
        if high_conf.sum() > 0:
            acc = (y_pred_binary[high_conf] == y_test[high_conf]).mean()
            print(f"\n✓ Accuracy (confidence > {threshold}): {acc:.4f} ({acc*100:.2f}%)")
            print(f"  Coverage: {high_conf.sum():,} ({high_conf.sum()/len(y_test)*100:.1f}%)")
    
    # Save
    model.save(MODELS_DIR / 'final_model.h5')
    joblib.dump(prep.scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(features, MODELS_DIR / 'features.pkl')
    
    print(f"\n✓ Model saved to: {MODELS_DIR}")
    print("\n" + "="*80)
    print("Training Complete! 🎉")
    print("="*80)
    
    return model, history, prep


if __name__ == "__main__":
    model, history, prep = train_balanced_model()
