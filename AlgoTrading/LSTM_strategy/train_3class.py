"""
3-CLASS MODEL: Predict Up, Down, or Neutral
- Class 0: DOWN (< -0.15%) - SHORT signals
- Class 1: NEUTRAL (-0.15% to +0.15%) - STAY OUT
- Class 2: UP (> +0.15%) - LONG signals
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

# Configuration
import os
if os.path.exists("reliance_data_5min_full_year.csv"):
    DATA_FILE = "reliance_data_5min_full_year.csv"
elif os.path.exists("../../reliance_data_5min_full_year.csv"):
    DATA_FILE = "../../reliance_data_5min_full_year.csv"
else:
    DATA_FILE = "reliance_data_5min_full_year.csv"

MODELS_DIR = Path("models/reliance_3class")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ThreeClassDataPrep:
    """Data preparation for 3-class classification"""
    
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
        """Add focused features"""
        print("Adding features...")
        
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
    
    def create_3class_target(self, df, horizon=5, threshold=0.003):
        """
        Create 3-class target:
        0 = DOWN (< -threshold)
        1 = NEUTRAL (-threshold to +threshold)
        2 = UP (> +threshold)
        
        threshold=0.003 (0.3%) - bigger moves, easier to predict
        """
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
        df['Target'] = 1  # Default to neutral
        df.loc[df['Future_Return'] < -threshold, 'Target'] = 0  # Down
        df.loc[df['Future_Return'] > threshold, 'Target'] = 2   # Up
        
        return df
    
    def prepare_sequences(self, df, sequence_length=30):
        """Prepare sequences"""
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


def build_3class_lstm(input_shape):
    """Build LSTM for 3-class classification"""
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(3, activation='softmax')(x)  # 3 classes
    
    model = Model(inputs, outputs)
    return model


def train_3class_model():
    """Train 3-class model"""
    print("="*80)
    print("3-CLASS MODEL TRAINING")
    print("="*80)
    print("\n📊 Classes (0.3% threshold - bigger moves):")
    print("  Class 0: DOWN (< -0.3%) - SHORT these")
    print("  Class 1: NEUTRAL (-0.3% to +0.3%) - STAY OUT")
    print("  Class 2: UP (> +0.3%) - LONG these")
    print()
    
    # Load data
    prep = ThreeClassDataPrep()
    df = prep.load_data(DATA_FILE)
    df = prep.add_features(df)
    df = prep.create_3class_target(df, horizon=5, threshold=0.003)
    
    # Create sequences
    X, y, features = prep.prepare_sequences(df, sequence_length=30)
    
    print(f"\nData: X={X.shape}, y={y.shape}")
    print(f"\nClass distribution BEFORE balancing:")
    for class_val in [0, 1, 2]:
        count = (y == class_val).sum()
        pct = count / len(y) * 100
        class_name = ["DOWN", "NEUTRAL", "UP"][class_val]
        print(f"  Class {class_val} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # SMOTE for balancing
    print("\n🔄 Applying SMOTE...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    print(f"\nClass distribution AFTER SMOTE:")
    for class_val in [0, 1, 2]:
        count = (y_train_balanced == class_val).sum()
        pct = count / len(y_train_balanced) * 100
        class_name = ["DOWN", "NEUTRAL", "UP"][class_val]
        print(f"  Class {class_val} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_balanced, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Build model
    model = build_3class_lstm(X_train.shape[1:])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Class weights (slightly favor trading classes over neutral)
    class_weight = {0: 1.5, 1: 1.0, 2: 1.5}
    print(f"\n⚡ Class weights: {class_weight}")
    print("  (Slightly favor DOWN and UP over NEUTRAL)")
    
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
        X_train_balanced, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Evaluate
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✓ Overall Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Per-class accuracy
    print(f"\n{'='*80}")
    print("PER-CLASS ACCURACY")
    print(f"{'='*80}")
    
    class_names = ["DOWN (Short)", "NEUTRAL (Stay Out)", "UP (Long)"]
    for class_val in [0, 1, 2]:
        mask = y_test == class_val
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"\n✓ Class {class_val} - {class_names[class_val]}")
            print(f"  Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")
            print(f"  Samples: {mask.sum():,}")
    
    # Confidence-based accuracy
    confidence = np.max(y_pred_probs, axis=1)
    
    print(f"\n{'='*80}")
    print("CONFIDENCE-BASED ACCURACY")
    print(f"{'='*80}")
    
    for threshold in [0.6, 0.7, 0.8]:
        high_conf = confidence > threshold
        if high_conf.sum() > 0:
            acc = (y_pred[high_conf] == y_test[high_conf]).mean()
            coverage = high_conf.sum() / len(y_test)
            print(f"\n✓ Confidence > {threshold}: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  Coverage: {high_conf.sum():,} ({coverage*100:.1f}%)")
            
            # Show distribution at this confidence
            for class_val in [0, 1, 2]:
                class_count = (y_pred[high_conf] == class_val).sum()
                if class_count > 0:
                    print(f"    {class_names[class_val]}: {class_count}")
    
    # Save
    model.save(MODELS_DIR / 'model.h5')
    joblib.dump(prep.scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(features, MODELS_DIR / 'features.pkl')
    
    print(f"\n✓ Model saved to: {MODELS_DIR}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE! 🎉")
    print("="*80)
    print("\n💡 How to use:")
    print("  Class 0 (DOWN) → SHORT the stock")
    print("  Class 1 (NEUTRAL) → Stay out, no trade")
    print("  Class 2 (UP) → LONG the stock")
    
    return model, history, prep


if __name__ == "__main__":
    model, history, prep = train_3class_model()
