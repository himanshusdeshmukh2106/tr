"""
TCN (Temporal Convolutional Network) with Focal Loss
Based on latest research for imbalanced time series classification

Key improvements:
1. TCN architecture with dilated causal convolutions
2. Focal Loss to handle class imbalance
3. Residual connections for better gradient flow
4. Faster training than LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

MODELS_DIR = Path("models/reliance_tcn_focal")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for handling class imbalance
    FL = -α(1-pt)^γ * log(pt)
    
    Args:
        alpha: Weighting factor for classes [class0, class1, class2]
        gamma: Focusing parameter (1-5, higher = more focus on hard examples)
    """
    def __init__(self, alpha=[0.25, 0.25, 0.5], gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        # (1 - pt)^gamma term focuses on hard examples
        weight = tf.pow(1 - y_pred, self.gamma)
        
        # Apply alpha weighting
        alpha_weight = y_true * self.alpha
        
        # Combine
        focal_loss = alpha_weight * weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


# ============================================================================
# TCN ARCHITECTURE
# ============================================================================

def residual_block(x, dilation_rate, nb_filters, kernel_size, padding='causal', dropout_rate=0.2):
    """
    TCN Residual Block with dilated causal convolutions
    """
    # First conv layer
    conv1 = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        activation='relu'
    )(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(dropout_rate)(conv1)
    
    # Second conv layer
    conv2 = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        activation='relu'
    )(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(dropout_rate)(conv2)
    
    # Residual connection
    # Match dimensions if needed
    if x.shape[-1] != nb_filters:
        x = layers.Conv1D(nb_filters, 1, padding='same')(x)
    
    # Add residual
    out = layers.Add()([x, conv2])
    out = layers.Activation('relu')(out)
    
    return out


def build_tcn_model(input_shape, num_classes=3):
    """
    Build TCN model with multiple dilated residual blocks
    
    Architecture:
    Input → TCN Block (d=1) → TCN Block (d=2) → TCN Block (d=4) 
    → TCN Block (d=8) → Global Pool → Dense → Output
    """
    inputs = layers.Input(shape=input_shape)
    
    # TCN blocks with increasing dilation
    x = residual_block(inputs, dilation_rate=1, nb_filters=64, kernel_size=3, dropout_rate=0.2)
    x = residual_block(x, dilation_rate=2, nb_filters=64, kernel_size=3, dropout_rate=0.2)
    x = residual_block(x, dilation_rate=4, nb_filters=64, kernel_size=3, dropout_rate=0.2)
    x = residual_block(x, dilation_rate=8, nb_filters=32, kernel_size=3, dropout_rate=0.2)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


# ============================================================================
# DATA PREPARATION
# ============================================================================

class DataPrep:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        print(f"Loaded {len(df)} rows (5-minute data)")
        
        # Convert to 15-minute candles
        print("Converting to 15-minute timeframe...")
        df_15min = df.resample('15T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        print(f"After resampling: {len(df_15min)} rows (15-minute data)")
        print(f"Noise reduction: {len(df) / len(df_15min):.1f}x fewer candles")
        
        return df_15min
    
    def add_features(self, df):
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
    
    def create_target(self, df, horizon=3, threshold=0.003):
        """
        Create 3-class target for 15-minute data
        horizon=3 means 45 minutes ahead (3 x 15min candles)
        threshold=0.003 means 0.3% move
        """
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = 1  # Neutral
        df.loc[df['Future_Return'] < -threshold, 'Target'] = 0  # Down
        df.loc[df['Future_Return'] > threshold, 'Target'] = 2   # Up
        return df
    
    def prepare_sequences(self, df, sequence_length=20):
        """
        sequence_length=20 for 15-min data = 5 hours of history
        (same as 30 candles of 5-min data = 2.5 hours, but with less noise)
        """
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
        
        return np.array(X), np.array(y)


# ============================================================================
# TRAINING
# ============================================================================

def train_tcn_focal():
    print("="*80)
    print("TCN WITH FOCAL LOSS - 15-MINUTE TIMEFRAME")
    print("="*80)
    print("\n🔬 Key Improvements:")
    print("  1. 15-minute candles: 3x less noise than 5-minute")
    print("  2. TCN: Faster convergence, better long-term dependencies")
    print("  3. Focal Loss: Focuses on hard-to-classify examples")
    print("  4. Dilated convolutions: Captures patterns at multiple scales")
    print("  5. Residual connections: Better gradient flow")
    print("\n📊 Prediction Target:")
    print("  Horizon: 3 candles (45 minutes)")
    print("  Threshold: 0.3% move")
    print("  Lookback: 20 candles (5 hours)")
    print()
    
    # Load data
    prep = DataPrep()
    df = prep.load_data(DATA_FILE)  # Automatically converts to 15-min
    df = prep.add_features(df)
    df = prep.create_target(df, horizon=3, threshold=0.003)  # 45 min, 0.3%
    
    # Create sequences
    X, y = prep.prepare_sequences(df, sequence_length=20)  # 5 hours lookback
    
    print(f"\nData: X={X.shape}, y={y.shape}")
    print(f"\nClass distribution:")
    for class_val in [0, 1, 2]:
        count = (y == class_val).sum()
        pct = count / len(y) * 100
        class_name = ["DOWN", "NEUTRAL", "UP"][class_val]
        print(f"  Class {class_val} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # SMOTE
    print("\n🔄 Applying SMOTE...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    print(f"\nAfter SMOTE:")
    for class_val in [0, 1, 2]:
        count = (y_train_balanced == class_val).sum()
        print(f"  Class {class_val}: {count:,}")
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_balanced, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Build TCN model
    print("\n🏗️  Building TCN model...")
    model = build_tcn_model(X_train.shape[1:], num_classes=3)
    
    # Compile with Focal Loss
    # Alpha: [DOWN, NEUTRAL, UP] - give more weight to trading classes
    focal_loss = FocalLoss(alpha=[0.35, 0.3, 0.35], gamma=2.0)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODELS_DIR / 'best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
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
            
            # Show distribution
            for class_val in [0, 1, 2]:
                class_count = (y_pred[high_conf] == class_val).sum()
                if class_count > 0:
                    print(f"    {class_names[class_val]}: {class_count}")
    
    # Save
    model.save(MODELS_DIR / 'model.h5')
    import joblib
    joblib.dump(prep.scaler, MODELS_DIR / 'scaler.pkl')
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'architecture': 'TCN with Focal Loss',
        'focal_loss_params': {'alpha': [0.35, 0.3, 0.35], 'gamma': 2.0},
        'overall_accuracy': float(test_acc),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Model saved to: {MODELS_DIR}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE! 🎉")
    print("="*80)
    print("\n💡 15-min TCN + Focal Loss should provide:")
    print("  - Better UP/DOWN accuracy (less noise, clearer patterns)")
    print("  - More reliable signals (3x noise reduction)")
    print("  - Still intraday (25 candles per day)")
    print("  - Focuses on hard examples (Focal Loss)")
    print("\n📈 Expected improvement:")
    print("  - UP/DOWN accuracy should be >50% (vs 0-20% on 5-min)")
    print("  - More balanced predictions (not just NEUTRAL)")
    print("  - Tradeable signals with good risk/reward")
    
    return model, history, prep


if __name__ == "__main__":
    model, history, prep = train_tcn_focal()
