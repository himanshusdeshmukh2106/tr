"""
Train Improved LSTM Model on Reliance 5-Minute Data
Implements all accuracy improvements from the guide
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, 
    BatchNormalization, Attention, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, Concatenate, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
# Auto-detect data file location (works on both local and Colab)
import os
if os.path.exists("reliance_data_5min_full_year.csv"):
    DATA_FILE = "reliance_data_5min_full_year.csv"
elif os.path.exists("../../reliance_data_5min_full_year.csv"):
    DATA_FILE = "../../reliance_data_5min_full_year.csv"
elif os.path.exists(r"C:\Users\Lenovo\Desktop\tr\reliance_data_5min_full_year.csv"):
    DATA_FILE = r"C:\Users\Lenovo\Desktop\tr\reliance_data_5min_full_year.csv"
else:
    DATA_FILE = "reliance_data_5min_full_year.csv"  # Default, will error if not found

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/reliance_improved")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class RelianceDataPreparation:
    """Enhanced data preparation for Reliance 5-min data"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        """Load Reliance 5-minute data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Rename columns to standard format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        return df
    
    def add_all_features(self, df):
        """Add comprehensive technical indicators"""
        print("Adding technical indicators...")
        
        # ===== BASIC FEATURES =====
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # ===== MOVING AVERAGES =====
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # ===== MOMENTUM INDICATORS =====
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['Stochastic'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        
        # ===== TREND INDICATORS =====
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        # ===== VOLATILITY INDICATORS =====
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
        
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['Keltner_Upper'] = keltner.keltner_channel_hband()
        df['Keltner_Lower'] = keltner.keltner_channel_lband()
        
        # ===== VOLUME INDICATORS =====
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        # ===== PRICE ACTION FEATURES =====
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Body_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # ===== PRICE LEVELS (NEW!) =====
        df['Distance_from_High_20'] = (df['High'].rolling(20).max() - df['Close']) / df['Close']
        df['Distance_from_Low_20'] = (df['Close'] - df['Low'].rolling(20).min()) / df['Close']
        
        # ===== SUPPORT/RESISTANCE (NEW!) =====
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['Distance_from_Pivot'] = (df['Close'] - df['Pivot']) / df['Pivot']
        df['Distance_from_R1'] = (df['R1'] - df['Close']) / df['Close']
        df['Distance_from_S1'] = (df['Close'] - df['S1']) / df['Close']
        
        # ===== STATISTICAL FEATURES =====
        for window in [5, 10, 20]:
            df[f'Return_Mean_{window}'] = df['Returns'].rolling(window).mean()
            df[f'Return_Std_{window}'] = df['Returns'].rolling(window).std()
            df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'Volume_Std_{window}'] = df['Volume'].rolling(window).std()
        
        # ===== MARKET REGIME =====
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['Close']
        df['Is_Uptrend'] = (df['Close'] > df['SMA_50']).astype(int)
        df['Is_Downtrend'] = (df['Close'] < df['SMA_50']).astype(int)
        
        # ===== TIME-BASED FEATURES =====
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Minute_Sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
        df['Minute_Cos'] = np.cos(2 * np.pi * df['Minute'] / 60)
        df['Is_First_Hour'] = (df['Hour'] == 9).astype(int)
        df['Is_Last_Hour'] = (df['Hour'] == 15).astype(int)
        df['Is_Market_Open'] = ((df['Hour'] >= 9) & (df['Hour'] < 15)).astype(int)
        
        print(f"Total features: {len(df.columns)}")
        return df
    
    def create_target(self, df, horizon=5, threshold=0.002):
        """
        Create multi-class target
        0 = Down (< -threshold)
        1 = Neutral (-threshold to +threshold)
        2 = Up (> +threshold)
        
        Note: Threshold lowered to 0.002 (0.2%) to get more balanced classes
        """
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
        df['Target'] = 1  # Neutral
        df.loc[df['Future_Return'] > threshold, 'Target'] = 2  # Up
        df.loc[df['Future_Return'] < -threshold, 'Target'] = 0  # Down
        
        return df
    
    def prepare_sequences(self, df, sequence_length=60):
        """Prepare sequences for LSTM"""
        # Select numeric features
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['Target', 'Future_Return']]
        
        # Drop NaN
        df = df.dropna()
        
        print(f"Data after dropping NaN: {len(df)} rows")
        
        # Remove highly correlated features
        print("Removing highly correlated features...")
        corr_matrix = df[feature_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        feature_columns = [col for col in feature_columns if col not in to_drop]
        
        print(f"Using {len(feature_columns)} features after correlation filtering")
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y), feature_columns


def build_improved_lstm(input_shape, num_classes=3):
    """
    Improved LSTM Architecture with Multi-Scale Processing
    """
    inputs = Input(shape=input_shape)
    
    # ===== MULTI-SCALE LSTM =====
    # Short-term patterns (last 20 steps)
    lstm_short = Bidirectional(LSTM(64, return_sequences=False))(inputs[:, -20:, :])
    
    # Medium-term patterns (last 40 steps)
    lstm_medium = Bidirectional(LSTM(64, return_sequences=False))(inputs[:, -40:, :])
    
    # Long-term patterns (all steps)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # Attention mechanism
    attention = Attention()([x, x])
    x = GlobalAveragePooling1D()(attention)
    
    # ===== CONCATENATE MULTI-SCALE FEATURES =====
    concat = Concatenate()([lstm_short, lstm_medium, x])
    
    # ===== DENSE LAYERS =====
    x = Dense(256, activation='relu')(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # ===== OUTPUT =====
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


def build_transformer_lstm(input_shape, num_classes=3):
    """
    Transformer + LSTM Hybrid Architecture
    """
    inputs = Input(shape=input_shape)
    
    # ===== TRANSFORMER BLOCK =====
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    attn_output = Dropout(0.2)(attn_output)
    x = LayerNormalization()(Add()([inputs, attn_output]))
    
    # Feed-forward network
    ff_output = Dense(128, activation='relu')(x)
    ff_output = Dense(input_shape[-1])(ff_output)
    ff_output = Dropout(0.2)(ff_output)
    x = LayerNormalization()(Add()([x, ff_output]))
    
    # ===== LSTM LAYERS =====
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # ===== DENSE LAYERS =====
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # ===== OUTPUT =====
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


def train_model(architecture='improved_lstm'):
    """Train the model"""
    print("=" * 80)
    print(f"Training {architecture} on Reliance 5-minute data")
    print("=" * 80)
    
    # ===== LOAD AND PREPARE DATA =====
    prep = RelianceDataPreparation()
    df = prep.load_data(DATA_FILE)
    df = prep.add_all_features(df)
    df = prep.create_target(df, horizon=5, threshold=0.005)
    
    # Create sequences
    X, y, feature_columns = prep.prepare_sequences(df, sequence_length=60)
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"\nClass distribution:")
    for i in range(3):
        count = (y == i).sum()
        print(f"  Class {i}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # ===== SPLIT DATA =====
    # Use time-based split (don't shuffle!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    # ===== CALCULATE CLASS WEIGHTS (FIX IMBALANCE) =====
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\n⚠️  Class weights (to handle imbalance):")
    for i, weight in class_weight_dict.items():
        class_name = ['Down', 'Neutral', 'Up'][i]
        print(f"  Class {i} ({class_name}): {weight:.2f}x penalty")
    
    # ===== BUILD MODEL =====
    if architecture == 'improved_lstm':
        model = build_improved_lstm(X_train.shape[1:], num_classes=3)
    elif architecture == 'transformer_lstm':
        model = build_transformer_lstm(X_train.shape[1:], num_classes=3)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # ===== COMPILE =====
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*80}")
    print(f"Model: {architecture}")
    print(f"{'='*80}")
    model.summary()
    
    # ===== CALLBACKS =====
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODELS_DIR / f'{architecture}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # ===== TRAIN =====
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Fix class imbalance!
        verbose=1
    )
    
    # ===== EVALUATE =====
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"✓ Test Loss: {test_loss:.4f}")
    
    # ===== PREDICTIONS WITH CONFIDENCE =====
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confidence = np.max(y_pred, axis=1)
    
    # Overall accuracy
    overall_acc = (y_pred_classes == y_test).mean()
    print(f"\n✓ Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # Confidence-based accuracy
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        high_conf_mask = confidence > threshold
        if high_conf_mask.sum() > 0:
            high_conf_acc = (y_pred_classes[high_conf_mask] == y_test[high_conf_mask]).mean()
            coverage = high_conf_mask.sum() / len(y_test) * 100
            print(f"\n✓ Accuracy (confidence > {threshold}): {high_conf_acc:.4f} ({high_conf_acc*100:.2f}%)")
            print(f"  Coverage: {high_conf_mask.sum():,} samples ({coverage:.1f}%)")
    
    # Per-class accuracy
    print(f"\n{'='*80}")
    print("Per-Class Performance")
    print(f"{'='*80}")
    for i in range(3):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred_classes[class_mask] == y_test[class_mask]).mean()
            class_name = ['Down', 'Neutral', 'Up'][i]
            print(f"\n✓ Class {i} ({class_name}): {class_acc:.4f} ({class_acc*100:.2f}%)")
            print(f"  Samples: {class_mask.sum():,}")
    
    # ===== SAVE =====
    print(f"\n{'='*80}")
    print("Saving Model")
    print(f"{'='*80}")
    
    model.save(MODELS_DIR / f'{architecture}_final.h5')
    joblib.dump(prep.scaler, MODELS_DIR / f'{architecture}_scaler.pkl')
    joblib.dump(feature_columns, MODELS_DIR / f'{architecture}_features.pkl')
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODELS_DIR / f'{architecture}_history.csv', index=False)
    
    print(f"\n✓ Model saved to: {MODELS_DIR / f'{architecture}_final.h5'}")
    print(f"✓ Scaler saved to: {MODELS_DIR / f'{architecture}_scaler.pkl'}")
    print(f"✓ Features saved to: {MODELS_DIR / f'{architecture}_features.pkl'}")
    print(f"✓ History saved to: {MODELS_DIR / f'{architecture}_history.csv'}")
    
    print(f"\n{'='*80}")
    print("Training Complete! 🎉")
    print(f"{'='*80}")
    
    return model, history, prep


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved model on Reliance data')
    parser.add_argument('--architecture', type=str, default='improved_lstm',
                       choices=['improved_lstm', 'transformer_lstm'],
                       help='Model architecture')
    
    args = parser.parse_args()
    
    # Train model
    model, history, prep = train_model(architecture=args.architecture)
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Check training history: models/reliance_improved/*_history.csv")
    print("2. Backtest the model on historical data")
    print("3. Paper trade with real-time data")
    print("4. Deploy to production")
    print("\nGood luck! 📈")
