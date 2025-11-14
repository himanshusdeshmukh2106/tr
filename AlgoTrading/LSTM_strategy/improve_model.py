"""
Improved Model Training Script
Implements key accuracy improvements from ACCURACY_IMPROVEMENT_GUIDE.md
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, 
    BatchNormalization, Attention, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/improved")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ImprovedDataPreparation:
    """Enhanced data preparation with more features"""
    
    def __init__(self, ticker="^NSEI", use_intraday=True):
        self.ticker = ticker
        self.use_intraday = use_intraday
        self.scaler = MinMaxScaler()
        
    def download_data(self, start_date="2021-01-01", end_date="2025-01-01"):
        """Download data (intraday or daily)"""
        print(f"Downloading {'5-minute' if self.use_intraday else 'daily'} data for {self.ticker}...")
        
        if self.use_intraday:
            # Download 5-minute data (much more samples!)
            df = yf.download(self.ticker, start=start_date, end=end_date, interval="5m", progress=False)
        else:
            # Daily data
            df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        
        print(f"Downloaded {len(df)} rows")
        return df
    
    def add_advanced_features(self, df):
        """Add comprehensive technical indicators"""
        print("Adding advanced features...")
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['Stochastic'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        
        # Trend Indicators
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        # Volatility Indicators
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['Keltner_Upper'] = keltner.keltner_channel_hband()
        df['Keltner_Lower'] = keltner.keltner_channel_lband()
        
        # Volume Indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price Action Features
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Body_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Support/Resistance
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Distance_from_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_from_Low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        # Statistical Features
        for window in [5, 10, 20]:
            df[f'Return_Mean_{window}'] = df['Returns'].rolling(window).mean()
            df[f'Return_Std_{window}'] = df['Returns'].rolling(window).std()
            df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window).mean()
        
        # Market Regime
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['Close']
        df['Is_Uptrend'] = (df['Close'] > df['SMA_50']).astype(int)
        
        # Time-based features (for intraday)
        if self.use_intraday:
            df['Hour'] = df.index.hour
            df['Minute'] = df.index.minute
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['Is_First_Hour'] = (df['Hour'] == 9).astype(int)
            df['Is_Last_Hour'] = (df['Hour'] == 15).astype(int)
        
        print(f"Total features: {len(df.columns)}")
        return df
    
    def create_improved_target(self, df, horizon=5, threshold=0.005):
        """
        Create multi-class target
        0 = Down (< -threshold)
        1 = Neutral (-threshold to +threshold)
        2 = Up (> +threshold)
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
        
        # Remove highly correlated features
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


class ImprovedModelBuilder:
    """Build advanced model architectures"""
    
    @staticmethod
    def build_transformer_lstm(input_shape, num_classes=3):
        """Transformer + LSTM hybrid"""
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
        attn_output = LayerNormalization()(inputs + attn_output)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(attn_output)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(64))(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape, num_classes=3):
        """CNN-LSTM hybrid"""
        inputs = Input(shape=input_shape)
        
        # CNN layers
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    @staticmethod
    def build_multi_scale_lstm(input_shape, num_classes=3):
        """Multi-scale LSTM (different lookback windows)"""
        inputs = Input(shape=input_shape)
        
        # Short-term (last 20 steps)
        lstm_short = LSTM(64, return_sequences=False)(inputs[:, -20:, :])
        
        # Medium-term (last 40 steps)
        lstm_medium = LSTM(64, return_sequences=False)(inputs[:, -40:, :])
        
        # Long-term (all steps)
        lstm_long = LSTM(64, return_sequences=False)(inputs)
        
        # Concatenate
        concat = Concatenate()([lstm_short, lstm_medium, lstm_long])
        
        # Dense layers
        x = Dense(128, activation='relu')(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model


def train_improved_model(architecture='transformer_lstm', use_intraday=True):
    """Train improved model"""
    print("=" * 80)
    print(f"Training {architecture} with {'intraday' if use_intraday else 'daily'} data")
    print("=" * 80)
    
    # Prepare data
    prep = ImprovedDataPreparation(ticker="^NSEI", use_intraday=use_intraday)
    df = prep.download_data()
    df = prep.add_advanced_features(df)
    df = prep.create_improved_target(df, horizon=5, threshold=0.005)
    
    # Create sequences
    X, y, feature_columns = prep.prepare_sequences(df, sequence_length=60)
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:")
    for i in range(3):
        count = (y == i).sum()
        print(f"  Class {i}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series!
    )
    
    # Build model
    builder = ImprovedModelBuilder()
    if architecture == 'transformer_lstm':
        model = builder.build_transformer_lstm(X_train.shape[1:], num_classes=3)
    elif architecture == 'cnn_lstm':
        model = builder.build_cnn_lstm(X_train.shape[1:], num_classes=3)
    elif architecture == 'multi_scale_lstm':
        model = builder.build_multi_scale_lstm(X_train.shape[1:], num_classes=3)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel: {architecture}")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(
            MODELS_DIR / f'{architecture}_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluation:")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confidence filtering
    confidence = np.max(y_pred, axis=1)
    high_confidence_mask = confidence > 0.6
    
    if high_confidence_mask.sum() > 0:
        filtered_acc = (y_pred_classes[high_confidence_mask] == y_test[high_confidence_mask]).mean()
        print(f"High Confidence Accuracy (>{0.6}): {filtered_acc:.4f}")
        print(f"High Confidence Samples: {high_confidence_mask.sum()} ({high_confidence_mask.sum()/len(y_test)*100:.1f}%)")
    
    # Save
    model.save(MODELS_DIR / f'{architecture}_final.h5')
    joblib.dump(prep.scaler, MODELS_DIR / f'{architecture}_scaler.pkl')
    joblib.dump(feature_columns, MODELS_DIR / f'{architecture}_features.pkl')
    
    print(f"\nModel saved to {MODELS_DIR}")
    
    return model, history, prep


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved model')
    parser.add_argument('--architecture', type=str, default='transformer_lstm',
                       choices=['transformer_lstm', 'cnn_lstm', 'multi_scale_lstm'],
                       help='Model architecture')
    parser.add_argument('--intraday', action='store_true', default=True,
                       help='Use 5-minute intraday data')
    parser.add_argument('--daily', action='store_true',
                       help='Use daily data instead of intraday')
    
    args = parser.parse_args()
    
    use_intraday = not args.daily
    
    # Train model
    model, history, prep = train_improved_model(
        architecture=args.architecture,
        use_intraday=use_intraday
    )
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Backtest the model")
    print("2. Compare with ensemble")
    print("3. Paper trade")
    print("4. Deploy to production")
