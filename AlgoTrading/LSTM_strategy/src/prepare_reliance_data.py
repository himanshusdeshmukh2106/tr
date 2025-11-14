"""
Prepare RELIANCE NSE data for LSTM training (Daily timeframe)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import DATA_DIR, LSTM_CONFIG
from src.data_preparation import DataPreparation


def clean_nse_data(filepath):
    """Clean and format NSE data"""
    print(f"Loading data from {filepath}...")
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    # Rename columns to standard format
    column_mapping = {
        'Date': 'Date',
        'OPEN': 'Open',
        'HIGH': 'High',
        'LOW': 'Low',
        'close': 'Close',
        'VOLUME': 'Volume',
        'PREV. CLOSE': 'Prev_Close'
    }
    df = df.rename(columns=column_mapping)
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.set_index('Date')
    
    # Remove commas and convert to numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Sort by date (oldest first)
    df = df.sort_index()
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def add_technical_indicators(df):
    """Add technical indicators for daily data"""
    print("Adding technical indicators...")
    
    prep = DataPreparation()
    
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # ADX
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr_smooth = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_smooth)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Price position
    df['Price_vs_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
    df['Price_vs_EMA21'] = ((df['Close'] - df['EMA_21']) / df['EMA_21']) * 100
    
    return df


def create_target(df, horizon=1, threshold=0.0):
    """Create target variable (1 = price up, 0 = price down)"""
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > threshold).astype(int)
    return df


def prepare_sequences(df, sequence_length=60):
    """Prepare sequences for LSTM"""
    print(f"Creating sequences with length {sequence_length}...")
    
    # Select features
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'EMA_21',
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
        'ATR', 'ADX',
        'Returns', 'Log_Returns', 'Volume_Change', 'Volatility',
        'Price_vs_SMA20', 'Price_vs_EMA21'
    ]
    
    # Drop NaN
    df_clean = df.dropna()
    print(f"After removing NaN: {len(df_clean)} rows")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_clean[feature_cols])
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df_clean['Target'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(feature_cols)}")
    
    return X, y, scaler, feature_cols


def main():
    """Main data preparation pipeline"""
    # File path
    data_file = Path("C:/Users/Lenovo/Desktop/tr/AlgoTrading/Quote-Equity-RELIANCE-EQ-12-11-2024-to-12-11-2025.csv")
    
    # Load and clean data
    df = clean_nse_data(data_file)
    
    # Add indicators
    df = add_technical_indicators(df)
    
    # Create target
    df = create_target(df, horizon=1, threshold=0.0)
    
    # Save processed data
    df.to_csv(DATA_DIR / "reliance_daily_processed.csv")
    print(f"Saved processed data to {DATA_DIR / 'reliance_daily_processed.csv'}")
    
    # Prepare sequences
    X, y, scaler, features = prepare_sequences(df, sequence_length=60)
    
    # Print class distribution
    print(f"\nTarget distribution:")
    print(f"  Up days (1): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Down days (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    
    # Save for training
    np.save(DATA_DIR / "X_reliance_daily.npy", X)
    np.save(DATA_DIR / "y_reliance_daily.npy", y)
    joblib.dump(scaler, DATA_DIR / "scaler_reliance_daily.pkl")
    
    # Save feature names
    with open(DATA_DIR / "features_reliance_daily.txt", 'w') as f:
        f.write('\n'.join(features))
    
    print("\n✓ Data preparation complete!")
    print(f"  - Sequences: {DATA_DIR / 'X_reliance_daily.npy'}")
    print(f"  - Targets: {DATA_DIR / 'y_reliance_daily.npy'}")
    print(f"  - Scaler: {DATA_DIR / 'scaler_reliance_daily.pkl'}")
    
    return X, y, scaler


if __name__ == "__main__":
    main()
