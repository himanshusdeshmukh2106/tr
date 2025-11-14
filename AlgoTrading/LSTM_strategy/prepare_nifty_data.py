"""
Prepare NIFTY 50 data from multiple years (2021-2025)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config.config import DATA_DIR


def load_and_combine_nifty_data():
    """Load all NIFTY 50 CSV files and combine them"""
    print("Loading NIFTY 50 data from multiple years...")
    
    # Get the parent directory (AlgoTrading root)
    base_dir = Path(__file__).parent.parent
    
    data_files = [
        base_dir / "NIFTY 50-12-11-2021-to-12-11-2022.csv",
        base_dir / "NIFTY 50-12-11-2022-to-12-11-2023.csv",
        base_dir / "NIFTY 50-12-11-2023-to-12-11-2024.csv",
        base_dir / "NIFTY 50-12-11-2024-to-12-11-2025.csv"
    ]
    
    dfs = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            print(f"  ✓ Loaded {file}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ Error loading {file}: {e}")
    
    # Combine all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows combined: {len(df_combined)}")
    
    return df_combined


def clean_nifty_data(df):
    """Clean and format NIFTY data"""
    print("\nCleaning data...")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Rename columns
    column_mapping = {
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Shares Traded': 'Volume',
        'Turnover (₹ Cr)': 'Turnover'
    }
    df = df.rename(columns=column_mapping)
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.set_index('Date')
    
    # Remove commas and convert to numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Sort by date and remove duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"After cleaning: {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def add_technical_indicators(df):
    """Add technical indicators"""
    print("\nAdding technical indicators...")
    
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
    
    # Price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Price_vs_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
    df['Price_vs_EMA21'] = ((df['Close'] - df['EMA_21']) / df['EMA_21']) * 100
    
    return df


def create_target(df, horizon=1):
    """Create target (1 = up, 0 = down)"""
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    return df


def prepare_sequences(df, sequence_length=60):
    """Create sequences for LSTM"""
    print(f"\nCreating sequences (length={sequence_length})...")
    
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
    
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_clean[feature_cols])
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df_clean['Target'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nSequences created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"\nTarget distribution:")
    print(f"  Up days (1): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Down days (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    
    return X, y, scaler, feature_cols


def main():
    """Main pipeline"""
    print("="*70)
    print("NIFTY 50 DATA PREPARATION (2021-2025)")
    print("="*70)
    
    # Load and combine
    df = load_and_combine_nifty_data()
    
    # Clean
    df = clean_nifty_data(df)
    
    # Add indicators
    df = add_technical_indicators(df)
    
    # Create target
    df = create_target(df, horizon=1)
    
    # Save processed
    df.to_csv(DATA_DIR / "nifty50_processed.csv")
    print(f"\n✓ Saved: {DATA_DIR / 'nifty50_processed.csv'}")
    
    # Prepare sequences
    X, y, scaler, features = prepare_sequences(df, sequence_length=60)
    
    # Save
    np.save(DATA_DIR / "X_nifty50.npy", X)
    np.save(DATA_DIR / "y_nifty50.npy", y)
    joblib.dump(scaler, DATA_DIR / "scaler_nifty50.pkl")
    
    with open(DATA_DIR / "features_nifty50.txt", 'w') as f:
        f.write('\n'.join(features))
    
    print("\n" + "="*70)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"  Sequences: {DATA_DIR / 'X_nifty50.npy'}")
    print(f"  Targets: {DATA_DIR / 'y_nifty50.npy'}")
    print(f"  Scaler: {DATA_DIR / 'scaler_nifty50.pkl'}")
    print(f"\nReady for training with {len(X)} samples!")


if __name__ == "__main__":
    main()
