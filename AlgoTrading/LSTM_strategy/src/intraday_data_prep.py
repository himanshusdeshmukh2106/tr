"""
Intraday data preparation for trap strategy (5-minute data)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import DATA_CONFIG, DATA_DIR, TRAP_STRATEGY_CONFIG
from src.trap_strategy import TrapStrategy


class IntradayDataPrep:
    def __init__(self, ticker=None):
        self.ticker = ticker or DATA_CONFIG["ticker"]
        self.strategy = TrapStrategy()
        
    def download_intraday_data(self, period="60d"):
        """
        Download 5-minute intraday data
        Note: Yahoo Finance limits intraday data to last 60 days
        """
        print(f"Downloading 5-minute data for {self.ticker}...")
        df = yf.download(
            self.ticker,
            period=period,
            interval="5m",
            progress=False
        )
        print(f"Downloaded {len(df)} 5-minute candles")
        return df
    
    def add_indicators(self, df):
        """Add technical indicators for trap strategy"""
        print("Adding indicators...")
        
        # 21 EMA (main indicator)
        df['EMA_21'] = self.strategy.calculate_ema(df['Close'], period=21)
        
        # ADX
        df['ADX'] = self.strategy.calculate_adx(
            df['High'], df['Low'], df['Close'], period=14
        )
        
        # Candle body size
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Pct'] = (df['Body_Size'] / df['Close']) * 100
        
        # Price position relative to EMA
        df['Price_vs_EMA'] = ((df['Close'] - df['EMA_21']) / df['EMA_21']) * 100
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def filter_trading_hours(self, df):
        """Filter data to only include market hours (9:15 AM - 3:30 PM IST)"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        # Filter by time
        df = df.between_time('09:15', '15:30')
        
        return df
    
    def create_labels(self, df):
        """
        Create labels based on trap strategy signals
        1 = LONG signal, 0 = SHORT signal, -1 = No signal
        """
        df['Signal'] = -1
        
        for idx in range(len(df)):
            signal = self.strategy.generate_signal(df, idx)
            if signal == 'LONG':
                df.iloc[idx, df.columns.get_loc('Signal')] = 1
            elif signal == 'SHORT':
                df.iloc[idx, df.columns.get_loc('Signal')] = 0
        
        return df

    
    def prepare_lstm_sequences(self, df, sequence_length=30):
        """Prepare sequences for LSTM training"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Select features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'EMA_21', 'ADX', 'Body_Pct', 'Price_vs_EMA', 'Volume_Ratio']
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            # Only create sequences for rows with valid signals
            if df['Signal'].iloc[i] != -1:
                X.append(scaled_data[i-sequence_length:i])
                y.append(df['Signal'].iloc[i])
        
        return np.array(X), np.array(y), scaler, feature_cols
    
    def save_data(self, df, filename="intraday_data.csv"):
        """Save processed data"""
        filepath = DATA_DIR / filename
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def generate_backtest_data(self):
        """Generate data suitable for backtesting"""
        # Download data
        df = self.download_intraday_data(period="60d")
        
        # Add indicators
        df = self.add_indicators(df)
        
        # Filter trading hours
        df = self.filter_trading_hours(df)
        
        # Create labels
        df = self.create_labels(df)
        
        # Save
        self.save_data(df, "intraday_trap_data.csv")
        
        return df


def main():
    """Example usage"""
    # For Indian market, use NSE stocks
    # Examples: "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"
    # For US market: "SPY", "QQQ", "AAPL", etc.
    
    prep = IntradayDataPrep(ticker="SPY")
    
    # Generate backtest data
    df = prep.generate_backtest_data()
    
    print(f"\nData Summary:")
    print(f"Total candles: {len(df)}")
    print(f"Long signals: {(df['Signal'] == 1).sum()}")
    print(f"Short signals: {(df['Signal'] == 0).sum()}")
    print(f"No signals: {(df['Signal'] == -1).sum()}")
    
    # Prepare LSTM sequences
    X, y, scaler, features = prep.prepare_lstm_sequences(df)
    print(f"\nLSTM Sequences:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {features}")
    
    # Save for training
    np.save(DATA_DIR / "X_intraday.npy", X)
    np.save(DATA_DIR / "y_intraday.npy", y)
    
    import joblib
    joblib.dump(scaler, DATA_DIR / "scaler_intraday.pkl")
    
    print("\n✓ Intraday data preparation complete!")


if __name__ == "__main__":
    main()
