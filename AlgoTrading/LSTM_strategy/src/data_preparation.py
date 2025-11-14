"""
Data preparation and feature engineering for LSTM model
"""
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import TECHNICAL_INDICATORS, DATA_CONFIG, DATA_DIR


class DataPreparation:
    def __init__(self, ticker=None, start_date=None, end_date=None):
        self.ticker = ticker or DATA_CONFIG["ticker"]
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def download_data(self):
        """Download historical data from Yahoo Finance"""
        print(f"Downloading data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        print(f"Downloaded {len(df)} rows")
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators as features"""
        print("Adding technical indicators...")
        
        # Simple Moving Averages
        for period in TECHNICAL_INDICATORS["sma_periods"]:
            df[f'SMA_{period}'] = SMAIndicator(df['Close'], window=period).sma_indicator()
        
        # Exponential Moving Averages
        for period in TECHNICAL_INDICATORS["ema_periods"]:
            df[f'EMA_{period}'] = EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # RSI
        rsi_period = TECHNICAL_INDICATORS["rsi_period"]
        df['RSI'] = RSIIndicator(df['Close'], window=rsi_period).rsi()
        
        # MACD
        macd = MACD(
            df['Close'],
            window_fast=TECHNICAL_INDICATORS["macd_fast"],
            window_slow=TECHNICAL_INDICATORS["macd_slow"],
            window_sign=TECHNICAL_INDICATORS["macd_signal"]
        )
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(
            df['Close'],
            window=TECHNICAL_INDICATORS["bb_period"],
            window_dev=TECHNICAL_INDICATORS["bb_std"]
        )
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # ATR
        atr = AverageTrueRange(
            df['High'], df['Low'], df['Close'],
            window=TECHNICAL_INDICATORS["atr_period"]
        )
        df['ATR'] = atr.average_true_range()
        
        # ADX
        adx = ADXIndicator(
            df['High'], df['Low'], df['Close'],
            window=TECHNICAL_INDICATORS["adx_period"]
        )
        df['ADX'] = adx.adx()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df
    
    def create_target(self, df, horizon=1, threshold=0.0):
        """
        Create target variable for classification
        horizon: number of periods to look ahead
        threshold: minimum return to consider as buy signal
        """
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > threshold).astype(int)
        return df
    
    def prepare_sequences(self, df, sequence_length=60, feature_columns=None):
        """
        Prepare sequences for LSTM input
        Returns: X (sequences), y (targets)
        """
        if feature_columns is None:
            # Use all numeric columns except target and future return
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col not in ['Target', 'Future_Return']]
        
        # Drop NaN values
        df = df.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y), feature_columns
    
    def save_data(self, df, filename):
        """Save processed data"""
        filepath = DATA_DIR / filename
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def save_scaler(self, filename="scaler.pkl"):
        """Save the fitted scaler"""
        filepath = DATA_DIR / filename
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filename="scaler.pkl"):
        """Load a fitted scaler"""
        filepath = DATA_DIR / filename
        self.scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
        return self.scaler


def main():
    """Example usage"""
    # Initialize
    prep = DataPreparation(ticker="SPY", start_date="2020-01-01", end_date="2024-08-31")
    
    # Download and process data
    df = prep.download_data()
    df = prep.add_technical_indicators(df)
    df = prep.create_target(df, horizon=1, threshold=0.0)
    
    # Save processed data
    prep.save_data(df, "processed_data.csv")
    
    # Prepare sequences
    X, y, feature_cols = prep.prepare_sequences(df, sequence_length=60)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Features used: {len(feature_cols)}")
    
    # Save scaler
    prep.save_scaler()
    
    # Save sequences
    np.save(DATA_DIR / "X_sequences.npy", X)
    np.save(DATA_DIR / "y_targets.npy", y)
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
