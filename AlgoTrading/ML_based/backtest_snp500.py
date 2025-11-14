"""
ML-Based Trading Strategy Backtest for S&P 500
Converted from Jupyter notebook to standalone Python script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading S&P 500 data...")
file_path = "AlgoTrading/ML_based/data/snp500_daily.csv"
snp500_data = pd.read_csv(file_path)
snp500_data["Date"] = pd.to_datetime(snp500_data["Date"])
snp500_data.sort_values(by="Date", inplace=True)

# Calculate technical indicators
print("Calculating technical indicators...")
snp500_data["SMA_20"] = snp500_data["Close"].rolling(window=20).mean()
snp500_data["SMA_50"] = snp500_data["Close"].rolling(window=50).mean()
snp500_data["MACD"] = (snp500_data["Close"].ewm(span=12, adjust=False).mean() - 
                       snp500_data["Close"].ewm(span=26, adjust=False).mean())
snp500_data["MACD_Signal"] = snp500_data["MACD"].ewm(span=9, adjust=False).mean()
snp500_data["ATR"] = (snp500_data["High"] - snp500_data["Low"]).rolling(window=14).mean()

# RSI calculation
delta = snp500_data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
snp500_data["RSI"] = 100 - (100 / (1 + rs))

# Bollinger Bands
snp500_data["BB_Middle"] = snp500_data["Close"].rolling(window=20).mean()
snp500_data["BB_Std"] = snp500_data["Close"].rolling(window=20).std()
snp500_data["BB_Upper"] = snp500_data["BB_Middle"] + (2 * snp500_data["BB_Std"])
snp500_data["BB_Lower"] = snp500_data["BB_Middle"] - (2 * snp500_data["BB_Std"])

# ADX calculation
snp500_data["TR"] = np.maximum(snp500_data["High"] - snp500_data["Low"],
                                np.maximum(abs(snp500_data["High"] - snp500_data["Close"].shift(1)),
                                          abs(snp500_data["Low"] - snp500_data["Close"].shift(1))))
snp500_data["DMplus"] = np.where((snp500_data["High"] - snp500_data["High"].shift(1)) > 
                                  (snp500_data["Low"].shift(1) - snp500_data["Low"]),
                                  np.maximum(snp500_data["High"] - snp500_data["High"].shift(1), 0), 0)
snp500_data["DMminus"] = np.where((snp500_data["Low"].shift(1) - snp500_data["Low"]) > 
                                   (snp500_data["High"] - snp500_data["High"].shift(1)),
                                   np.maximum(snp500_data["Low"].shift(1) - snp500_data["Low"], 0), 0)
snp500_data["TR14"] = snp500_data["TR"].rolling(window=14).sum()
snp500_data["DMplus14"] = snp500_data["DMplus"].rolling(window=14).sum()
snp500_data["DMminus14"] = snp500_data["DMminus"].rolling(window=14).sum()
snp500_data["DIplus14"] = 100 * (snp500_data["DMplus14"] / snp500_data["TR14"])
snp500_data["DIminus14"] = 100 * (snp500_data["DMminus14"] / snp500_data["TR14"])
snp500_data["DX"] = 100 * (np.abs(snp500_data["DIplus14"] - snp500_data["DIminus14"]) / 
                           (snp500_data["DIplus14"] + snp500_data["DIminus14"]))
snp500_data["ADX"] = snp500_data["DX"].rolling(window=14).mean()

snp500_data.dropna(inplace=True)
print(f"Data loaded: {len(snp500_data)} rows")

def adjust_signals(data, rsi_oversold=40, rsi_overbought=60, bb_band=0.5, 
                   adx_threshold=25, buy_threshold=3, sell_threshold=4):
    """Generate trading signals based on technical indicators"""
    buy_conditions = [
        (data["RSI"] < rsi_oversold),
        (data["SMA_20"] > data["SMA_50"]),
        (data["MACD"] > data["MACD_Signal"]),
        (data["Close"] <= (data["BB_Lower"] + ((data["BB_Middle"] - data["BB_Lower"]) * bb_band))),
        (data["ADX"] > adx_threshold),
    ]
    
    sell_conditions = [
        (data["RSI"] > rsi_overbought),
        (data["SMA_20"] < data["SMA_50"]),
        (data["MACD"] < data["MACD_Signal"]),
        (data["Close"] >= (data["BB_Upper"] - ((data["BB_Upper"] - data["BB_Middle"]) * bb_band))),
        (data["ADX"] > adx_threshold),
    ]
    
    data["Buy_Signal"] = sum(buy_conditions)
    data["Sell_Signal"] = sum(sell_conditions)
    
    data["Adjusted_Signal"] = 0
    data.loc[data["Buy_Signal"] >= buy_threshold, "Adjusted_Signal"] = 1
    data.loc[data["Sell_Signal"] >= sell_threshold, "Adjusted_Signal"] = -1
    
    return data

# Apply signals
print("Generating trading signals...")
snp500_data = adjust_signals(snp500_data)

# Backtest function
def backtest_strategy(data, initial_capital=100000):
    """Run backtest on trading strategy"""
    position = 0
    cash = initial_capital
    portfolio_value = []
    
    for _, row in data.iterrows():
        if row["Adjusted_Signal"] == 1 and cash > 0:
            position = cash / row["Close"]
            cash = 0
        elif row["Adjusted_Signal"] == -1 and position > 0:
            cash = position * row["Close"]
            position = 0
        
        portfolio_value.append(cash + (position * row["Close"]))
    
    data["Portfolio_Value"] = portfolio_value
    data["Daily_Return"] = data["Portfolio_Value"].pct_change()
    
    final_value = portfolio_value[-1]
    cumulative_return = (final_value - initial_capital) / initial_capital * 100
    sharpe_ratio = data["Daily_Return"].mean() / data["Daily_Return"].std() * (252**0.5)
    max_drawdown = (data["Portfolio_Value"] / data["Portfolio_Value"].cummax() - 1).min()
    num_trades = data["Adjusted_Signal"].abs().sum()
    
    return {
        "Final Portfolio Value": final_value,
        "Cumulative Return (%)": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown (%)": max_drawdown * 100,
        "Number of Trades": num_trades
    }

# Run initial backtest
print("\n" + "="*60)
print("INITIAL BACKTEST RESULTS (Basic Strategy)")
print("="*60)
results = backtest_strategy(snp500_data.copy())
for key, value in results.items():
    print(f"{key}: {value:.2f}")

# Calculate buy and hold benchmark
initial_price = snp500_data["Close"].iloc[0]
final_price = snp500_data["Close"].iloc[-1]
buy_hold_return = (final_price - initial_price) / initial_price * 100
print(f"\nBuy & Hold Return (%): {buy_hold_return:.2f}")
print(f"Strategy Outperformance: {results['Cumulative Return (%)'] - buy_hold_return:.2f}%")

print("\n" + "="*60)
print("Backtest completed successfully!")
print("="*60)
