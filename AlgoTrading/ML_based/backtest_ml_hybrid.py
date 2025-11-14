"""
ML-Based Hybrid Trading Strategy Backtest for S&P 500
Combines Technical Analysis with Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML-BASED HYBRID TRADING STRATEGY BACKTEST")
print("="*70)

# Load data
print("\n[1/6] Loading S&P 500 data...")
snp500_data = pd.read_csv("AlgoTrading/ML_based/data/snp500_daily.csv")
snp500_data["Date"] = pd.to_datetime(snp500_data["Date"])
snp500_data.sort_values(by="Date", inplace=True)

# Calculate technical indicators
print("[2/6] Calculating technical indicators...")
snp500_data["SMA_20"] = snp500_data["Close"].rolling(window=20).mean()
snp500_data["SMA_50"] = snp500_data["Close"].rolling(window=50).mean()
snp500_data["MACD"] = (snp500_data["Close"].ewm(span=12, adjust=False).mean() - 
                       snp500_data["Close"].ewm(span=26, adjust=False).mean())
snp500_data["MACD_Signal"] = snp500_data["MACD"].ewm(span=9, adjust=False).mean()
snp500_data["ATR"] = (snp500_data["High"] - snp500_data["Low"]).rolling(window=14).mean()

# RSI
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

# ADX
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

def adjust_signals(data, rsi_oversold=40, rsi_overbought=60, bb_band=0.5, 
                   adx_threshold=25, buy_threshold=3, sell_threshold=4):
    """Generate trading signals"""
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

# Generate signals
print("[3/6] Generating technical analysis signals...")
snp500_data = adjust_signals(snp500_data)

# Prepare ML data
print("[4/6] Preparing machine learning features...")
snp500_data["Close_Change"] = snp500_data["Close"].pct_change().shift(-1)
snp500_data["Target"] = 0
snp500_data.loc[snp500_data["Close_Change"] > 0.002, "Target"] = 1
snp500_data.loc[snp500_data["Close_Change"] < -0.002, "Target"] = -1

snp500_data["Momentum"] = snp500_data["Close"].pct_change(5)
snp500_data["ATR_Lag_1"] = snp500_data["ATR"].shift(1)
snp500_data["RSI_Lag_1"] = snp500_data["RSI"].shift(1)
snp500_data["SMA_20_Lag_1"] = snp500_data["SMA_20"].shift(1)
snp500_data["SMA_50_Lag_1"] = snp500_data["SMA_50"].shift(1)
snp500_data["Daily_Return"] = snp500_data["Close"].pct_change()
snp500_data["Daily_Return_Lag_1"] = snp500_data["Daily_Return"].shift(1)
snp500_data["BB_Width_Lag_1"] = (snp500_data["BB_Upper"] - snp500_data["BB_Lower"]).shift(1)

snp500_data = snp500_data.dropna()

# Filter data from 2021 onwards for testing
test_data = snp500_data[snp500_data["Date"] >= "2021-01-01"].copy()

features = ["RSI_Lag_1", "SMA_20_Lag_1", "SMA_50_Lag_1", "Daily_Return_Lag_1", 
            "BB_Width_Lag_1", "Momentum", "ADX", "ATR_Lag_1"]
X = test_data[features]
y = test_data["Target"]

# Walk-forward ML training
print("[5/6] Training ML ensemble model with walk-forward validation...")
tscv = TimeSeriesSplit(n_splits=5)
ensemble_model = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=100)),
        ("gb", GradientBoostingClassifier(random_state=42, n_estimators=100)),
    ],
    voting="soft"
)

results_df = pd.DataFrame(columns=["Date", "ML_Prediction", "Confidence", "TA_Signal", "Hybrid_Signal"])

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train = y.iloc[train_index]
    dates_test = test_data.iloc[test_index]["Date"]
    ta_signals_test = test_data.iloc[test_index]["Adjusted_Signal"]
    
    ensemble_model.fit(X_train, y_train)
    probabilities = ensemble_model.predict_proba(X_test)
    predictions = ensemble_model.predict(X_test)
    confidences = probabilities.max(axis=1)
    
    high_confidence_threshold = 0.75
    hybrid_signals = [pred if conf > high_confidence_threshold else ta 
                     for pred, conf, ta in zip(predictions, confidences, ta_signals_test)]
    
    temp_df = pd.DataFrame({
        "Date": dates_test,
        "ML_Prediction": predictions,
        "Confidence": confidences,
        "TA_Signal": ta_signals_test,
        "Hybrid_Signal": hybrid_signals,
    })
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

results_df.sort_values(by="Date", inplace=True)
results_df.reset_index(drop=True, inplace=True)

# Backtest hybrid strategy
print("[6/6] Running backtest on hybrid strategy...")
initial_capital = 100000
position = 0
cash = initial_capital
portfolio_value = []
entry_price = None
stop_loss_pct = 0.05
take_profit_pct = 0.1

for _, row in results_df.iterrows():
    current_price = test_data.loc[test_data["Date"] == row["Date"], "Close"].values[0]
    
    # Stop-loss and take-profit
    if position > 0 and entry_price:
        if (current_price <= entry_price * (1 - stop_loss_pct)) or \
           (current_price >= entry_price * (1 + take_profit_pct)):
            cash = position * current_price
            position = 0
            entry_price = None
    
    # Execute signals
    if row["Hybrid_Signal"] == 1 and cash > 0:
        position = cash / current_price
        entry_price = current_price
        cash = 0
    elif row["Hybrid_Signal"] == -1 and position > 0:
        cash = position * current_price
        position = 0
        entry_price = None
    
    portfolio_value.append(cash + (position * current_price))

results_df["Portfolio_Value"] = portfolio_value
results_df["Daily_Return"] = results_df["Portfolio_Value"].pct_change()

# Calculate metrics
final_value = portfolio_value[-1]
cumulative_return = (final_value - initial_capital) / initial_capital * 100
sharpe_ratio = results_df["Daily_Return"].mean() / results_df["Daily_Return"].std() * (252**0.5)
max_drawdown = (results_df["Portfolio_Value"] / results_df["Portfolio_Value"].cummax() - 1).min()
num_trades = results_df["Hybrid_Signal"].diff().abs().sum()

# Buy and hold benchmark
results_df["Close"] = test_data.loc[test_data["Date"].isin(results_df["Date"]), "Close"].values
initial_price = results_df["Close"].iloc[0]
final_price = results_df["Close"].iloc[-1]
buy_hold_return = (final_price - initial_price) / initial_price * 100

# Print results
print("\n" + "="*70)
print("BACKTEST RESULTS - ML HYBRID STRATEGY")
print("="*70)
print(f"Test Period: {results_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {results_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"\nFinal Portfolio Value: ${final_value:,.2f}")
print(f"Cumulative Return: {cumulative_return:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
print(f"Number of Trades: {int(num_trades)}")
print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy Outperformance: {cumulative_return - buy_hold_return:.2f}%")
print("="*70)

# Save results
results_df.to_csv("AlgoTrading/ML_based/backtest_results.csv", index=False)
print("\nResults saved to: AlgoTrading/ML_based/backtest_results.csv")
print("\nBacktest completed successfully!")
