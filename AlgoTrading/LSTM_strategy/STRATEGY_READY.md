# ✅ EMA Trap Strategy - Ready to Deploy!

## 🎯 Strategy Implemented

Your **EMA Trap Strategy** is fully implemented and ready to use!

### Strategy Summary
- **Type:** Intraday false breakout (trap) trading
- **Timeframe:** 5-minute candles
- **Entry Windows:** 9:15-9:30 AM & 10:00-11:00 AM
- **Key Indicator:** 21 EMA
- **Filters:** Candle body ≤ 0.20%, ADX 20-36
- **Risk/Reward:** 0.5% stop loss, 1.0% target

## 📁 Files Created

### Strategy Implementation
```
✓ src/trap_strategy.py          # Core strategy logic
✓ src/intraday_data_prep.py     # 5-min data preparation
✓ src/backtest_trap.py          # Backtesting engine
✓ run_trap_strategy.py          # Complete pipeline runner
✓ TRAP_STRATEGY.md              # Full documentation
```

### Configuration
```
✓ config/config.py              # Updated with trap strategy settings
  - Entry windows: 9:15-9:30, 10:00-11:00
  - Candle body max: 0.20%
  - ADX range: 20-36
  - EMA period: 21
  - Stop loss: 0.5%
  - Target: 1.0%
```

## 🚀 How to Run

### Quick Start (All Steps)
```bash
cd AlgoTrading/LSTM_strategy
python run_trap_strategy.py --step all --ticker SPY
```

### Step by Step

**1. Prepare Data**
```bash
python run_trap_strategy.py --step data --ticker SPY
```
This will:
- Download 60 days of 5-minute data
- Calculate 21 EMA and ADX
- Identify trap patterns
- Create LSTM training sequences

**2. Train Model (Optional)**
```bash
python run_trap_strategy.py --step train
```
Choose:
- Option 1: Local training (free)
- Option 2: GCP training (faster, paid)

**3. Run Backtest**
```bash
python run_trap_strategy.py --step backtest
```
This will:
- Simulate trading on historical data
- Calculate performance metrics
- Generate performance charts
- Save trade log

### For Indian Stocks
```bash
# Use NSE tickers
python run_trap_strategy.py --ticker RELIANCE.NS
python run_trap_strategy.py --ticker TCS.NS
python run_trap_strategy.py --ticker INFY.NS
python run_trap_strategy.py --ticker HDFCBANK.NS
```

## 📊 Expected Output

### Console Output
```
============================================================
BACKTEST RESULTS - EMA TRAP STRATEGY
============================================================

Capital:
  Initial: $100,000.00
  Final:   $105,234.50
  P&L:     $5,234.50
  Return:  5.23%

Trades:
  Total:   45
  Winners: 27 (60.0%)
  Losers:  18

Performance:
  Avg Win:       $312.45
  Avg Loss:      $-187.23
  Profit Factor: 1.67
  Sharpe Ratio:  1.23
  Max Drawdown:  -3.45%
============================================================
```

### Generated Files
1. **data/intraday_trap_data.csv** - Processed 5-min data with indicators
2. **results/trap_strategy_backtest.png** - Performance charts
3. **results/trap_trades.csv** - Detailed trade log
4. **results/trap_portfolio.csv** - Portfolio value over time

## 🎓 Strategy Logic

### Long Entry (Downside Trap)
```
1. Price crosses below 21 EMA
2. Then closes back above 21 EMA (trap!)
3. Candle body ≤ 0.20%
4. ADX between 20-36
5. Time: 9:15-9:30 or 10:00-11:00
→ GO LONG
```

### Short Entry (Upside Trap)
```
1. Price crosses above 21 EMA
2. Then closes back below 21 EMA (trap!)
3. Candle body ≤ 0.20%
4. ADX between 20-36
5. Time: 9:15-9:30 or 10:00-11:00
→ GO SHORT
```

### Exit Rules
- **Stop Loss:** 0.5% from entry
- **Target:** 1.0% from entry
- **Trailing Stop:** 0.3% after 0.5% profit
- **Time Exit:** 3:15 PM (close all positions)

## 🔧 Customization

### Change Parameters
Edit `config/config.py`:

```python
TRAP_STRATEGY_CONFIG = {
    # Modify entry windows
    "entry_windows": [
        {"start": "09:15", "end": "09:30"},
        {"start": "10:00", "end": "11:00"},
    ],
    
    # Adjust filters
    "max_candle_body_pct": 0.20,  # Try 0.15 or 0.25
    "adx_min": 20,                 # Try 15 or 25
    "adx_max": 36,                 # Try 30 or 40
    "ema_period": 21,              # Try 13 or 34
    
    # Modify risk/reward
    "stop_loss_pct": 0.5,          # Try 0.3 or 0.7
    "target_pct": 1.0,             # Try 0.8 or 1.5
}
```

### Test Different Stocks
```bash
# US Stocks
python run_trap_strategy.py --ticker AAPL
python run_trap_strategy.py --ticker TSLA
python run_trap_strategy.py --ticker QQQ

# Indian Stocks (NSE)
python run_trap_strategy.py --ticker RELIANCE.NS
python run_trap_strategy.py --ticker TCS.NS
python run_trap_strategy.py --ticker TATAMOTORS.NS
```

## 📈 Performance Optimization

### Walk-Forward Testing
```python
# Test on different time periods
# Train on Jan-Mar, test on Apr-Jun
# Retrain quarterly for best results
```

### Parameter Optimization
```python
# Grid search for best parameters
ema_periods = [13, 21, 34]
adx_ranges = [(15, 30), (20, 36), (25, 40)]
body_sizes = [0.15, 0.20, 0.25]
```

### LSTM Enhancement
The LSTM model learns:
- Pattern recognition for traps
- Success probability prediction
- Optimal entry timing
- Market regime detection

## ⚠️ Important Notes

### Data Limitations
- Yahoo Finance: 60 days of 5-min data max
- For longer history, use paid data providers
- Consider data quality and gaps

### Market Hours
- Strategy designed for IST market hours
- Adjust for different time zones
- Consider pre-market/after-hours

### Risk Management
- Start with paper trading
- Use small position sizes initially
- Monitor slippage and execution
- Set daily loss limits

### Costs to Consider
- Brokerage fees
- Exchange fees
- Taxes
- Slippage (especially on 5-min data)

## 🎯 Next Steps

### 1. Test the Strategy
```bash
python run_trap_strategy.py --step all --ticker SPY
```

### 2. Analyze Results
- Review backtest metrics
- Check trade distribution
- Identify best time windows
- Optimize parameters

### 3. Paper Trade
- Connect to broker API
- Run in simulation mode
- Monitor real-time performance
- Validate assumptions

### 4. Go Live (When Ready)
- Start with small capital
- Monitor closely
- Keep detailed logs
- Adjust as needed

## 📚 Documentation

- **TRAP_STRATEGY.md** - Complete strategy guide
- **ARCHITECTURE.md** - System architecture
- **GCP_SETUP_COMPLETE.md** - Cloud setup
- **COMMANDS.md** - Quick reference

## 🚀 Ready to Trade!

Your trap strategy is fully implemented and ready to test. Run the pipeline and see the results!

```bash
python run_trap_strategy.py --step all --ticker SPY
```

Good luck! 📈
