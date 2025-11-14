# EMA Trap Strategy - Complete Documentation

## Strategy Overview

**Type:** Intraday Mean Reversion / False Breakout  
**Timeframe:** 5-minute candles  
**Market:** Indian stocks (NSE) or US stocks  
**Trading Hours:** 9:15 AM - 3:30 PM IST

## Core Concept

The strategy exploits **false breakouts (traps)** around the 21 EMA. When price breaks the EMA but fails to sustain, it creates a trap that often reverses strongly.

### Trap Types

**1. Downside Trap → LONG Signal**
```
Price crosses below 21 EMA (bearish)
↓
Then closes back above 21 EMA (trap!)
↓
GO LONG (expect upward reversal)
```

**2. Upside Trap → SHORT Signal**
```
Price crosses above 21 EMA (bullish)
↓
Then closes back below 21 EMA (trap!)
↓
GO SHORT (expect downward reversal)
```

## Entry Rules

### Time Windows (Only enter during these times)
1. **Morning Window:** 9:15 AM - 9:30 AM
2. **Mid-Morning Window:** 10:00 AM - 11:00 AM

### Entry Conditions (ALL must be met)

1. **Candle Body Size**
   - Body = |Close - Open|
   - Body must be ≤ 0.20% of price
   - Ensures we're not chasing big moves

2. **ADX Range**
   - ADX must be between 20 and 36
   - ADX < 20: Too choppy, no trend
   - ADX > 36: Trend too strong, trap less likely
   - Sweet spot: 20-36 (moderate trend)

3. **EMA Trap Pattern**
   - For LONG: Price was below 21 EMA, now closes above
   - For SHORT: Price was above 21 EMA, now closes below
   - Lookback: Check last 5 candles for the cross
   - Minimum distance: 0.05% from EMA for valid trap

## Exit Rules

### Stop Loss
- **LONG:** 0.5% below entry
- **SHORT:** 0.5% above entry

### Target
- **LONG:** 1.0% above entry
- **SHORT:** 1.0% below entry

### Trailing Stop
- Activates after 0.5% profit
- Trails by 0.3%
- Locks in profits while letting winners run

### Time Exit
- Close all positions by 3:15 PM
- Avoid overnight risk

## Risk Management

### Position Sizing
- 10% of capital per trade
- Maximum 3 concurrent positions
- Never risk more than 1.5% total capital

### Daily Limits
- Max 5 trades per day
- Stop trading after 3 consecutive losses
- Max daily loss: 2% of capital

## Example Trade Scenarios

### Scenario 1: Successful Long Trade
```
Time: 9:20 AM
Price: 100.00
21 EMA: 100.10

Setup:
- 9:18 AM: Price at 99.85 (below EMA) ✓
- 9:20 AM: Price closes at 100.15 (above EMA) ✓
- Candle body: 0.15% ✓
- ADX: 28 ✓

Action: GO LONG at 100.15
Stop Loss: 99.65 (0.5% below)
Target: 101.15 (1.0% above)

Result: Target hit at 10:05 AM
P&L: +1.0% = $1,000 profit
```

### Scenario 2: Stop Loss Hit
```
Time: 10:15 AM
Price: 200.00
21 EMA: 199.80

Setup:
- 10:12 AM: Price at 200.20 (above EMA) ✓
- 10:15 AM: Price closes at 199.70 (below EMA) ✓
- Candle body: 0.18% ✓
- ADX: 32 ✓

Action: GO SHORT at 199.70
Stop Loss: 200.70 (0.5% above)
Target: 197.70 (1.0% below)

Result: Stop loss hit at 10:25 AM
P&L: -0.5% = -$500 loss
```

## Technical Indicators

### 21 EMA (Exponential Moving Average)
- **Purpose:** Dynamic support/resistance
- **Calculation:** EMA = Price(t) × k + EMA(y) × (1 − k)
  - k = 2 / (21 + 1) = 0.0909
- **Why 21?** Represents ~1 month of trading days

### ADX (Average Directional Index)
- **Purpose:** Measure trend strength
- **Range:** 0-100
- **Interpretation:**
  - < 20: Weak/no trend (avoid)
  - 20-36: Moderate trend (ideal for traps)
  - > 36: Strong trend (traps less likely)

## LSTM Enhancement

The LSTM model learns to:
1. Identify trap patterns more accurately
2. Filter false signals
3. Predict trap success probability
4. Optimize entry timing

### Model Input Features
- OHLCV data (5-min candles)
- 21 EMA values
- ADX values
- Candle body size
- Price distance from EMA
- Volume ratios

### Model Output
- Probability of successful trap (0-1)
- Use threshold (e.g., 0.7) for high-confidence trades

## Backtesting Results

### Expected Performance (Typical)
- **Win Rate:** 55-65%
- **Profit Factor:** 1.5-2.0
- **Sharpe Ratio:** 1.0-1.5
- **Max Drawdown:** 5-10%
- **Average Trade:** 0.3-0.5%

### Key Metrics to Monitor
1. Win rate by time window
2. Performance by ADX range
3. Trap success rate
4. Average holding time
5. Slippage impact

## Implementation Steps

### 1. Data Preparation
```bash
python src/intraday_data_prep.py
```
- Downloads 5-min data
- Calculates indicators
- Identifies trap patterns
- Creates LSTM sequences

### 2. Model Training
```bash
# Local training
python src/train_local.py

# Or GCP training
python src/train_gcp.py
```

### 3. Backtesting
```bash
python src/backtest_trap.py
```

### 4. Live Trading (Future)
- Connect to broker API
- Real-time data feed
- Automated order execution
- Risk monitoring

## Optimization Ideas

### Parameter Tuning
- EMA period (try 13, 21, 34)
- ADX range (try 15-30, 25-40)
- Candle body threshold (try 0.15%, 0.25%)
- Stop loss / target ratios

### Additional Filters
- Volume confirmation (above average)
- Market regime filter (trending vs ranging)
- Volatility filter (ATR-based)
- Time-of-day performance

### Advanced Features
- Multiple timeframe confirmation
- Order flow analysis
- Market microstructure
- Sentiment indicators

## Risk Warnings

⚠️ **Important Considerations:**

1. **Slippage:** 5-min data can have significant slippage
2. **Liquidity:** Ensure sufficient volume for entries/exits
3. **Gap Risk:** Overnight gaps can invalidate stops
4. **Market Conditions:** Strategy works best in ranging markets
5. **Overfitting:** Backtest on out-of-sample data
6. **Transaction Costs:** Include brokerage, taxes, slippage

## Next Steps

1. ✅ Strategy implemented
2. ⏳ Download historical data
3. ⏳ Train LSTM model
4. ⏳ Run backtest
5. ⏳ Optimize parameters
6. ⏳ Paper trade
7. ⏳ Live trade (small size)

Ready to run the strategy! 🚀
