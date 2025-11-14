# Freqtrade Analysis - Complete Overview

## What is Freqtrade?

**Freqtrade** is a free, open-source cryptocurrency trading bot written in Python. It's one of the most popular and mature algorithmic trading frameworks in the crypto space.

**Repository**: https://github.com/freqtrade/freqtrade
**Stars**: 30k+ (very popular)
**Language**: Python 3.11+
**License**: GPL-3.0

## Key Features

### 1. **Multi-Exchange Support**
- Binance, Bybit, OKX, Kraken, Gate.io, etc.
- Supports both spot and futures trading
- Uses CCXT library for exchange connectivity

### 2. **FreqAI - Machine Learning Module**
- **Adaptive ML**: Self-retraining models during live trading
- **Multiple Models**: LightGBM, XGBoost, CatBoost, PyTorch, TensorFlow
- **Feature Engineering**: Create 10k+ features from price data
- **Backtesting**: Realistic ML backtesting with periodic retraining
- **Outlier Detection**: Smart removal of anomalous data
- **Dimensionality Reduction**: PCA for large feature sets

### 3. **Strategy Development**
- Write strategies in Python using pandas
- 100+ example strategies available
- Technical indicators via TA-Lib
- Custom indicator creation

### 4. **Backtesting Engine**
- Historical data testing
- Realistic simulation with fees, slippage
- Hyperparameter optimization
- Walk-forward analysis

### 5. **Risk Management**
- Stop loss (fixed, trailing, dynamic)
- Take profit levels
- Position sizing
- Max drawdown protection
- Pair locks (prevent trading bad pairs)

### 6. **Control & Monitoring**
- **Telegram Bot**: Control via Telegram
- **Web UI**: Built-in web interface
- **REST API**: Programmatic control
- **Dry-run mode**: Test without real money

## Architecture

```
freqtrade/
├── freqtrade/
│   ├── freqtradebot.py       # Main bot logic
│   ├── strategy/             # Strategy framework
│   ├── freqai/               # Machine learning module
│   │   ├── prediction_models/  # ML models
│   │   ├── data_kitchen.py     # Data preprocessing
│   │   └── RL/                 # Reinforcement learning
│   ├── optimize/             # Backtesting & hyperopt
│   ├── exchange/             # Exchange connectors
│   ├── persistence/          # Database (SQLite)
│   ├── rpc/                  # Telegram/WebUI
│   └── data/                 # Data management
├── user_data/
│   ├── strategies/           # Your strategies
│   ├── data/                 # Downloaded market data
│   └── backtest_results/     # Backtest outputs
└── config.json               # Configuration
```

## How It Works

### 1. **Strategy Execution Flow**

```
1. Download market data (OHLCV)
2. Calculate technical indicators
3. Generate buy/sell signals
4. Execute orders on exchange
5. Monitor open positions
6. Exit based on strategy rules
```

### 2. **FreqAI ML Flow**

```
1. Define features (indicators)
2. Define labels (future price movement)
3. Train model on historical data
4. Predict on new candles
5. Use predictions in strategy
6. Retrain periodically (adaptive)
```

### 3. **Example Strategy Structure**

```python
class MyStrategy(IStrategy):
    # Strategy parameters
    minimal_roi = {"0": 0.10}  # 10% profit target
    stoploss = -0.05           # 5% stop loss
    timeframe = '5m'           # 5-minute candles
    
    def populate_indicators(self, dataframe, metadata):
        # Add technical indicators
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['ema'] = ta.EMA(dataframe, 21)
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata):
        # Buy signal
        dataframe.loc[
            (dataframe['rsi'] < 30) &
            (dataframe['close'] > dataframe['ema']),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe, metadata):
        # Sell signal
        dataframe.loc[
            (dataframe['rsi'] > 70),
            'exit_long'] = 1
        return dataframe
```

## FreqAI Machine Learning

### Supported Models

1. **LightGBM** (Gradient Boosting)
   - Fast training
   - Good for tabular data
   - Default choice

2. **XGBoost** (Gradient Boosting)
   - High accuracy
   - Robust to overfitting

3. **CatBoost** (Gradient Boosting)
   - Handles categorical features
   - Good default parameters

4. **PyTorch** (Deep Learning)
   - Neural networks
   - LSTM, CNN support

5. **TensorFlow/Keras** (Deep Learning)
   - Complex architectures
   - Transfer learning

6. **Reinforcement Learning**
   - Q-Learning, PPO
   - Learn optimal actions

### Feature Engineering

FreqAI automatically creates features from your indicators:

```python
def populate_any_indicators(self, dataframe):
    # Base indicators
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['ema_21'] = ta.EMA(dataframe, 21)
    
    # FreqAI creates:
    # - rsi_shift_1, rsi_shift_2, ... (historical values)
    # - rsi_pct_change (percentage change)
    # - rsi_rolling_mean_10 (rolling statistics)
    # - Hundreds more automatically!
    
    return dataframe

def set_freqai_targets(self, dataframe):
    # Define what to predict
    dataframe['&-target'] = (
        dataframe['close'].shift(-5) / dataframe['close'] - 1
    ) * 100  # Predict 5-candle future return
    
    return dataframe
```

### Training Process

1. **Data Collection**: Download historical data
2. **Feature Creation**: Generate 1000s of features
3. **Train/Test Split**: Time-based split (no lookahead)
4. **Model Training**: Fit model on train data
5. **Validation**: Test on unseen data
6. **Prediction**: Use model in live trading
7. **Retraining**: Periodically retrain (e.g., every 24h)

## Performance & Accuracy

### Important Notes

⚠️ **Freqtrade does NOT guarantee profits**
- No strategy has guaranteed accuracy
- Past performance ≠ future results
- Most strategies lose money long-term

### Typical Results

Based on community reports and research:

**Good Strategies:**
- Win rate: 45-55%
- Profit factor: 1.2-1.8
- Sharpe ratio: 0.5-1.5
- Max drawdown: 10-30%

**FreqAI Models:**
- Prediction accuracy: 52-58% (slightly better than random)
- Improvement over baseline: 2-8%
- Requires constant retraining
- Works best in trending markets

### Why Low Accuracy?

1. **Market Efficiency**: Crypto markets are semi-efficient
2. **Noise**: High volatility, random movements
3. **Fees**: Trading costs eat profits
4. **Slippage**: Price moves during execution
5. **Overfitting**: Models learn noise, not signal

### What Makes Freqtrade Good?

Despite low accuracy, Freqtrade is valuable because:

1. **Risk Management**: Stop losses prevent big losses
2. **Position Sizing**: Proper capital allocation
3. **Diversification**: Trade multiple pairs
4. **Automation**: No emotional trading
5. **Backtesting**: Test before risking money
6. **Community**: 1000s of users sharing strategies

## Comparison: Freqtrade vs Our Project

| Feature | Freqtrade | Our NIFTY Project |
|---------|-----------|-------------------|
| **Market** | Crypto (24/7) | Indian Stocks (6.5h/day) |
| **Exchanges** | 10+ crypto exchanges | NSE/BSE |
| **ML Models** | LightGBM, XGBoost, PyTorch, RL | LSTM, XGBoost, RF, LightGBM |
| **Accuracy** | 52-58% | 56.5% (ensemble) |
| **Backtesting** | ✅ Advanced | ⏳ Basic (to implement) |
| **Live Trading** | ✅ Full automation | ❌ Not implemented |
| **Risk Management** | ✅ Comprehensive | ⏳ Basic |
| **UI** | ✅ Web + Telegram | ❌ None |
| **Data** | Intraday (1m-1d) | Daily only |
| **Retraining** | ✅ Adaptive | ❌ Manual |
| **Community** | 30k+ stars | New project |

## Key Learnings for Our Project

### 1. **Architecture Lessons**

✅ **Modular Design**: Separate strategy, data, execution
✅ **Plugin System**: Easy to add new models/strategies
✅ **Configuration**: JSON config files
✅ **Persistence**: SQLite for trade history

### 2. **ML Improvements**

✅ **Adaptive Retraining**: Retrain models periodically
✅ **Feature Engineering**: Auto-generate features
✅ **Multiple Models**: Ensemble different approaches
✅ **Outlier Detection**: Remove bad data

### 3. **Risk Management**

✅ **Stop Loss**: Always use stop losses
✅ **Position Sizing**: Risk % of capital per trade
✅ **Max Positions**: Limit concurrent trades
✅ **Pair Locks**: Avoid repeatedly losing pairs

### 4. **Backtesting**

✅ **Realistic Fees**: Include transaction costs
✅ **Slippage**: Simulate execution delays
✅ **Walk-Forward**: Test on unseen data
✅ **Multiple Timeframes**: Test different periods

## What We Can Implement

### Short Term (1-2 weeks)

1. **Backtesting Engine**
   - Test ensemble on historical data
   - Calculate metrics (Sharpe, drawdown)
   - Include fees and slippage

2. **Risk Management**
   - Stop loss logic
   - Position sizing
   - Max drawdown limits

3. **Strategy Framework**
   - Modular strategy class
   - Easy to add new strategies
   - Configuration files

### Medium Term (1 month)

1. **Adaptive Retraining**
   - Retrain models weekly
   - Track model performance
   - Auto-switch to best model

2. **Feature Engineering**
   - Auto-generate features
   - Feature selection
   - Dimensionality reduction

3. **Paper Trading**
   - Simulate live trading
   - Track performance
   - No real money

### Long Term (3+ months)

1. **Live Trading**
   - Broker integration (Zerodha, Upstox)
   - Order execution
   - Position monitoring

2. **Web UI**
   - Dashboard
   - Performance charts
   - Trade history

3. **Telegram Bot**
   - Start/stop bot
   - View positions
   - Get alerts

## Conclusion

**Freqtrade is NOT a money-printing machine**, but it's an excellent framework for:
- Learning algorithmic trading
- Testing strategies systematically
- Automating trading decisions
- Managing risk properly

**Key Takeaway**: Even with 56% accuracy, profitable trading is possible with:
- Good risk management (stop losses)
- Proper position sizing
- Low fees
- Discipline (no emotional trading)

Our ensemble model (56.5% accuracy) is **comparable to Freqtrade's typical results**. The next step is implementing proper backtesting and risk management, not chasing higher accuracy.

## Resources

- **Freqtrade Docs**: https://www.freqtrade.io
- **GitHub**: https://github.com/freqtrade/freqtrade
- **Discord**: Active community for support
- **Strategy Repo**: https://github.com/freqtrade/freqtrade-strategies

---

**Bottom Line**: Focus on risk management and backtesting, not just model accuracy. A 55% accurate model with good risk management beats a 70% accurate model with poor risk management.


---

## Deep Dive: How Freqtrade Actually Works

### Core Trading Loop

The bot runs in a continuous loop (default: every 5 seconds):

```python
# Simplified version of the main loop
while True:
    # 1. Refresh market data
    exchange.reload_markets()
    
    # 2. Get active trading pairs
    active_pairs = pairlist.refresh()
    
    # 3. Download latest candles
    dataprovider.refresh(active_pairs)
    
    # 4. Run strategy analysis
    strategy.analyze(active_pairs)
    
    # 5. Check exit conditions for open trades
    for trade in open_trades:
        if should_exit(trade):
            place_exit_order(trade)
    
    # 6. Check entry conditions for new trades
    if can_open_new_trade():
        for pair in active_pairs:
            if strategy.has_entry_signal(pair):
                place_entry_order(pair)
    
    # 7. Update funding fees (futures only)
    update_funding_fees()
    
    # 8. Sleep for throttle period
    sleep(5)
```

### Strategy Signal Generation

**Step 1: Populate Indicators**
```python
def populate_indicators(self, dataframe, metadata):
    # Calculate all technical indicators
    dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
    dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
    dataframe['macd'] = ta.MACD(dataframe)
    
    # Bollinger Bands
    bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
    dataframe['bb_lower'] = bollinger['lower']
    dataframe['bb_middle'] = bollinger['mid']
    dataframe['bb_upper'] = bollinger['upper']
    
    return dataframe
```

**Step 2: Generate Entry Signals**
```python
def populate_entry_trend(self, dataframe, metadata):
    # Long entry conditions
    dataframe.loc[
        (
            # RSI oversold
            (dataframe['rsi'] < 30) &
            # Price above EMA
            (dataframe['close'] > dataframe['ema_fast']) &
            # MACD crossover
            (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
            # Volume confirmation
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean())
        ),
        'enter_long'  # Set signal
    ] = 1
    
    # Short entry conditions (if can_short = True)
    dataframe.loc[
        (
            (dataframe['rsi'] > 70) &
            (dataframe['close'] < dataframe['ema_fast']) &
            (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))
        ),
        'enter_short'
    ] = 1
    
    return dataframe
```

**Step 3: Generate Exit Signals**
```python
def populate_exit_trend(self, dataframe, metadata):
    # Long exit conditions
    dataframe.loc[
        (
            (dataframe['rsi'] > 70) |  # Overbought
            (dataframe['close'] < dataframe['bb_lower'])  # Below BB
        ),
        'exit_long'
    ] = 1
    
    # Short exit conditions
    dataframe.loc[
        (
            (dataframe['rsi'] < 30) |
            (dataframe['close'] > dataframe['bb_upper'])
        ),
        'exit_short'
    ] = 1
    
    return dataframe
```

### Order Execution Flow

**Entry Order:**
```
1. Strategy generates entry signal
2. Check if max_open_trades reached → NO: continue, YES: skip
3. Check if pair is locked → YES: skip, NO: continue
4. Calculate stake amount (position size)
5. Get entry price (from orderbook or last price)
6. Apply custom_entry_price() if defined
7. Call confirm_trade_entry() → FALSE: skip, TRUE: continue
8. Place order on exchange
9. Store order in database
10. Send notification (Telegram/WebUI)
```

**Exit Order:**
```
1. Check exit conditions:
   - ROI reached?
   - Stoploss hit?
   - Exit signal?
   - Custom exit?
   - Force exit?
2. Calculate exit price
3. Apply custom_exit_price() if defined
4. Call confirm_trade_exit() → FALSE: skip, TRUE: continue
5. Place exit order
6. Update trade in database
7. Calculate profit/loss
8. Send notification
```

### Position Adjustment (DCA)

If `position_adjustment_enable = True`:

```python
def adjust_trade_position(self, trade, current_time, current_rate, 
                         current_profit, min_stake, max_stake, **kwargs):
    # Example: Add to position if price drops 2%
    if current_profit < -0.02:
        # Calculate additional stake
        additional_stake = trade.stake_amount * 0.5  # Add 50% more
        
        # Return positive value to increase position
        return additional_stake
    
    # Example: Reduce position if profit > 5%
    if current_profit > 0.05:
        # Return negative value to decrease position
        return -trade.stake_amount * 0.3  # Sell 30%
    
    return None  # No adjustment
```

### Dynamic Stoploss

```python
def custom_stoploss(self, pair, trade, current_time, current_rate, 
                    current_profit, **kwargs):
    # Example: Trailing stoploss
    if current_profit > 0.05:  # If profit > 5%
        return -0.02  # Move stoploss to -2% (lock in profit)
    
    if current_profit > 0.10:  # If profit > 10%
        return -0.01  # Move stoploss to -1%
    
    # Example: Time-based stoploss
    trade_duration = (current_time - trade.open_date_utc).seconds / 3600
    if trade_duration > 24:  # After 24 hours
        return -0.05  # Tighter stoploss
    
    return self.stoploss  # Use default stoploss
```

### FreqAI Integration

**1. Define Features**
```python
def populate_any_indicators(self, pair, df, tf, informative=None, 
                           set_generalized_indicators=False):
    # Base indicators
    df['rsi'] = ta.RSI(df)
    df['ema_21'] = ta.EMA(df, 21)
    df['atr'] = ta.ATR(df)
    
    # FreqAI will automatically create:
    # - Historical shifts: rsi_shift_1, rsi_shift_2, ...
    # - Percentage changes: rsi_pct_change
    # - Rolling stats: rsi_rolling_mean_10, rsi_rolling_std_10
    # - Differences: rsi_diff
    
    return df
```

**2. Define Prediction Target**
```python
def set_freqai_targets(self, dataframe, **kwargs):
    # Predict future price movement
    dataframe['&-target'] = (
        dataframe['close'].shift(-5) / dataframe['close'] - 1
    ) * 100  # 5-candle future return (%)
    
    # Classification target (up/down)
    dataframe['&-target_class'] = (
        dataframe['&-target'] > 0
    ).astype(int)
    
    return dataframe
```

**3. Use Predictions in Strategy**
```python
def populate_entry_trend(self, dataframe, metadata):
    # Get FreqAI predictions
    dataframe.loc[
        (
            # Model predicts price increase
            (dataframe['&-prediction'] > 0.5) &
            # High confidence
            (dataframe['&-prediction'] > dataframe['&-prediction'].rolling(20).mean()) &
            # Confirm with RSI
            (dataframe['rsi'] < 40)
        ),
        'enter_long'
    ] = 1
    
    return dataframe
```

### Backtesting Process

**Command:**
```bash
freqtrade backtesting \
    --strategy MyStrategy \
    --timerange 20230101-20231231 \
    --timeframe 5m \
    --stake-amount 100
```

**What Happens:**
```
1. Load historical data from disk/database
2. Initialize strategy
3. For each candle in timerange:
   a. Update dataframe with new candle
   b. Calculate indicators
   c. Check entry signals
   d. Check exit signals for open trades
   e. Simulate order execution (with fees)
   f. Update trade records
4. Calculate performance metrics:
   - Total profit/loss
   - Win rate
   - Sharpe ratio
   - Max drawdown
   - Average trade duration
5. Generate report and charts
```

**Realistic Simulation:**
- Applies exchange fees (0.1% typical)
- Simulates slippage (price moves during execution)
- Respects order types (market vs limit)
- Handles partial fills
- Accounts for minimum order sizes

### Hyperparameter Optimization

**Define Optimizable Parameters:**
```python
class MyStrategy(IStrategy):
    # Hyperopt parameters
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    sell_rsi = IntParameter(60, 80, default=70, space='sell')
    roi_p1 = DecimalParameter(0.01, 0.05, default=0.02, space='roi')
    stoploss_value = DecimalParameter(-0.15, -0.05, default=-0.10, space='stoploss')
```

**Run Optimization:**
```bash
freqtrade hyperopt \
    --strategy MyStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --epochs 1000 \
    --spaces buy sell roi stoploss
```

**Process:**
```
1. Generate random parameter combinations
2. For each combination:
   a. Run backtest with those parameters
   b. Calculate loss function (e.g., negative Sharpe ratio)
3. Use optimization algorithm (Optuna) to find better parameters
4. After N epochs, return best parameters
5. Update strategy with optimal values
```

### Exchange Integration (CCXT)

**Supported Operations:**
```python
# Fetch OHLCV data
candles = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=100)

# Get orderbook
orderbook = exchange.fetch_order_book('BTC/USDT')

# Place market order
order = exchange.create_market_buy_order('BTC/USDT', amount=0.01)

# Place limit order
order = exchange.create_limit_sell_order('BTC/USDT', amount=0.01, price=50000)

# Place stop-loss order
order = exchange.create_order(
    symbol='BTC/USDT',
    type='stop_loss_limit',
    side='sell',
    amount=0.01,
    price=49000,
    params={'stopPrice': 49500}
)

# Get balance
balance = exchange.fetch_balance()

# Cancel order
exchange.cancel_order(order_id, 'BTC/USDT')

# Fetch open orders
open_orders = exchange.fetch_open_orders('BTC/USDT')
```

### Database Schema

**Trades Table:**
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    exchange TEXT,
    pair TEXT,
    is_open BOOLEAN,
    fee_open REAL,
    fee_close REAL,
    open_rate REAL,
    close_rate REAL,
    amount REAL,
    stake_amount REAL,
    open_date TIMESTAMP,
    close_date TIMESTAMP,
    stop_loss REAL,
    initial_stop_loss REAL,
    max_rate REAL,
    min_rate REAL,
    strategy TEXT,
    timeframe TEXT,
    leverage REAL,
    is_short BOOLEAN
);
```

**Orders Table:**
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    ft_trade_id INTEGER,
    order_id TEXT,
    order_type TEXT,
    status TEXT,
    side TEXT,
    price REAL,
    amount REAL,
    filled REAL,
    remaining REAL,
    cost REAL,
    order_date TIMESTAMP,
    FOREIGN KEY(ft_trade_id) REFERENCES trades(id)
);
```

### Risk Management Features

**1. Stoploss Types:**
- Fixed: `-0.10` (10% loss)
- Trailing: Follows price up, locks in profit
- Dynamic: Changes based on conditions
- On-exchange: Stoploss order on exchange (faster execution)

**2. ROI Table:**
```python
minimal_roi = {
    "0": 0.10,    # 10% profit anytime
    "30": 0.05,   # 5% after 30 minutes
    "60": 0.02,   # 2% after 60 minutes
    "120": 0.01   # 1% after 120 minutes
}
```

**3. Position Sizing:**
```python
# Fixed amount
stake_amount = 100  # $100 per trade

# Percentage of wallet
stake_amount = "unlimited"  # Use all available
tradable_balance_ratio = 0.99  # Use 99% of balance

# Dynamic sizing
def custom_stake_amount(self, pair, current_time, current_rate, 
                       proposed_stake, min_stake, max_stake, **kwargs):
    # Risk 2% of capital per trade
    capital = self.wallets.get_total_stake_amount()
    risk_per_trade = capital * 0.02
    
    # Calculate position size based on stoploss
    stoploss_distance = abs(self.stoploss)
    position_size = risk_per_trade / stoploss_distance
    
    return min(position_size, max_stake)
```

**4. Protections:**
```python
protections = [
    {
        "method": "StoplossGuard",
        "lookback_period_candles": 60,
        "trade_limit": 4,
        "stop_duration_candles": 30,
        "required_profit": 0.0
    },
    {
        "method": "MaxDrawdown",
        "lookback_period_candles": 200,
        "trade_limit": 20,
        "stop_duration_candles": 50,
        "max_allowed_drawdown": 0.2
    },
    {
        "method": "LowProfitPairs",
        "lookback_period_candles": 360,
        "trade_limit": 1,
        "stop_duration_candles": 120,
        "required_profit": -0.05
    }
]
```

### Telegram Bot Commands

**Trading Control:**
- `/start` - Start the bot
- `/stop` - Stop the bot (keeps open trades)
- `/stopentry` - Stop opening new trades
- `/reload_config` - Reload configuration

**Trade Management:**
- `/status` - Show open trades
- `/status table` - Show trades in table format
- `/profit` - Show profit summary
- `/profit 7` - Show profit for last 7 days
- `/forceexit <trade_id>` - Force close a trade
- `/forceexit all` - Close all trades
- `/forceenter BTC/USDT` - Force open a trade

**Information:**
- `/balance` - Show wallet balance
- `/daily` - Daily profit/loss
- `/weekly` - Weekly profit/loss
- `/monthly` - Monthly profit/loss
- `/performance` - Performance by pair
- `/count` - Number of trades
- `/locks` - Show locked pairs

**Configuration:**
- `/show_config` - Display current config
- `/whitelist` - Show active pairs
- `/blacklist` - Show blacklisted pairs
- `/edge` - Show edge positions (if enabled)

### Web UI Features

**Dashboard:**
- Real-time profit/loss
- Open trades table
- Trade history
- Performance charts
- Balance overview

**Trade Management:**
- Force entry/exit
- Adjust stoploss
- View trade details
- Cancel orders

**Strategy:**
- View strategy code
- See indicator values
- Plot candlestick charts
- Backtest results

**Configuration:**
- Edit config.json
- Manage pairlists
- Set protections
- Adjust risk parameters

### Performance Metrics

**Calculated Metrics:**
```python
# Win rate
win_rate = winning_trades / total_trades

# Profit factor
profit_factor = gross_profit / gross_loss

# Sharpe ratio
sharpe = (mean_return - risk_free_rate) / std_return

# Sortino ratio (only downside volatility)
sortino = (mean_return - risk_free_rate) / downside_std

# Calmar ratio
calmar = annual_return / max_drawdown

# Max drawdown
max_drawdown = (peak_value - trough_value) / peak_value

# Average trade duration
avg_duration = sum(trade_durations) / total_trades

# Expectancy
expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
```

### Common Strategy Patterns

**1. Trend Following:**
```python
# Enter on EMA crossover
dataframe.loc[
    (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])) &
    (dataframe['volume'] > dataframe['volume'].rolling(20).mean()),
    'enter_long'
] = 1
```

**2. Mean Reversion:**
```python
# Enter when price touches lower Bollinger Band
dataframe.loc[
    (dataframe['close'] < dataframe['bb_lower']) &
    (dataframe['rsi'] < 30),
    'enter_long'
] = 1
```

**3. Breakout:**
```python
# Enter on volume breakout above resistance
dataframe.loc[
    (dataframe['close'] > dataframe['high'].rolling(20).max()) &
    (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2),
    'enter_long'
] = 1
```

**4. Momentum:**
```python
# Enter on strong momentum
dataframe.loc[
    (dataframe['rsi'] > 60) &
    (dataframe['macd'] > dataframe['macdsignal']) &
    (dataframe['close'] > dataframe['ema_21']),
    'enter_long'
] = 1
```

---

## Summary: What Makes Freqtrade Powerful

1. **Modular Architecture** - Easy to extend and customize
2. **Battle-Tested** - Used by thousands of traders
3. **Comprehensive Backtesting** - Realistic simulation
4. **Risk Management** - Multiple layers of protection
5. **ML Integration** - FreqAI for adaptive strategies
6. **Multi-Exchange** - Works with major exchanges
7. **Active Community** - Constant improvements
8. **Open Source** - Transparent and auditable

**Key Insight:** Freqtrade's success isn't from having the "best" strategy, but from providing a robust framework for systematic trading with proper risk management.
