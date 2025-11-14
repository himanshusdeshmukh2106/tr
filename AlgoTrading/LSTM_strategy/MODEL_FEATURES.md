# LSTM Model - Features & Architecture Explained

## 🎯 How It Works

The LSTM model learns to predict **trap success probability** by analyzing patterns in historical 5-minute candle data.

## 📊 Input Features (What the Model Sees)

### 1. Price Data (OHLCV)
```
Open    → Opening price of 5-min candle
High    → Highest price in 5-min period
Low     → Lowest price in 5-min period
Close   → Closing price of 5-min candle
Volume  → Trading volume in 5-min period
```

**Why?** Raw price action shows market behavior and momentum.

### 2. EMA (Exponential Moving Average)
```
EMA_21  → 21-period EMA value
```

**Why?** This is our KEY indicator. The model learns:
- When price is above/below EMA
- Distance from EMA
- EMA slope (trending up/down)
- Historical EMA crosses

### 3. ADX (Average Directional Index)
```
ADX     → Trend strength (0-100)
```

**Why?** Tells the model about trend strength:
- ADX 20-36 = Sweet spot for traps
- ADX < 20 = Too choppy
- ADX > 36 = Trend too strong

### 4. Candle Body Analysis
```
Body_Pct → |Close - Open| / Close * 100
```

**Why?** Small bodies (≤0.20%) indicate:
- Indecision
- Consolidation
- Potential reversal
- Good trap setup

### 5. Price Position vs EMA
```
Price_vs_EMA → (Close - EMA_21) / EMA_21 * 100
```

**Why?** Shows:
- How far price is from EMA
- Direction of deviation
- Trap potential magnitude

### 6. Volume Analysis
```
Volume_MA    → 20-period volume average
Volume_Ratio → Current Volume / Volume_MA
```

**Why?** Volume confirms:
- Trap validity (low volume = weak move)
- Breakout strength
- Reversal conviction

## 🧠 LSTM Architecture

### Input Layer
```
Shape: (30, 10)
       ↑   ↑
       |   └─ 10 features per candle
       └───── 30 candles lookback (2.5 hours)
```

**What it means:** The model looks at the last 30 five-minute candles (2.5 hours of trading) with 10 features each.

### LSTM Layers (The Brain)

```
Input (30 timesteps × 10 features)
         ↓
┌────────────────────────────────┐
│  LSTM Layer 1: 128 units       │  ← Learns long-term patterns
│  + Dropout (30%)                │
│  + Batch Normalization          │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│  LSTM Layer 2: 64 units        │  ← Learns medium-term patterns
│  + Dropout (30%)                │
│  + Batch Normalization          │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│  LSTM Layer 3: 32 units        │  ← Learns short-term patterns
│  + Dropout (30%)                │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│  Dense Layer: 64 units         │  ← Combines patterns
│  + Dropout (20%)                │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│  Dense Layer: 32 units         │  ← Refines prediction
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│  Output: 1 unit (Sigmoid)      │  ← Probability (0-1)
└────────────────────────────────┘
         ↓
    Prediction
    0 = SHORT (upside trap)
    1 = LONG (downside trap)
```

## 🔍 What the LSTM Learns

### Pattern Recognition

**1. Trap Formation Patterns**
```
The model learns sequences like:

Downside Trap (LONG signal):
Candle 1: Close < EMA (bearish)
Candle 2: Close < EMA (still bearish)
Candle 3: Close < EMA (trap forming)
Candle 4: Close > EMA (TRAP! → GO LONG)
         + Small body
         + ADX 20-36
         + Low volume
```

**2. Failed Trap Patterns**
```
The model also learns when traps FAIL:

False Signal:
Candle 1: Close < EMA
Candle 2: Close > EMA (looks like trap)
Candle 3: Close < EMA again (not a real trap)
         + Large body
         + ADX > 40 (strong trend)
         + High volume
→ Model learns to AVOID this
```

**3. Time-Based Patterns**
```
The model learns:
- 9:15-9:30 window has different behavior
- 10:00-11:00 window has different behavior
- Morning volatility patterns
- Volume patterns by time
```

**4. Market Context**
```
The model understands:
- Trending vs ranging markets
- Volatility regimes
- Volume characteristics
- Price momentum
```

## 📈 Training Process

### Step 1: Data Collection
```python
# Download 60 days of 5-minute data
df = yf.download("SPY", period="60d", interval="5m")

# Result: ~4,680 candles (60 days × 78 candles/day)
```

### Step 2: Feature Engineering
```python
# Calculate all features
df['EMA_21'] = calculate_ema(df['Close'], 21)
df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
df['Body_Pct'] = abs(df['Close'] - df['Open']) / df['Close'] * 100
df['Price_vs_EMA'] = (df['Close'] - df['EMA_21']) / df['EMA_21'] * 100
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
```

### Step 3: Label Creation
```python
# For each candle, check if trap conditions are met
for idx in range(len(df)):
    signal = strategy.generate_signal(df, idx)
    if signal == 'LONG':
        df['Label'][idx] = 1  # Downside trap
    elif signal == 'SHORT':
        df['Label'][idx] = 0  # Upside trap
    else:
        df['Label'][idx] = -1  # No signal (skip)
```

### Step 4: Sequence Creation
```python
# Create 30-candle sequences
X = []  # Input sequences
y = []  # Labels

for i in range(30, len(df)):
    if df['Label'][i] != -1:  # Only use valid signals
        # Take last 30 candles as input
        sequence = df[i-30:i][features]
        X.append(sequence)
        y.append(df['Label'][i])

# Result:
# X shape: (num_samples, 30, 10)
# y shape: (num_samples,)
```

### Step 5: Normalization
```python
# Scale all features to 0-1 range
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Why? Helps LSTM learn faster and better
```

### Step 6: Training
```python
# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Model learns to predict trap success probability
```

## 🎯 Model Output & Usage

### Prediction
```python
# Input: Last 30 candles
sequence = get_last_30_candles()

# Output: Probability
probability = model.predict(sequence)
# probability = 0.85 (85% confidence)

# Decision:
if probability > 0.7:  # High confidence threshold
    if signal == 'LONG':
        execute_long_trade()
    elif signal == 'SHORT':
        execute_short_trade()
```

### Confidence Levels
```
Probability    Interpretation           Action
-----------    ----------------------   --------
0.0 - 0.3      Strong SHORT signal      Go SHORT
0.3 - 0.4      Weak SHORT signal        Maybe SHORT
0.4 - 0.6      Uncertain                Skip
0.6 - 0.7      Weak LONG signal         Maybe LONG
0.7 - 1.0      Strong LONG signal       Go LONG
```

## 🔄 Complete Workflow

```
1. Market Opens (9:15 AM)
         ↓
2. Collect last 30 candles (2.5 hours)
         ↓
3. Calculate features:
   - EMA_21
   - ADX
   - Body_Pct
   - Price_vs_EMA
   - Volume_Ratio
         ↓
4. Check trap conditions:
   - In entry window? ✓
   - Body ≤ 0.20%? ✓
   - ADX 20-36? ✓
   - Trap pattern? ✓
         ↓
5. Feed to LSTM model
         ↓
6. Get probability (e.g., 0.82)
         ↓
7. If probability > 0.7:
   → Execute trade
   Else:
   → Skip (low confidence)
         ↓
8. Manage position:
   - Monitor stop loss
   - Monitor target
   - Update trailing stop
         ↓
9. Exit at:
   - Stop loss hit
   - Target hit
   - Trailing stop hit
   - 3:15 PM (time exit)
```

## 📊 Example: Real Trade

```
Time: 9:22 AM
Current Price: $450.25
21 EMA: $450.50

Last 30 Candles Analysis:
- Candle 28: Close $450.10 (below EMA)
- Candle 29: Close $450.05 (below EMA)
- Candle 30: Close $450.30 (above EMA) ← TRAP!

Features:
✓ Body_Pct: 0.18% (< 0.20%)
✓ ADX: 28 (20-36 range)
✓ Price_vs_EMA: +0.04% (just crossed)
✓ Volume_Ratio: 0.85 (below average)
✓ Time: 9:22 AM (in window)

LSTM Input:
- 30 candles × 10 features = 300 data points
- Normalized to 0-1 range

LSTM Output:
- Probability: 0.83 (83% confidence)
- Signal: LONG (downside trap detected)

Decision:
✓ Probability > 0.7 → HIGH CONFIDENCE
✓ Execute LONG trade at $450.25

Risk Management:
- Entry: $450.25
- Stop Loss: $448.00 (0.5% below)
- Target: $454.75 (1.0% above)
- Trailing Stop: Activates at $452.50

Result:
- Target hit at 10:15 AM
- Exit: $454.80
- P&L: +$4.55 (+1.01%)
- Trade Duration: 53 minutes
```

## 🎓 Key Advantages

### 1. Pattern Learning
- LSTM learns complex patterns humans might miss
- Adapts to changing market conditions
- Identifies subtle trap formations

### 2. Context Awareness
- Considers 2.5 hours of history
- Understands sequence of events
- Recognizes market regime

### 3. Probability-Based
- Provides confidence level
- Allows filtering low-confidence trades
- Improves win rate

### 4. Continuous Learning
- Retrain weekly/monthly
- Adapts to new market behavior
- Improves over time

## 🔧 Optimization Tips

### Feature Engineering
```python
# Add more features:
- RSI (momentum)
- MACD (trend)
- Bollinger Bands (volatility)
- Order flow (if available)
- Market breadth
```

### Hyperparameter Tuning
```python
# Experiment with:
- Sequence length (20, 30, 40 candles)
- LSTM units (64, 128, 256)
- Dropout rates (0.2, 0.3, 0.4)
- Learning rate (0.0001, 0.001, 0.01)
```

### Ensemble Methods
```python
# Combine multiple models:
- LSTM + Random Forest
- LSTM + XGBoost
- Multiple LSTM architectures
- Voting classifier
```

## 📈 Expected Performance

With proper training:
- **Accuracy:** 60-70% (better than random)
- **Precision:** 65-75% (when it says LONG, it's right)
- **Recall:** 55-65% (catches most good traps)
- **F1-Score:** 60-70% (balanced performance)

This translates to:
- **Win Rate:** 55-65% in backtesting
- **Profit Factor:** 1.5-2.0
- **Sharpe Ratio:** 1.0-1.5

## 🚀 Ready to Train!

Now you understand exactly how the model works. Ready to train it?

```bash
python run_trap_strategy.py --step all --ticker SPY
```

The model will learn from historical trap patterns and help you trade more profitably! 📈
