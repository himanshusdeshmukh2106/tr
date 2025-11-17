# DETAILED PROBLEM STATEMENT FOR AI RESEARCH

## OBJECTIVE
Build a profitable intraday trading model that predicts directional price movements (UP/DOWN/NEUTRAL) for Reliance stock using 5-minute OHLCV data with LSTM neural networks.

## CURRENT SITUATION

### Data Available
- **Asset**: Reliance Industries (Indian stock)
- **Timeframe**: 5-minute candles
- **Duration**: Full year of historical data
- **Features**: OHLCV (Open, High, Low, Close, Volume)
- **Total samples**: ~50,000 candles

### Problem Definition
We're trying to predict 3 classes:
- **Class 0 (DOWN)**: Price will drop by threshold% in next N candles → SHORT signal
- **Class 1 (NEUTRAL)**: Price will stay within ±threshold% → STAY OUT
- **Class 2 (UP)**: Price will rise by threshold% in next N candles → LONG signal

### Current Model Architecture
```
Input: 30 candles × 21 features
↓
Bidirectional LSTM (64 units) + Dropout (0.3)
↓
Bidirectional LSTM (32 units) + Dropout (0.3)
↓
Dense (64, relu) + Dropout (0.2)
↓
Output: Softmax (3 classes)
```

### Features Used (21 total)
**Candle Patterns:**
- Body size, upper wick, lower wick, bullish/bearish flag

**Price Action:**
- Returns (1, 2, 5 periods), high-low ratio, close-open ratio

**Trend Indicators:**
- SMA (5, 20), EMA (21), price-to-MA ratios, ADX

**Volume:**
- Volume change, volume ratio vs MA

**Support/Resistance:**
- Distance to 20-period high/low

**Time:**
- Hour (sin/cos encoded)

## THE CORE PROBLEM

### What We've Tried
1. **Binary classification** (UP vs DOWN/NEUTRAL)
   - Result: 71% overall, but only 38% on UP class
   - Model biased toward predicting DOWN/NEUTRAL

2. **3-class with SMOTE balancing**
   - Result: Model predicts NEUTRAL 97% of the time
   - DOWN accuracy: 5.6%, UP accuracy: 28%
   - Essentially learned "don't trade"

3. **Aggressive class weighting** (5x penalty for minority classes)
   - Result: Swung too far, predicted UP for everything
   - Overall accuracy dropped to 34%

4. **Moderate class weighting** (2.5x)
   - Result: Back to predicting mostly NEUTRAL
   - DOWN: 74%, UP: 36%, NEUTRAL: 97%

5. **Increased threshold** (0.15% → 0.3%)
   - Result: Even more NEUTRAL predictions (92.8%)
   - Fewer samples to learn from

### Current Best Result
- Overall accuracy: 83%
- But 87% of samples are NEUTRAL
- Only 15 DOWN and 39 UP predictions at high confidence (>0.6)
- Model essentially says "can't predict 5-min moves reliably"

## CONSTRAINTS & REQUIREMENTS

### Must Have
1. **Tradeable signals**: Need at least 50-100 UP and DOWN signals per year
2. **Accuracy target**: >60% on both UP and DOWN classes
3. **Use 5-minute data**: Cannot change to daily/hourly (requirement)
4. **Real-time capable**: Model must run fast enough for live trading
5. **Indian market hours**: 9:15 AM - 3:30 PM IST (6.25 hours)

### Nice to Have
1. Confidence scores for risk management
2. Separate accuracy for different market conditions (trending vs ranging)
3. Explainability of predictions

## WHAT WE NEED FROM AI RESEARCH

### Primary Question
**How can we improve UP and DOWN class accuracy from 5-30% to >60% while maintaining reasonable signal frequency (not just predicting NEUTRAL)?**

### Specific Research Areas

#### 1. Architecture Improvements
- Should we use Transformer instead of LSTM?
- Attention mechanisms?
- CNN-LSTM hybrid?
- Temporal Convolutional Networks (TCN)?
- Multi-scale architecture (different timeframes)?

#### 2. Feature Engineering
- Are we missing critical features?
- Should we add:
  - Order imbalance indicators?
  - Microstructure features?
  - Market regime detection?
  - Volatility clustering?
  - Fractal indicators?
  - Market breadth indicators?
- Should we remove some features (curse of dimensionality)?

#### 3. Data Preprocessing
- Different normalization techniques?
- Denoising methods (wavelets, Kalman filter)?
- Feature selection algorithms?
- Dimensionality reduction (PCA, autoencoders)?

#### 4. Training Strategies
- Focal loss instead of categorical cross-entropy?
- Curriculum learning (easy examples first)?
- Meta-learning approaches?
- Ensemble methods (multiple models)?
- Transfer learning from other stocks?

#### 5. Class Imbalance Solutions
- Better than SMOTE? (ADASYN, BorderlineSMOTE, SMOTE-ENN)?
- Cost-sensitive learning?
- One-vs-rest classifiers?
- Anomaly detection approach (treat UP/DOWN as anomalies)?

#### 6. Problem Reformulation
- Should we predict magnitude instead of direction?
- Regression then threshold?
- Predict probability distribution of returns?
- Multi-task learning (predict multiple horizons)?
- Sequence-to-sequence (predict next N candles)?

#### 7. Market-Specific Considerations
- Indian market has different characteristics than US markets
- Opening range breakouts important?
- Lunch time (12:30-1:30) low volume period?
- Last 30 minutes high volatility?
- Should we train separate models for different times of day?

#### 8. Validation Strategy
- Walk-forward optimization?
- Purged K-fold cross-validation?
- How to avoid look-ahead bias?
- How to test for overfitting in time series?

## RESEARCH QUESTIONS FOR AI

1. **What does recent academic research (2023-2024) say about predicting intraday stock movements with deep learning?**

2. **What are the state-of-the-art architectures for financial time series classification?**

3. **How do successful quant firms approach this problem? (Based on public papers/talks)**

4. **Are there specific techniques for handling extreme class imbalance in time series?**

5. **What features have been proven to work for 5-minute stock prediction?**

6. **Should we be using reinforcement learning instead of supervised learning?**

7. **What are the theoretical limits of predictability at 5-minute timeframe?**

8. **Are there papers specifically on Indian stock market prediction?**

9. **What data augmentation techniques work for financial time series?**

10. **How to detect if the problem is fundamentally unpredictable vs poorly modeled?**

## SUCCESS CRITERIA

A solution is successful if it achieves:
1. **UP class accuracy**: >60% (currently 28-38%)
2. **DOWN class accuracy**: >60% (currently 5-30%)
3. **Signal frequency**: >50 trades per class per year (currently 15-39)
4. **High confidence accuracy**: >80% when confidence >0.7
5. **Backtested profitability**: Positive Sharpe ratio >1.0

## ADDITIONAL CONTEXT

### Why This Matters
- 5-minute predictions allow for:
  - Multiple trades per day
  - Quick profit taking
  - Tight stop losses
  - Scalping strategies
  
### Why We Can't Just Use Longer Timeframes
- Requirement is specifically for intraday trading
- Need to enter and exit same day
- 5-minute allows for 75 candles per day
- Longer timeframes (15-min, 30-min) reduce opportunities

### What Makes This Hard
- High noise-to-signal ratio at 5-minute level
- Market microstructure effects dominate
- Bid-ask spread significant relative to moves
- Random walk hypothesis suggests it might be impossible
- But: Some traders do make money at this timeframe, so patterns must exist

## REQUEST TO AI

Please provide:
1. **Literature review**: Recent papers (2020-2024) on this exact problem
2. **Architecture recommendations**: Specific model architectures to try
3. **Feature suggestions**: What features are we missing?
4. **Training techniques**: Better ways to handle class imbalance
5. **Alternative approaches**: Completely different ways to frame the problem
6. **Reality check**: Is this problem solvable with current ML techniques?
7. **Code examples**: Links to GitHub repos solving similar problems
8. **Practical advice**: What would you try next if you were in our shoes?

## CURRENT CODE REPOSITORY
- GitHub: himanshusdeshmukh2106/tr
- Main training script: `AlgoTrading/LSTM_strategy/train_3class.py`
- Optimization script: `AlgoTrading/LSTM_strategy/optimize_3class.py`
- Data: `reliance_data_5min_full_year.csv`

---

**Bottom Line**: We need to predict 5-minute stock price direction with >60% accuracy on both UP and DOWN moves, but our model keeps predicting NEUTRAL for everything. What should we try differently?
