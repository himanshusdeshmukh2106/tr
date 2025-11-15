# Hyperparameter Optimization Guide

## 🎯 Goal
Improve Class 1 (Up) accuracy from **34-42%** to **50%+** while maintaining overall performance.

## 📊 Current Performance
```
Overall: 63-68%
Class 0 (Down): 71-79%
Class 1 (Up): 34-42% ❌ TOO LOW
High Confidence (>0.7): 75-85%
```

## 🚀 Two Optimization Approaches

### Option 1: Quick Optimization (Recommended)
**File:** `quick_optimization.py`  
**Time:** 2-3 hours  
**Configs:** 20 strategic combinations

```bash
cd AlgoTrading/LSTM_strategy
python quick_optimization.py
```

**What it tests:**
- ✅ Threshold variations (0.10% - 0.20%)
- ✅ Horizon variations (8-12 candles)
- ✅ Model sizes (small, medium, large)
- ✅ Dropout rates (0.2 - 0.4)
- ✅ Learning rates (0.0005 - 0.002)
- ✅ Alternative architectures (LSTM, GRU, Deep LSTM)
- ✅ Strategic combinations

**Best for:** Quick results, testing key parameters

---

### Option 2: Comprehensive Search
**File:** `hyperparameter_search.py`  
**Time:** 8-12 hours  
**Configs:** 500+ combinations

```bash
cd AlgoTrading/LSTM_strategy
python hyperparameter_search.py
```

**What it tests:**
- Full grid search over all parameters
- Every combination of:
  - Horizons: 8, 10, 12
  - Thresholds: 0.0012, 0.0015, 0.0018, 0.0020
  - Sequence lengths: 25, 30, 35
  - LSTM units: [64,32], [96,48], [80,40]
  - Dropout: 0.25, 0.30, 0.35
  - Learning rates: 0.0008, 0.001, 0.0015
  - Batch sizes: 32, 64

**Best for:** Finding absolute best configuration, overnight runs

---

## 📈 Expected Results

### Quick Optimization
```
Expected Class 1: 45-52%
Expected Overall: 66-70%
Improvement: +7-14% on Class 1
```

### Comprehensive Search
```
Expected Class 1: 50-58%
Expected Overall: 68-72%
Improvement: +12-20% on Class 1
```

---

## 🔍 How It Works

### 1. Data Preparation
- Loads Reliance 5-min data
- Creates technical indicators
- Generates target based on horizon/threshold
- Checks class balance (skips if too imbalanced)

### 2. Model Training
- Applies SMOTE to balance classes
- Trains with early stopping (prevents overfitting)
- Uses validation set for evaluation

### 3. Evaluation Metrics
- **Overall Accuracy**: General performance
- **Class 0 Accuracy**: Down/Neutral predictions
- **Class 1 Accuracy**: Up predictions ⭐ (our focus)
- **High Confidence Accuracy**: Predictions with >70% confidence
- **Weighted Score**: 0.25×Overall + 0.45×Class1 + 0.30×HighConf

### 4. Results
- Saves progress incrementally (can resume if interrupted)
- Generates JSON and CSV reports
- Shows top 5 configurations by different metrics

---

## 📊 Understanding Results

### Results Files
```
quick_optimization_results/
├── progress.json          # Incremental saves
├── results.json          # Final results with best config
└── results.csv           # All results for analysis

hyperparameter_results/
├── progress.json
├── final_results.json
└── all_results.csv
```

### Key Metrics to Watch

**Class 1 Accuracy** (Most Important)
```
< 40%: Poor (current state)
40-50%: Acceptable
50-60%: Good ⭐
> 60%: Excellent
```

**Weighted Score**
```
< 0.500: Poor
0.500-0.600: Acceptable
0.600-0.700: Good ⭐
> 0.700: Excellent
```

**High Confidence Accuracy**
```
< 75%: Not reliable
75-85%: Good
> 85%: Excellent ⭐
```

---

## 🎯 What to Do After Optimization

### 1. Check Results
```bash
# View best configuration
cat quick_optimization_results/results.json | grep -A 20 "best"

# Or open in Python
import json
with open('quick_optimization_results/results.json') as f:
    results = json.load(f)
    best = results['best']
    print(f"Class 1: {best['class1']:.1%}")
    print(f"Parameters: {best}")
```

### 2. Update Training Script
Copy the best parameters to `train_reliance_balanced.py`:

```python
# Update these lines
df = prep.create_binary_target(
    df, 
    horizon=10,        # Use best horizon
    threshold=0.0015   # Use best threshold
)

X, y, features = prep.prepare_sequences(
    df, 
    sequence_length=30  # Use best seq_len
)

# Update model
model = build_simple_lstm(X_train.shape[1:])  # Adjust if needed

# Update training
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Use best lr
    ...
)

history = model.fit(
    ...,
    batch_size=64,  # Use best batch_size
    ...
)
```

### 3. Retrain Final Model
```bash
python train_reliance_balanced.py
```

### 4. Verify Improvement
Check that Class 1 accuracy improved:
```
Before: 34-42%
After: 50%+ ⭐
```

---

## 💡 Tips & Tricks

### Running Overnight
Both scripts save progress incrementally, so you can:
1. Start the script
2. Let it run overnight
3. Check results in the morning
4. If interrupted, results are still saved

### Monitoring Progress
```bash
# Watch progress in real-time
tail -f quick_optimization_results/progress.json

# Count completed configs
cat quick_optimization_results/progress.json | grep '"id"' | wc -l
```

### If It Fails
- Check data file path is correct
- Ensure you have enough RAM (8GB+ recommended)
- Try quick_optimization.py first (less memory intensive)
- Reduce number of configs if needed

### Interpreting Class Imbalance Warnings
```
"⚠️ Skipped: Class imbalance"
```
This means the threshold is too high/low for that horizon. The script automatically skips these to save time.

---

## 🔬 Why This Approach Works

### Problem Analysis
**Current Issue:**
- Threshold 0.2% is too strict → Only 25% Class 1 samples
- SMOTE can't create information from nothing
- Model can't learn patterns from limited real data

**Solution:**
- Test multiple thresholds (0.12% - 0.20%)
- Find sweet spot with 30-35% Class 1 samples
- More real samples → Better SMOTE augmentation → Better learning

### Key Insights
1. **More data ≠ Better results**
   - 0.1% threshold: Too noisy (42% accuracy)
   - 0.2% threshold: Too rare (34% accuracy)
   - 0.15% threshold: Just right (expected 50%+)

2. **Horizon matters**
   - Shorter (5-8): More noise, harder to predict
   - Medium (10-12): Clear trends, easier to predict ⭐
   - Longer (15+): Too far ahead, unpredictable

3. **Model size balance**
   - Too small: Can't learn complex patterns
   - Too large: Overfits to training data
   - Just right: Generalizes well ⭐

---

## 📈 Expected Improvements

### Scenario 1: Conservative (Quick Optimization)
```
Class 1: 34% → 48% (+14%)
Overall: 68% → 69% (+1%)
High Conf: 85% → 83% (-2%)

Trade-off: Slightly lower high-conf accuracy
Benefit: Much better Class 1 predictions
```

### Scenario 2: Optimal (Comprehensive Search)
```
Class 1: 34% → 55% (+21%)
Overall: 68% → 71% (+3%)
High Conf: 85% → 86% (+1%)

Trade-off: None!
Benefit: Better across all metrics ⭐
```

### Scenario 3: Aggressive
```
Class 1: 34% → 62% (+28%)
Overall: 68% → 68% (0%)
High Conf: 85% → 78% (-7%)

Trade-off: Lower overall and high-conf
Benefit: Excellent Class 1 predictions
Use case: When you specifically need Up signals
```

---

## 🎓 Learning from Results

### Analyze Patterns
After optimization, look for patterns in `results.csv`:

```python
import pandas as pd
df = pd.read_csv('quick_optimization_results/results.csv')

# What thresholds work best?
print(df.groupby('threshold')['class1'].mean())

# What horizons work best?
print(df.groupby('horizon')['class1'].mean())

# Correlation analysis
print(df[['threshold', 'horizon', 'class1', 'score']].corr())
```

### Common Findings
- **Lower thresholds** (0.12-0.14%): More signals, lower accuracy
- **Medium thresholds** (0.15-0.16%): Balanced ⭐
- **Higher thresholds** (0.18-0.20%): Fewer signals, higher accuracy

- **Shorter horizons** (8-9): Noisy but frequent
- **Medium horizons** (10-11): Best balance ⭐
- **Longer horizons** (12+): Clear but rare

---

## ✅ Success Criteria

### Minimum Acceptable
```
Class 1: > 45%
Overall: > 65%
High Conf: > 75%
```

### Good Performance
```
Class 1: > 50% ⭐
Overall: > 68%
High Conf: > 80%
```

### Excellent Performance
```
Class 1: > 55%
Overall: > 70%
High Conf: > 85%
```

---

## 🚀 Quick Start

### For Impatient Users
```bash
# Just run this and come back in 3 hours
cd AlgoTrading/LSTM_strategy
python quick_optimization.py

# Check results
cat quick_optimization_results/results.json | grep "class1"

# If Class 1 > 50%, you're done! 🎉
```

### For Perfectionists
```bash
# Run comprehensive search overnight
cd AlgoTrading/LSTM_strategy
python hyperparameter_search.py

# Check in the morning
cat hyperparameter_results/final_results.json | grep "class1"

# Find the absolute best configuration
```

---

## 📝 Summary

| Script | Time | Configs | Best For |
|--------|------|---------|----------|
| quick_optimization.py | 2-3h | 20 | Quick results |
| hyperparameter_search.py | 8-12h | 500+ | Best possible |

**Recommendation:** Start with `quick_optimization.py`. If results are good (Class 1 > 50%), use them. If not, run `hyperparameter_search.py` overnight.

**Expected outcome:** Class 1 accuracy improves from 34-42% to 50-58% 🎯

Good luck! 📈
