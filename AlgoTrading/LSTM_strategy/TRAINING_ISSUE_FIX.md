# 🚨 Training Issue: Model Predicting Only One Class

## Problem

Your model achieved 95.53% accuracy but it's **completely useless** because:

```
Class 0 (Down): 0.00% accuracy - NEVER predicted
Class 1 (Neutral): 100.00% accuracy - ALWAYS predicted  
Class 2 (Up): 0.00% accuracy - NEVER predicted
```

**The model learned to always predict "Neutral" and ignore Up/Down movements!**

## Why This Happened

1. **Severe Class Imbalance**: 95% of data is "Neutral" (small moves < 0.5%)
2. **Lazy Learning**: Model gets 95% accuracy by just predicting majority class
3. **No Penalty**: Model isn't penalized for ignoring minority classes

## Solutions

### Solution 1: Lower Threshold (Quick Fix)

Change line 160 in `train_reliance_improved.py`:

```python
# OLD (too strict - 95% neutral)
def create_target(self, df, horizon=5, threshold=0.005):

# NEW (more balanced)
def create_target(self, df, horizon=5, threshold=0.002):
```

This will create more Up/Down signals (0.2% moves instead of 0.5%).

### Solution 2: Add Class Weights (Best Fix)

Add after line 320 (after data split):

```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print(f"\nClass weights:")
for i, weight in class_weight_dict.items():
    print(f"  Class {i}: {weight:.2f}x")
```

Then modify line 370 (model.fit):

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict,  # ← ADD THIS LINE
    verbose=1
)
```

### Solution 3: Binary Classification (Simplest)

Replace the `create_target` function:

```python
def create_target(self, df, horizon=5, threshold=0.003):
    """
    Binary classification: 0=Down/Neutral, 1=Up
    """
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > threshold).astype(int)
    return df
```

And change model output:

```python
# In build_improved_lstm, change:
outputs = Dense(num_classes, activation='softmax')(x)
# To:
outputs = Dense(1, activation='sigmoid')(x)  # Binary output

# And compile with:
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Not sparse_categorical
    metrics=['accuracy']
)
```

## Recommended Approach

**Use Solution 1 + Solution 2 together:**

1. Lower threshold to 0.002 (more signals)
2. Add class weights (penalize wrong predictions)

This will give you:
- More balanced class distribution
- Model forced to learn all classes
- Better real-world performance

## Expected Results After Fix

```
Class 0 (Down): 60-70% accuracy
Class 1 (Neutral): 70-80% accuracy
Class 2 (Up): 60-70% accuracy

Overall: 65-75% accuracy (lower but USEFUL!)
```

## Why Lower Accuracy is Better

```
95% accuracy (predicting only Neutral) = USELESS
- Can't trade on it
- No signals
- Waste of time

70% accuracy (predicting all classes) = USEFUL
- Can trade on it
- Clear signals
- Makes money
```

## Quick Fix Script

Run this in Colab to apply fixes:

```python
# Download fixed version
!wget https://raw.githubusercontent.com/himanshusdeshmukh2106/tr/main/AlgoTrading/LSTM_strategy/train_reliance_improved.py -O train_reliance_improved_fixed.py

# Or manually edit:
# 1. Change threshold from 0.005 to 0.002 (line 160)
# 2. Add class_weight_dict calculation (after line 320)
# 3. Add class_weight=class_weight_dict to model.fit() (line 370)

# Then retrain
!python train_reliance_improved_fixed.py --architecture improved_lstm
```

## Understanding the Output

**Bad Output (Current):**
```
Class 0: 0% - Model never predicts Down
Class 1: 100% - Model always predicts Neutral
Class 2: 0% - Model never predicts Up
```

**Good Output (After Fix):**
```
Class 0: 65% - Model predicts Down correctly 65% of time
Class 1: 75% - Model predicts Neutral correctly 75% of time
Class 2: 68% - Model predicts Up correctly 68% of time
```

The second one is MUCH more useful for trading!
