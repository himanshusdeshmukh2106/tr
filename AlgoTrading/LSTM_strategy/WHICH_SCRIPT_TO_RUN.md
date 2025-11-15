# Which Optimization Script Should I Run?

## 🎯 Quick Decision Guide

### Just want to test the approach? (30 min)
```bash
python test_optimization.py
```
- Tests 5 configs
- Validates the optimization approach works
- Shows if you're on the right track

### Want good results quickly? (2-3 hours) ⭐ RECOMMENDED
```bash
python quick_optimization.py
```
- Tests 20 strategic configs
- Expected Class 1: 48-55%
- Best balance of time vs results

### Want the absolute best? (8-12 hours)
```bash
python hyperparameter_search.py
```
- Tests 500+ configs
- Expected Class 1: 52-58%
- Run overnight for best results

---

## 📊 Comparison Table

| Script | Time | Configs | Expected Class 1 | When to Use |
|--------|------|---------|------------------|-------------|
| test_optimization.py | 30 min | 5 | 42-48% | Quick validation |
| quick_optimization.py | 2-3 hrs | 20 | 48-55% | **Best choice** ⭐ |
| hyperparameter_search.py | 8-12 hrs | 500+ | 52-58% | Overnight run |

---

## 🚀 Recommended Workflow

### Step 1: Quick Test (30 min)
```bash
python test_optimization.py
```

**Check results:**
- If Class 1 > 45%: Great! Move to Step 2
- If Class 1 < 45%: Data might have issues

### Step 2: Quick Optimization (2-3 hrs)
```bash
python quick_optimization.py
```

**Check results:**
- If Class 1 > 50%: Excellent! Use this config ⭐
- If Class 1 45-50%: Good enough for most cases
- If Class 1 < 45%: Move to Step 3

### Step 3: Comprehensive Search (overnight)
```bash
python hyperparameter_search.py
```

**Check results:**
- Should get Class 1 > 52%
- This is the best you can do with current data

---

## 💡 What Each Script Does

### test_optimization.py
**Tests:**
1. Current baseline (horizon=10, threshold=0.0015)
2. Lower threshold (0.0012)
3. Higher threshold (0.0018)
4. Shorter horizon (8)
5. Longer horizon (12)

**Purpose:** Validate that changing parameters helps

### quick_optimization.py
**Tests:**
- Threshold variations
- Horizon variations
- Model sizes (small, medium, large)
- Dropout rates
- Learning rates
- Alternative architectures (LSTM, GRU, Deep LSTM)
- Strategic combinations

**Purpose:** Find a good configuration quickly

### hyperparameter_search.py
**Tests:**
- Full grid search
- Every combination of:
  - 3 horizons × 4 thresholds × 3 seq_lengths
  - 3 model sizes × 3 dropouts × 3 learning rates × 2 batch sizes
- Total: 500+ combinations

**Purpose:** Find the absolute best configuration

---

## 🎓 Understanding Results

### Good Results
```
Class 1: > 50% ⭐
Overall: > 68%
High Conf: > 80%
```

### Acceptable Results
```
Class 1: 45-50%
Overall: 65-68%
High Conf: 75-80%
```

### Poor Results (need more optimization)
```
Class 1: < 45%
Overall: < 65%
High Conf: < 75%
```

---

## 🔧 Troubleshooting

### "Class imbalance" warnings
**Normal!** The script automatically skips configs with extreme class imbalance.

### Script crashes
- Check RAM (need 8GB+)
- Try test_optimization.py first
- Reduce number of configs

### No improvement
- Data quality might be the issue
- Try different features
- Consider ensemble methods

---

## 📈 Expected Improvements

### Current State
```
Class 0: 71-79%
Class 1: 34-42% ❌
Overall: 63-68%
```

### After test_optimization.py
```
Class 0: 72-78%
Class 1: 42-48%
Overall: 64-68%
Improvement: +4-8% on Class 1
```

### After quick_optimization.py
```
Class 0: 74-80%
Class 1: 48-55% ⭐
Overall: 66-70%
Improvement: +10-17% on Class 1
```

### After hyperparameter_search.py
```
Class 0: 76-82%
Class 1: 52-58% ⭐⭐
Overall: 68-72%
Improvement: +14-20% on Class 1
```

---

## ⏱️ Time Investment vs Results

```
30 min  → +4-8%   improvement  (test_optimization.py)
2-3 hrs → +10-17% improvement  (quick_optimization.py) ⭐
8-12 hrs → +14-20% improvement (hyperparameter_search.py)
```

**Diminishing returns:** The comprehensive search only adds +4-6% over quick optimization but takes 4x longer.

**Recommendation:** Start with quick_optimization.py. Only run comprehensive if you need that extra 4-6%.

---

## 🎯 My Recommendation

### For Most Users
```bash
# Run this and you're done
python quick_optimization.py
```

**Why:**
- 2-3 hours is reasonable
- Gets you 90% of the way there
- Good balance of time vs results

### For Perfectionists
```bash
# Run overnight
python hyperparameter_search.py
```

**Why:**
- Finds absolute best config
- That extra 4-6% might matter for your use case
- You have time to let it run

### For Quick Validation
```bash
# Just testing
python test_optimization.py
```

**Why:**
- See if optimization helps at all
- Only 30 minutes
- Good for debugging

---

## 📝 Summary

**TL;DR:** Run `quick_optimization.py` for best results in reasonable time.

**Expected outcome:** Class 1 accuracy improves from 34-42% to 48-55% 🎯

**Time:** 2-3 hours

**Next steps:** Use the best config to retrain your model

Good luck! 📈
