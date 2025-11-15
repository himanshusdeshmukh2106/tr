# Quick Reference - Optimized LSTM Model

## 🎯 Best Configuration (Copy-Paste Ready)

```python
# OPTIMIZED PARAMETERS - Use these!
horizon = 8              # 40 minutes
threshold = 0.0015       # 0.15%
sequence_length = 30     # 30 candles
lstm_units = [64, 32]    # Standard LSTM
dropout = 0.3            # 30% dropout
learning_rate = 0.001    # Standard LR
batch_size = 64          # Standard batch
```

## 📊 Expected Performance

```
Class 1 Accuracy:  51.2% (was 34-42%)
Overall Accuracy:  63.9%
High Confidence:   91.2% (>0.7 threshold)
Weighted Score:    0.657
```

## 🚀 Train Model

```bash
cd AlgoTrading/LSTM_strategy
python train_optimized_final.py
```

## 💰 Trading Strategy

### High Confidence (Recommended)
```python
if prediction > 0.7:
    BUY()  # 91% win rate
elif prediction < 0.3:
    SELL()
else:
    SKIP()
```

### Balanced
```python
if prediction > 0.5:
    BUY()  # 51% win rate
else:
    SELL()
```

## 📁 Key Files

- `train_optimized_final.py` - Train with best config
- `OPTIMIZATION_RESULTS.md` - Detailed analysis
- `NEXT_STEPS.md` - What to do next
- `quick_optimization_results/results.json` - Full results

## 🎓 Key Insights

1. **Shorter horizons win** (8 > 10 candles)
2. **High confidence is gold** (91% accuracy)
3. **Simple models work** (standard LSTM)
4. **Optimization pays off** (+9-17% improvement)

## ✅ Success Metrics

- Class 1: ✅ 51.2% (target: >50%)
- Overall: ✅ 63.9% (target: >60%)
- High Conf: ✅ 91.2% (target: >85%)
- Score: ✅ 0.657 (target: >0.60)

**All targets exceeded!** 🎉
