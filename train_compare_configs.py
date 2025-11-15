"""
Compare Different Training Configurations
Tests multiple threshold and horizon combinations to find the best
"""

import sys
sys.path.append('AlgoTrading/LSTM_strategy')

from train_reliance_balanced import *
import json
from datetime import datetime

print("="*80)
print("CONFIGURATION COMPARISON")
print("="*80)

# Test configurations
configs = [
    # (horizon, threshold, name)
    (5, 0.001, "Original (5 candles, 0.1%)"),
    (10, 0.001, "Longer horizon (10 candles, 0.1%)"),
    (5, 0.002, "Higher threshold (5 candles, 0.2%)"),
    (10, 0.002, "OPTIMIZED (10 candles, 0.2%)"),
    (10, 0.0015, "Balanced (10 candles, 0.15%)"),
]

results = []

for horizon, threshold, name in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    try:
        # Prepare data
        prep = BalancedDataPreparation()
        df = prep.load_data(DATA_FILE)
        df = prep.add_features(df)
        
        # Create target with specific params
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > threshold).astype(int)
        
        # Prepare sequences
        X, y, features = prep.prepare_sequences(df, sequence_length=30)
        
        print(f"\nClass distribution:")
        print(f"  Class 0: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"  Class 1: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # SMOTE
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
        X_train_balanced = X_train_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
        
        # Build and train model
        model = build_simple_lstm(X_train.shape[1:])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
        ]
        
        history = model.fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Per-class accuracy
        y_pred = model.predict(X_test, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        class0_acc = (y_pred_binary[y_test == 0] == y_test[y_test == 0]).mean()
        class1_acc = (y_pred_binary[y_test == 1] == y_test[y_test == 1]).mean()
        
        # Confidence-based accuracy
        confidence = np.maximum(y_pred, 1 - y_pred).flatten()
        high_conf_mask = confidence > 0.7
        high_conf_acc = (y_pred_binary[high_conf_mask] == y_test[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
        
        result = {
            'name': name,
            'horizon': horizon,
            'threshold': threshold,
            'overall_acc': float(test_acc),
            'class0_acc': float(class0_acc),
            'class1_acc': float(class1_acc),
            'high_conf_acc': float(high_conf_acc),
            'high_conf_coverage': float(high_conf_mask.sum() / len(y_test)),
            'class1_samples': int((y_test == 1).sum())
        }
        
        results.append(result)
        
        print(f"\n✓ Results:")
        print(f"  Overall: {test_acc:.1%}")
        print(f"  Class 0: {class0_acc:.1%}")
        print(f"  Class 1: {class1_acc:.1%}")
        print(f"  High Conf (>0.7): {high_conf_acc:.1%}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue

# Summary
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}\n")

# Sort by Class 1 accuracy (the problem class)
results_sorted = sorted(results, key=lambda x: x['class1_acc'], reverse=True)

print(f"{'Configuration':<40} {'Overall':<10} {'Class 0':<10} {'Class 1':<10} {'High Conf':<10}")
print("-" * 80)

for r in results_sorted:
    print(f"{r['name']:<40} {r['overall_acc']:<10.1%} {r['class0_acc']:<10.1%} {r['class1_acc']:<10.1%} {r['high_conf_acc']:<10.1%}")

# Best configuration
best = results_sorted[0]
print(f"\n🏆 BEST CONFIGURATION:")
print(f"  {best['name']}")
print(f"  Horizon: {best['horizon']} candles")
print(f"  Threshold: {best['threshold']:.3f} ({best['threshold']*100:.1f}%)")
print(f"  Class 1 Accuracy: {best['class1_acc']:.1%} ⭐")
print(f"  Overall Accuracy: {best['overall_acc']:.1%}")
print(f"  High Confidence: {best['high_conf_acc']:.1%}")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'results': results_sorted,
    'best_config': best
}

with open('config_comparison_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to config_comparison_results.json")
print("\n" + "="*80)
print("Done! 🎉")
print("="*80)
