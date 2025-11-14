"""
Ensemble model combining LSTM, XGBoost, Random Forest, and LightGBM
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config.config import DATA_DIR, MODELS_DIR as MODEL_DIR


def load_data():
    """Load NIFTY 50 data"""
    print("Loading NIFTY 50 data...")
    X = np.load(DATA_DIR / "X_nifty50.npy")
    y = np.load(DATA_DIR / "y_nifty50.npy")
    print(f"Data: X={X.shape}, y={y.shape}")
    return X, y


def build_lstm_model(input_shape):
    """Build LSTM model"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)


def train_lstm(X_train, y_train, X_val, y_val):
    """Train LSTM model"""
    print("\n" + "="*70)
    print("Training LSTM")
    print("="*70)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, mode='max')
        ],
        verbose=1
    )
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost"""
    print("\n" + "="*70)
    print("Training XGBoost")
    print("="*70)
    
    # Flatten for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=True
    )
    
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest"""
    print("\n" + "="*70)
    print("Training Random Forest")
    print("="*70)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_flat, y_train)
    
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM"""
    print("\n" + "="*70)
    print("Training LightGBM")
    print("="*70)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(10)]
    )
    
    return model


class EnsembleModel:
    """Ensemble combining all models"""
    
    def __init__(self):
        self.lstm = None
        self.xgb = None
        self.rf = None
        self.lgbm = None
        self.weights = None
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Train all models"""
        # Train individual models
        self.lstm = train_lstm(X_train, y_train, X_val, y_val)
        self.xgb = train_xgboost(X_train, y_train, X_val, y_val)
        self.rf = train_random_forest(X_train, y_train)
        self.lgbm = train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Calculate optimal weights based on validation performance
        print("\n" + "="*70)
        print("Calculating Ensemble Weights")
        print("="*70)
        
        lstm_pred = self.lstm.predict(X_val).flatten()
        xgb_pred = self.xgb.predict_proba(X_val.reshape(X_val.shape[0], -1))[:, 1]
        rf_pred = self.rf.predict_proba(X_val.reshape(X_val.shape[0], -1))[:, 1]
        lgbm_pred = self.lgbm.predict_proba(X_val.reshape(X_val.shape[0], -1))[:, 1]
        
        # Calculate AUC for each model
        from sklearn.metrics import roc_auc_score
        lstm_auc = roc_auc_score(y_val, lstm_pred)
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        rf_auc = roc_auc_score(y_val, rf_pred)
        lgbm_auc = roc_auc_score(y_val, lgbm_pred)
        
        print(f"LSTM AUC: {lstm_auc:.4f}")
        print(f"XGBoost AUC: {xgb_auc:.4f}")
        print(f"Random Forest AUC: {rf_auc:.4f}")
        print(f"LightGBM AUC: {lgbm_auc:.4f}")
        
        # Weight by AUC
        total_auc = lstm_auc + xgb_auc + rf_auc + lgbm_auc
        self.weights = {
            'lstm': lstm_auc / total_auc,
            'xgb': xgb_auc / total_auc,
            'rf': rf_auc / total_auc,
            'lgbm': lgbm_auc / total_auc
        }
        
        print(f"\nEnsemble Weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
    
    def predict_proba(self, X):
        """Predict with ensemble"""
        lstm_pred = self.lstm.predict(X, verbose=0).flatten()
        xgb_pred = self.xgb.predict_proba(X.reshape(X.shape[0], -1))[:, 1]
        rf_pred = self.rf.predict_proba(X.reshape(X.shape[0], -1))[:, 1]
        lgbm_pred = self.lgbm.predict_proba(X.reshape(X.shape[0], -1))[:, 1]
        
        # Weighted average
        ensemble_pred = (
            self.weights['lstm'] * lstm_pred +
            self.weights['xgb'] * xgb_pred +
            self.weights['rf'] * rf_pred +
            self.weights['lgbm'] * lgbm_pred
        )
        
        return ensemble_pred
    
    def predict(self, X):
        """Predict class"""
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def save(self, path):
        """Save ensemble"""
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        self.lstm.save(str(path / "lstm_model.h5"))
        joblib.dump(self.xgb, path / "xgb_model.pkl")
        joblib.dump(self.rf, path / "rf_model.pkl")
        joblib.dump(self.lgbm, path / "lgbm_model.pkl")
        joblib.dump(self.weights, path / "weights.pkl")
        
        print(f"\n✓ Ensemble saved to {path}")


def evaluate_model(model, X, y, name="Model"):
    """Evaluate model"""
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    
    pred_proba = model.predict_proba(X)
    pred = (pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred_proba)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    
    print(f"\n{name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


def main():
    """Main training pipeline"""
    print("="*70)
    print("ENSEMBLE MODEL TRAINING - NIFTY 50")
    print("="*70)
    
    # Load data
    X, y = load_data()
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")
    
    # Train ensemble
    ensemble = EnsembleModel()
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    train_results = evaluate_model(ensemble, X_train, y_train, "Training")
    val_results = evaluate_model(ensemble, X_val, y_val, "Validation")
    
    # Save
    ensemble.save(MODEL_DIR / "ensemble")
    
    # Save results
    results = {
        "train": train_results,
        "validation": val_results,
        "weights": ensemble.weights,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(MODEL_DIR / "ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ ENSEMBLE TRAINING COMPLETE!")
    print("="*70)
    print(f"Models saved: {MODEL_DIR / 'ensemble'}")
    print(f"Results: {MODEL_DIR / 'ensemble_results.json'}")


if __name__ == "__main__":
    import lightgbm as lgb
    main()
