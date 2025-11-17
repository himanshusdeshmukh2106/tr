"""
TCN with Focal Loss - NO SMOTE VERSION
Train on REAL imbalanced data only to see if genuine patterns exist

Key differences from SMOTE version:
1. NO synthetic data generation
2. Uses real market data only
3. Evaluates with F1-score (better for imbalanced data)
4. Heavy class weights to compensate for imbalance
5. This will reveal if SMOTE was creating fake patterns
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import ta
from pathlib import Path
import json
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
import os
if os.path.exists("reliance_data_5min_full_year.csv"):
    DATA_FILE = "reliance_data_5min_full_year.csv"
elif os.path.exists("../../reliance_data_5min_full_year.csv"):
    DATA_FILE = "../../reliance_data_5min_full_year.csv"
else:
    DATA_FILE = "reliance_data_5min_full_year.csv"

MODELS_DIR = Path("models/reliance_tcn_no_smote")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=[0.4, 0.2, 0.4], gamma=3.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, self.gamma)
        alpha_weight = y_true * self.alpha
        focal_loss = alpha_weight * weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


# ============================================================================
# TCN ARCHITECTURE
# ============================================================================

def residual_block(x, dilation_rate, nb_filters, kernel_size, padding='causal', dropout_rate=0.2):
    conv1 = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        activation='relu'
    )(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(dropout_rate)(conv1)
    
    conv2 = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        activation='relu'
    )(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(dropout_rate)(conv2)
    
    if x.shape[-1] != nb_filters:
        x = layers.Conv1D(nb_filters, 1, padding='same')(x)
    
    out = layers.Add()([x, conv2])
    out = layers.Activation('relu')(out)
    
    return out


def build_tcn_model(input_shape, num_classes=3):
    inputs = layers.Input(shape=input_shape)
    
    x = residual_block(inputs, dilation_rate=1, nb_filters=64, kernel_size=3, dropout_rate=0.3)
    x = residual_block(x, dilation_rate=2, nb_filters=64, kernel_size=3, dropout_rate=0.3)
    x = residual_block(x, dilation_rate=4, nb_filters=64, kernel_size=3, dropout_rate=0.3)
    x = residual_block(x, dilation_rate=8, nb_filters=32, kernel_size=3, dropout_rate=0.3)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


# ============================================================================
# DATA PREPARATION
# ============================================================================

class DataPrep:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        print(f"Loaded {len(df)} rows (5-minute data)")
        
        print("Converting to 15-minute timeframe...")
        df_15min = df.resample('15T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        print(f"After resampling: {len(df_15min)} rows (15-minute data)")
        return df_15min
    
    def add_features(self, df):
        # Candle characteristics
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        
        # Price action
        df['Returns'] = df['Close'].pct_change()
        df['Returns_2'] = df['Close'].pct_change(2)
        df['Returns_5'] = df['Close'].pct_change(5)
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['Price_to_SMA5'] = df['Close'] / df['SMA_5']
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_EMA21'] = df['Close'] / df['EMA_21']
        
        # Trend strength
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Volume
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        # Price levels
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Distance_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_Low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        # Time features
        df['Hour'] = df.index.hour
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        return df
    
    def create_target(self, df, horizon=3, threshold=0.004):
        """
        threshold=0.004 (0.4%) - slightly higher to get clearer signals
        """
        df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = 1  # Neutral
        df.loc[df['Future_Return'] < -threshold, 'Target'] = 0  # Down
        df.loc[df['Future_Return'] > threshold, 'Target'] = 2   # Up
        return df
    
    def prepare_sequences(self, df, sequence_length=20):
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['Target', 'Future_Return']]
        
        df = df.dropna()
        
        # Remove correlated features
        corr_matrix = df[feature_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        feature_columns = [col for col in feature_columns if col not in to_drop]
        
        print(f"Using {len(feature_columns)} features")
        
        # Scale
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        return np.array(X), np.array(y)


# ============================================================================
# TRAINING - NO SMOTE
# ============================================================================

def train_no_smote():
    print("="*80)
    print("TCN WITH FOCAL LOSS - NO SMOTE (REAL DATA ONLY)")
    print("="*80)
    print("\n🔬 Critical Test:")
    print("  - NO synthetic data (no SMOTE)")
    print("  - Train on real imbalanced data only")
    print("  - Heavy Focal Loss (gamma=3.0)")
    print("  - Aggressive class weights")
    print("  - Evaluate with F1-score")
    print("\n💡 This will reveal:")
    print("  - If SMOTE was creating fake patterns")
    print("  - If real predictive patterns exist")
    print("  - True model capability on real data")
    print()
    
    # Load data
    prep = DataPrep()
    df = prep.load_data(DATA_FILE)
    df = prep.add_features(df)
    df = prep.create_target(df, horizon=3, threshold=0.004)
    
    # Create sequences
    X, y = prep.prepare_sequences(df, sequence_length=20)
    
    print(f"\nData: X={X.shape}, y={y.shape}")
    print(f"\n⚠️  REAL DATA CLASS DISTRIBUTION (NO SMOTE):")
    for class_val in [0, 1, 2]:
        count = (y == class_val).sum()
        pct = count / len(y) * 100
        class_name = ["DOWN", "NEUTRAL", "UP"][class_val]
        print(f"  Class {class_val} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n✓ NO SMOTE - Training on real imbalanced data")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Build TCN model
    print("\n🏗️  Building TCN model...")
    model = build_tcn_model(X_train.shape[1:], num_classes=3)
    
    # Focal Loss with aggressive parameters
    focal_loss = FocalLoss(alpha=[0.4, 0.2, 0.4], gamma=3.0)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    # VERY aggressive class weights (no SMOTE to balance)
    class_weight = {0: 5.0, 1: 1.0, 2: 5.0}
    print(f"\n⚡ AGGRESSIVE class weights: {class_weight}")
    print("  (Compensating for imbalance without SMOTE)")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODELS_DIR / 'best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    ]
    
    # Train
    print(f"\n{'='*80}")
    print("Training on REAL DATA (no synthetic samples)...")
    print(f"{'='*80}\n")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Evaluate
    print(f"\n{'='*80}")
    print("EVALUATION ON REAL DATA")
    print(f"{'='*80}")
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✓ Overall Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Per-class metrics
    print(f"\n{'='*80}")
    print("PER-CLASS PERFORMANCE (REAL DATA)")
    print(f"{'='*80}")
    
    class_names = ["DOWN (Short)", "NEUTRAL (Stay Out)", "UP (Long)"]
    for class_val in [0, 1, 2]:
        mask = y_test == class_val
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            class_f1 = f1_score(y_test[mask], y_pred[mask], average='micro')
            print(f"\n✓ Class {class_val} - {class_names[class_val]}")
            print(f"  Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")
            print(f"  F1-Score: {class_f1:.4f}")
            print(f"  Samples: {mask.sum():,}")
    
    # Overall F1 scores
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"\n📊 Overall Metrics:")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    print(f"\n{'='*80}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*80}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print(f"{'='*80}")
    cm = confusion_matrix(y_test, y_pred)
    print("\n           Predicted")
    print("           DOWN  NEUTRAL  UP")
    print(f"Actual DOWN    {cm[0,0]:4d}    {cm[0,1]:4d}  {cm[0,2]:4d}")
    print(f"     NEUTRAL   {cm[1,0]:4d}    {cm[1,1]:4d}  {cm[1,2]:4d}")
    print(f"          UP   {cm[2,0]:4d}    {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Confidence analysis
    confidence = np.max(y_pred_probs, axis=1)
    
    print(f"\n{'='*80}")
    print("CONFIDENCE-BASED ANALYSIS")
    print(f"{'='*80}")
    
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        high_conf = confidence > threshold
        if high_conf.sum() > 0:
            acc = (y_pred[high_conf] == y_test[high_conf]).mean()
            coverage = high_conf.sum() / len(y_test)
            print(f"\n✓ Confidence > {threshold}: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  Coverage: {high_conf.sum():,} ({coverage*100:.1f}%)")
            
            for class_val in [0, 1, 2]:
                class_count = (y_pred[high_conf] == class_val).sum()
                if class_count > 0:
                    print(f"    {class_names[class_val]}: {class_count}")
    
    # Save
    model.save(MODELS_DIR / 'model.h5')
    joblib.dump(prep.scaler, MODELS_DIR / 'scaler.pkl')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'approach': 'NO SMOTE - Real data only',
        'overall_accuracy': float(test_acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Model saved to: {MODELS_DIR}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\n🔍 CRITICAL ANALYSIS:")
    print("  Compare these results with SMOTE version:")
    print("  - If similar/better → Real patterns exist!")
    print("  - If much worse → SMOTE was creating fake patterns")
    print("  - Look at F1-scores, not just accuracy")
    print("  - Check if UP/DOWN predictions actually happen")
    
    return model, history, prep


if __name__ == "__main__":
    model, history, prep = train_no_smote()
