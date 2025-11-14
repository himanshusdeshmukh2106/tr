"""
IMPROVED Vertex AI training task for RELIANCE LSTM model
Addresses overfitting and poor generalization
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from google.cloud import storage
import json
from datetime import datetime


def download_data_from_gcs(bucket_name, project_id):
    """Download training data from GCS"""
    print("Downloading data from GCS...")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob("reliance/X_reliance_daily.npy")
    blob.download_to_filename("/tmp/X_reliance_daily.npy")
    
    blob = bucket.blob("reliance/y_reliance_daily.npy")
    blob.download_to_filename("/tmp/y_reliance_daily.npy")
    
    X = np.load("/tmp/X_reliance_daily.npy")
    y = np.load("/tmp/y_reliance_daily.npy")
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    return X, y


def augment_data(X, y, noise_factor=0.01):
    """Add noise augmentation to training data"""
    X_aug = X + np.random.normal(0, noise_factor, X.shape)
    return np.concatenate([X, X_aug]), np.concatenate([y, y])


def build_improved_model(input_shape):
    """Build improved LSTM with better regularization"""
    inputs = layers.Input(shape=input_shape)
    
    # Bidirectional LSTM for better context
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, 
                   kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))
    )(inputs)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                   kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))
    )(x)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.LayerNormalization()(x)
    
    # Multi-head attention
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Global pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Dense with strong regularization
    x = layers.Dense(64, activation='relu', 
                    kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_transformer_model(input_shape):
    """Build transformer-based model"""
    inputs = layers.Input(shape=input_shape)
    
    # Positional encoding
    x = layers.Dense(128)(inputs)
    
    # Transformer blocks
    for _ in range(2):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.2
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dropout(0.2)(ff)
        ff = layers.Dense(128)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_cnn_lstm_model(input_shape):
    """Build CNN-LSTM hybrid model"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN for feature extraction
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # LSTM for sequence modeling
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class FocalLoss(keras.losses.Loss):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)


def train_model(X_train, y_train, X_val, y_val, config, model_dir):
    """Train model with improved techniques"""
    print(f"\n{'='*70}")
    print(f"Training: {config['name']}")
    print(f"{'='*70}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    if config['architecture'] == 'lstm':
        model = build_improved_model(input_shape)
    elif config['architecture'] == 'transformer':
        model = build_transformer_model(input_shape)
    else:
        model = build_cnn_lstm_model(input_shape)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Compile with focal loss or weighted BCE
    if config.get('use_focal_loss', False):
        loss = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        loss = 'binary_crossentropy'
    
    optimizer = keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        clipnorm=1.0,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=25,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=12,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"{config['name']}_best.h5"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def main(args):
    """Main training pipeline"""
    print("="*70)
    print("IMPROVED VERTEX AI TRAINING - RELIANCE LSTM")
    print("="*70)
    
    # Download data
    X, y = download_data_from_gcs(args.bucket_name, args.project_id)
    
    # Split data (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape} - Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)}")
    print(f"  Val: {X_val.shape} - Class 0: {np.sum(y_val==0)}, Class 1: {np.sum(y_val==1)}")
    
    # Data augmentation
    if args.augment:
        print("\nApplying data augmentation...")
        X_train, y_train = augment_data(X_train, y_train, noise_factor=0.005)
        print(f"  Augmented train: {X_train.shape}")
    
    # Training configurations
    configs = [
        {
            "name": "improved_lstm_v1",
            "architecture": "lstm",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 150,
            "use_focal_loss": False
        },
        {
            "name": "improved_lstm_focal",
            "architecture": "lstm",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 150,
            "use_focal_loss": True
        },
        {
            "name": "transformer_v1",
            "architecture": "transformer",
            "learning_rate": 0.0005,
            "batch_size": 16,
            "epochs": 150,
            "use_focal_loss": False
        },
        {
            "name": "cnn_lstm_hybrid",
            "architecture": "cnn_lstm",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 150,
            "use_focal_loss": False
        }
    ]
    
    results = []
    best_model = None
    best_score = 0
    best_config_name = None
    
    for config in configs:
        try:
            model, history = train_model(
                X_train, y_train, X_val, y_val,
                config, args.model_dir
            )
            
            # Evaluate
            val_metrics = model.evaluate(X_val, y_val, verbose=0)
            val_loss, val_acc, val_auc, val_precision, val_recall = val_metrics
            
            f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
            combined_score = val_auc * 0.6 + f1_score * 0.4
            
            result = {
                "config": config["name"],
                "architecture": config["architecture"],
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "val_auc": float(val_auc),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "f1_score": float(f1_score),
                "combined_score": float(combined_score)
            }
            results.append(result)
            
            print(f"\n{config['name']} Results:")
            print(f"  Accuracy: {val_acc:.4f}")
            print(f"  AUC: {val_auc:.4f}")
            print(f"  Precision: {val_precision:.4f}")
            print(f"  Recall: {val_recall:.4f}")
            print(f"  F1: {f1_score:.4f}")
            print(f"  Combined Score: {combined_score:.4f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
                best_config_name = config["name"]
        
        except Exception as e:
            print(f"Error training {config['name']}: {e}")
            continue
    
    # Save best model
    if best_model:
        best_model_path = os.path.join(args.model_dir, "best_model.h5")
        best_model.save(best_model_path)
        print(f"\n✓ Best model saved: {best_model_path}")
    
    # Save results
    results_path = os.path.join(args.model_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": results,
            "best_config": best_config_name,
            "best_score": float(best_score),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_config_name}")
    print(f"Best Combined Score: {best_score:.4f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, default=os.environ.get('PROJECT_ID', 'brave-operand-477117-n3'))
    parser.add_argument('--bucket-name', type=str, default=os.environ.get('BUCKET_NAME', 'lstm-trading-asia-south1'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR', '/tmp/model'))
    parser.add_argument('--augment', action='store_true', default=os.environ.get('AUGMENT', 'false').lower() == 'true')
    
    args = parser.parse_args()
    main(args)
