"""
Vertex AI training task for RELIANCE LSTM model
This script runs on GCP infrastructure
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from google.cloud import storage
import json
from datetime import datetime


def download_data_from_gcs(bucket_name, project_id):
    """Download training data from GCS"""
    print("Downloading data from GCS...")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Download data files
    blob = bucket.blob("reliance/X_reliance_daily.npy")
    blob.download_to_filename("/tmp/X_reliance_daily.npy")
    
    blob = bucket.blob("reliance/y_reliance_daily.npy")
    blob.download_to_filename("/tmp/y_reliance_daily.npy")
    
    X = np.load("/tmp/X_reliance_daily.npy")
    y = np.load("/tmp/y_reliance_daily.npy")
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    return X, y


def build_model(input_shape, config):
    """Build LSTM model"""
    inputs = layers.Input(shape=input_shape, name='input')
    
    # LSTM layers
    x = layers.LSTM(
        config["lstm_units"][0],
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(0.0005),
        recurrent_dropout=0.1
    )(inputs)
    x = layers.Dropout(config["dropout_rate"])(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(
        config["lstm_units"][1],
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(0.0005),
        recurrent_dropout=0.1
    )(x)
    x = layers.Dropout(config["dropout_rate"])(x)
    x = layers.BatchNormalization()(x)
    
    # Attention
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(config["lstm_units"][1])(attention)
    attention = layers.Permute([2, 1])(attention)
    x = layers.Multiply()([x, attention])
    
    x = layers.LSTM(
        config["lstm_units"][2],
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(0.0005)
    )(x)
    x = layers.Dropout(config["dropout_rate"])(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(X_train, y_train, X_val, y_val, config, model_dir):
    """Train the model"""
    print(f"\nTraining with config: {config['name']}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, config)
    
    optimizer = keras.optimizers.Adam(
        learning_rate=config["learning_rate"],
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(model_dir, f"{config['name']}_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history


def main(args):
    """Main training function"""
    print("="*70)
    print("VERTEX AI TRAINING - RELIANCE LSTM")
    print("="*70)
    
    # Download data
    X, y = download_data_from_gcs(args.bucket_name, args.project_id)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Training configurations
    configs = [
        {
            "name": "refined_1_wider_deeper",
            "lstm_units": [256, 128, 64],
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 8,
            "epochs": 200
        },
        {
            "name": "refined_2_optimal",
            "lstm_units": [192, 96, 48],
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "epochs": 200
        },
        {
            "name": "refined_3_balanced",
            "lstm_units": [224, 112, 56],
            "dropout_rate": 0.32,
            "learning_rate": 0.0006,
            "batch_size": 12,
            "epochs": 200
        },
        {
            "name": "refined_4_aggressive",
            "lstm_units": [320, 160, 80],
            "dropout_rate": 0.35,
            "learning_rate": 0.0003,
            "batch_size": 8,
            "epochs": 200
        }
    ]
    
    results = []
    best_model = None
    best_score = 0
    best_config = None
    
    for config in configs:
        model, history = train_model(
            X_train, y_train, X_val, y_val, 
            config, args.model_dir
        )
        
        # Evaluate
        val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        
        result = {
            "config": config["name"],
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_auc": float(val_auc),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "f1_score": float(f1_score)
        }
        results.append(result)
        
        print(f"\n{config['name']} Results:")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        
        combined_score = val_auc * 0.5 + f1_score * 0.5
        if combined_score > best_score:
            best_score = combined_score
            best_model = model
            best_config = config
    
    # Save best model
    best_model_path = os.path.join(args.model_dir, "best_model.h5")
    best_model.save(best_model_path)
    
    # Save results
    results_path = os.path.join(args.model_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_config['name']}")
    print(f"Best Score: {best_score:.4f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, default='brave-operand-477117-n3')
    parser.add_argument('--bucket-name', type=str, default='lstm-trading-asia-south1')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR', '/tmp/model'))
    
    args = parser.parse_args()
    main(args)
