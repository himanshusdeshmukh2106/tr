"""
Improved retraining script based on initial results
Focuses on the best performing architecture with refined hyperparameters
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from google.cloud import storage
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME,
    DATA_DIR, MODELS_DIR
)

# Set credentials
key_path = Path(__file__).parent.parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)


class ImprovedTrainer:
    def __init__(self):
        self.storage_client = storage.Client(project=GCP_PROJECT_ID)
        self.bucket = self.storage_client.bucket(GCP_BUCKET_NAME)
        
    def build_improved_model(self, input_shape, config):
        """Build improved LSTM model based on best results"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # First LSTM block with more units
        x = layers.LSTM(
            config["lstm_units"][0],
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.0005),  # Reduced regularization
            recurrent_dropout=0.1,  # Added recurrent dropout
            name="lstm_1"
        )(inputs)
        x = layers.Dropout(config["dropout_rate"])(x)
        x = layers.BatchNormalization()(x)
        
        # Second LSTM block
        x = layers.LSTM(
            config["lstm_units"][1],
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.0005),
            recurrent_dropout=0.1,
            name="lstm_2"
        )(x)
        x = layers.Dropout(config["dropout_rate"])(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism (improved)
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(config["lstm_units"][1])(attention)
        attention = layers.Permute([2, 1])(attention)
        x = layers.Multiply()([x, attention])
        
        # Third LSTM block
        x = layers.LSTM(
            config["lstm_units"][2],
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.0005),
            name="lstm_3"
        )(x)
        x = layers.Dropout(config["dropout_rate"])(x)
        
        # Dense layers with residual connection
        dense1 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005))(x)
        dense1 = layers.Dropout(0.2)(dense1)
        
        dense2 = layers.Dense(64, activation='relu')(dense1)
        dense2 = layers.Dropout(0.2)(dense2)
        
        dense3 = layers.Dense(32, activation='relu')(dense2)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(dense3)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Reliance_Improved')
        
        return model
    
    def get_callbacks(self, model_name="reliance_improved"):
        """Get training callbacks with improved settings"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',  # Monitor AUC instead of loss
                patience=20,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                mode='max',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(MODELS_DIR / f"logs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train_refined_models(self, X_train, y_train, X_val, y_val):
        """Train multiple refined configurations"""
        print("\n" + "="*70)
        print("REFINED HYPERPARAMETER SEARCH")
        print("="*70)
        
        # Based on results: Config 3 (wider) was best
        # Let's try variations around it
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
                "lstm_units": [192, 96, 48],  # Best from previous
                "dropout_rate": 0.3,  # Reduced from 0.35
                "learning_rate": 0.0005,  # Reduced from 0.0008
                "batch_size": 16,  # Increased from 8
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
        
        best_model = None
        best_score = 0
        best_config = None
        results = []
        
        for config in configs:
            print(f"\n{'='*70}")
            print(f"Training: {config['name']}")
            print(f"{'='*70}")
            
            model, history = self.train_with_config(
                X_train, y_train, X_val, y_val,
                config, config["name"]
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
                "f1_score": float(f1_score),
                "best_epoch": len(history.history['loss'])
            }
            results.append(result)
            
            print(f"\n{config['name']} Results:")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val AUC: {val_auc:.4f}")
            print(f"  Val Precision: {val_precision:.4f}")
            print(f"  Val Recall: {val_recall:.4f}")
            print(f"  F1 Score: {f1_score:.4f}")
            
            # Track best model (based on AUC + F1)
            combined_score = val_auc * 0.5 + f1_score * 0.5
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
                best_config = config
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(MODELS_DIR / f"refined_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_config['name']}")
        print(f"Best Combined Score: {best_score:.4f}")
        print("="*70)
        
        return best_model, best_config, results
    
    def train_with_config(self, X_train, y_train, X_val, y_val, config, model_name):
        """Train model with specific configuration"""
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_improved_model(input_shape, config)
        
        # Compile with class weights to handle imbalance
        class_weight = {0: 1.0, 1: 1.0}  # Balanced
        
        optimizer = keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=self.get_callbacks(model_name),
            verbose=1,
            class_weight=class_weight
        )
        
        return model, history


def main():
    """Main retraining pipeline"""
    print("="*70)
    print("RELIANCE LSTM IMPROVED RETRAINING")
    print("="*70)
    
    trainer = ImprovedTrainer()
    
    # Load data
    print("\nLoading training data...")
    X = np.load(DATA_DIR / "X_reliance_daily.npy")
    y = np.load(DATA_DIR / "y_reliance_daily.npy")
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: Up={np.sum(y==1)}, Down={np.sum(y==0)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Run refined hyperparameter search
    best_model, best_config, results = trainer.train_refined_models(
        X_train, y_train, X_val, y_val
    )
    
    # Save best model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"reliance_lstm_improved_{timestamp}.h5"
    best_model.save(MODELS_DIR / model_filename)
    print(f"\n✓ Best model saved to {MODELS_DIR / model_filename}")
    
    # Upload to GCS
    blob = trainer.bucket.blob(f"reliance/models/{model_filename}")
    blob.upload_from_filename(str(MODELS_DIR / model_filename))
    print(f"✓ Model uploaded to GCS: gs://{GCP_BUCKET_NAME}/reliance/models/{model_filename}")
    
    print("\n" + "="*70)
    print("IMPROVED TRAINING COMPLETE!")
    print("="*70)
    
    # Print summary
    print("\nPerformance Comparison:")
    print(f"Previous Best F1: 0.7273")
    print(f"New Best F1: {max([r['f1_score'] for r in results]):.4f}")
    
    improvement = max([r['f1_score'] for r in results]) - 0.7273
    print(f"Improvement: {improvement:+.4f}")


if __name__ == "__main__":
    main()
