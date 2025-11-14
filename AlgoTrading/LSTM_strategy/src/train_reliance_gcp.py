"""
Train LSTM model on RELIANCE daily data using Google Cloud Platform
With hyperparameter tuning and model refinement
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

sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME,
    DATA_DIR, MODELS_DIR, LSTM_CONFIG
)

# Set credentials
key_path = Path(__file__).parent.parent / "lstm-trading-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)


class RelianceLSTMTrainer:
    def __init__(self):
        self.storage_client = storage.Client(project=GCP_PROJECT_ID)
        self.bucket = self.storage_client.bucket(GCP_BUCKET_NAME)
        
    def upload_data_to_gcs(self):
        """Upload training data to Google Cloud Storage"""
        print("Uploading data to GCS...")
        
        files_to_upload = [
            ("X_reliance_daily.npy", "reliance/X_reliance_daily.npy"),
            ("y_reliance_daily.npy", "reliance/y_reliance_daily.npy"),
            ("scaler_reliance_daily.pkl", "reliance/scaler_reliance_daily.pkl"),
        ]
        
        for local_file, gcs_path in files_to_upload:
            local_path = DATA_DIR / local_file
            if local_path.exists():
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))
                print(f"  ✓ Uploaded {local_file}")
        
        print("Data upload complete!")
    
    def build_refined_model(self, input_shape, config):
        """Build refined LSTM model with attention and regularization"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # First LSTM block
        x = layers.LSTM(
            config["lstm_units"][0],
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.001),
            name="lstm_1"
        )(inputs)
        x = layers.Dropout(config["dropout_rate"])(x)
        x = layers.BatchNormalization()(x)
        
        # Second LSTM block
        x = layers.LSTM(
            config["lstm_units"][1],
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.001),
            name="lstm_2"
        )(x)
        x = layers.Dropout(config["dropout_rate"])(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(config["lstm_units"][1])(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        
        # Third LSTM block
        x = layers.LSTM(
            config["lstm_units"][2],
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.001),
            name="lstm_3"
        )(x)
        x = layers.Dropout(config["dropout_rate"])(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Reliance_Refined')
        
        return model
    
    def get_callbacks(self, model_name="reliance_lstm"):
        """Get training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(MODELS_DIR / "logs"),
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train_with_config(self, X_train, y_train, X_val, y_val, config, model_name):
        """Train model with specific configuration"""
        print(f"\nTraining with config: {config['name']}")
        print(f"  LSTM units: {config['lstm_units']}")
        print(f"  Dropout: {config['dropout_rate']}")
        print(f"  Learning rate: {config['learning_rate']}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_refined_model(input_shape, config)
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=self.get_callbacks(model_name),
            verbose=1,
            class_weight={0: 1.0, 1: 1.0}  # Balanced
        )
        
        return model, history
    
    def hyperparameter_search(self, X_train, y_train, X_val, y_val):
        """Search for best hyperparameters"""
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH")
        print("="*70)
        
        configs = [
            {
                "name": "config_1_baseline",
                "lstm_units": [128, 64, 32],
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 150
            },
            {
                "name": "config_2_deeper",
                "lstm_units": [256, 128, 64],
                "dropout_rate": 0.4,
                "learning_rate": 0.0005,
                "batch_size": 16,
                "epochs": 150
            },
            {
                "name": "config_3_wider",
                "lstm_units": [192, 96, 48],
                "dropout_rate": 0.35,
                "learning_rate": 0.0008,
                "batch_size": 8,
                "epochs": 150
            }
        ]
        
        best_model = None
        best_score = 0
        best_config = None
        results = []
        
        for config in configs:
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
                "f1_score": float(f1_score)
            }
            results.append(result)
            
            print(f"\n{config['name']} Results:")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val AUC: {val_auc:.4f}")
            print(f"  F1 Score: {f1_score:.4f}")
            
            # Track best model (based on F1 score)
            if f1_score > best_score:
                best_score = f1_score
                best_model = model
                best_config = config
        
        # Save results
        with open(MODELS_DIR / "hyperparameter_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_config['name']}")
        print(f"Best F1 Score: {best_score:.4f}")
        print("="*70)
        
        return best_model, best_config, results


def main():
    """Main training pipeline"""
    print("="*70)
    print("RELIANCE LSTM TRAINING ON GCP")
    print("="*70)
    
    trainer = RelianceLSTMTrainer()
    
    # Upload data to GCS
    trainer.upload_data_to_gcs()
    
    # Load data locally for training
    print("\nLoading training data...")
    X = np.load(DATA_DIR / "X_reliance_daily.npy")
    y = np.load(DATA_DIR / "y_reliance_daily.npy")
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Run hyperparameter search
    best_model, best_config, results = trainer.hyperparameter_search(
        X_train, y_train, X_val, y_val
    )
    
    # Save best model
    best_model.save(MODELS_DIR / "reliance_lstm_best_final.h5")
    print(f"\n✓ Best model saved to {MODELS_DIR / 'reliance_lstm_best_final.h5'}")
    
    # Upload best model to GCS
    blob = trainer.bucket.blob("reliance/models/reliance_lstm_best_final.h5")
    blob.upload_from_filename(str(MODELS_DIR / "reliance_lstm_best_final.h5"))
    print(f"✓ Model uploaded to GCS: gs://{GCP_BUCKET_NAME}/reliance/models/")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
