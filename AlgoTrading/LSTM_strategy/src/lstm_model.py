"""
LSTM model architecture for trading strategy
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import LSTM_CONFIG, MODELS_DIR


class LSTMTradingModel:
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or LSTM_CONFIG
        self.model = None
        
    def build_model(self):
        """Build LSTM model with attention mechanism"""
        inputs = layers.Input(shape=self.input_shape)
        
        # First LSTM layer
        x = layers.LSTM(
            self.config["lstm_units"][0],
            return_sequences=True,
            name="lstm_1"
        )(inputs)
        x = layers.Dropout(self.config["dropout_rate"])(x)
        x = layers.BatchNormalization()(x)
        
        # Second LSTM layer
        x = layers.LSTM(
            self.config["lstm_units"][1],
            return_sequences=True,
            name="lstm_2"
        )(x)
        x = layers.Dropout(self.config["dropout_rate"])(x)
        x = layers.BatchNormalization()(x)
        
        # Third LSTM layer
        x = layers.LSTM(
            self.config["lstm_units"][2],
            return_sequences=False,
            name="lstm_3"
        )(x)
        x = layers.Dropout(self.config["dropout_rate"])(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Trading_Model')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def get_callbacks(self, model_name="lstm_model"):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
