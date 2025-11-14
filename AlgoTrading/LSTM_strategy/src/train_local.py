"""
Local training script for LSTM model
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import LSTM_CONFIG, DATA_DIR, MODELS_DIR
from src.lstm_model import LSTMTradingModel


def load_data():
    """Load prepared sequences"""
    X = np.load(DATA_DIR / "X_sequences.npy")
    y = np.load(DATA_DIR / "y_targets.npy")
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    """Train LSTM model"""
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = LSTMTradingModel(input_shape)
    model = lstm_model.build_model()
    
    print("\nModel Architecture:")
    lstm_model.summary()
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_CONFIG["epochs"],
        batch_size=LSTM_CONFIG["batch_size"],
        callbacks=lstm_model.get_callbacks("lstm_trading"),
        verbose=1
    )
    
    return model, history


def main():
    # Load data
    print("Loading data...")
    X, y = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Save final model
    model.save(MODELS_DIR / "lstm_trading_final.h5")
    print(f"\nModel saved to {MODELS_DIR / 'lstm_trading_final.h5'}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODELS_DIR / "training_history.csv", index=False)
    print(f"Training history saved")


if __name__ == "__main__":
    main()
